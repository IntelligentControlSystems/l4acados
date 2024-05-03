# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3.9.13 ('zero-order-gp-mpc-code-2CX1fffa')
#     language: python
#     name: python3
# ---

# +
import sys, os

sys.path += ["../../"]

import numpy as np
from scipy.stats import norm
import casadi as cas
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimSolver,
    AcadosOcpSolver,
    AcadosOcpOptions,
)
import matplotlib.pyplot as plt
import torch
import gpytorch

# zoRO imports
from zero_order_gpmpc.controllers import ZoroAcados
from inverted_pendulum_model_acados import (
    export_simplependulum_ode_model,
    export_ocp_nominal,
)
from utils import base_plot, add_plot_trajectory, EllipsoidTubeData2D

# gpytorch_utils
from gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_zoro,
    generate_train_outputs_at_inputs,
    train_gp_model,
)
from gpytorch_utils.gp_utils import (
    gp_data_from_model_and_path,
    gp_derivative_data_from_model_and_path,
    plot_gp_data,
    generate_grid_points,
)
from zero_order_gpmpc.models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
)

# -

# ## Define model parameters
#
# We model the inverted pendulum
#
# $$
# \dot{x} = f(x,u) = \begin{bmatrix} \dot{\theta} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} \dot{\theta} \\ -\sin(\theta) + u \end{bmatrix},
# $$
#
# which is to be controlled from the hanging-down resting position, $(\theta_0, \dot{\theta}_0) = (\pi, 0)$, to the upright position ($(\theta_r, \dot{\theta}_r) = (0,0)$), subject to the constraints that overshoot should be avoided, i.e.,
#
# $$
# \theta_{lb} \leq \theta \leq \theta_{ub}.
# $$
#
# The model setup and controller definition can be found in the functions `export_simplependulum_ode_model()`, `export_ocp_nominal()` in the `inverted_pendulum_model_acados.py` file.

# +
# discretization
N = 30
T = 5
dT = T / N

# constraints
x0 = np.array([np.pi, 0])
nx = 2
nu = 1

# -

# ## Set up nominal solver

# +
ocp_init = export_ocp_nominal(N, T, only_lower_bounds=True)
ocp_init.solver_options.nlp_solver_type = "SQP"

acados_ocp_init_solver = AcadosOcpSolver(
    ocp_init, json_file="acados_ocp_init_simplependulum_ode.json"
)
# -

# ## Open-loop planning with nominal solver

# +
# get initial values
X_init = np.zeros((N + 1, nx))
U_init = np.zeros((N, nu))

# xcurrent = x0
X_init[0, :] = x0

# solve
status_init = acados_ocp_init_solver.solve()

if status_init != 0:
    raise Exception(
        "acados acados_ocp_solver returned status {}. Exiting.".format(status_init)
    )

# get data
for i in range(N):
    X_init[i, :] = acados_ocp_init_solver.get(i, "x")
    U_init[i, :] = acados_ocp_init_solver.get(i, "u")

X_init[N, :] = acados_ocp_init_solver.get(N, "x")
# -

# ## Simulation results (nominal)

# +
lb_theta = -ocp_init.constraints.lh[0]
fig, ax = base_plot(lb_theta=lb_theta)

plot_data_nom = EllipsoidTubeData2D(center_data=X_init, ellipsoid_data=None)
add_plot_trajectory(ax, plot_data_nom, prob_tighten=None)
# -

# ## Robustify simulation
#
# Now we consider the inverted pendulum subject to process noise, i.e., after discretization,
#
# $$
# x_{k+1} = F(x_k, u_k) + w_k,
# $$
#
# where $w_k \sim \mathcal{N}(0, \Sigma_w)$ denotes process noise and we jhave a potentially nois initial condition $x_0 \sim \mathcal{N}(0, \Sigma_{x0})$ (defined below). Constraints should now be satisfied with satisfaction probability $p_x$, i.e.,
#
# $$
# \mathrm{Pr}(h(x,u) \leq 0) \geq p_x.
# $$
#
# Note that currently zoRO only accepts upper bounds, i.e., we need to reformulate the nominal constraints $\theta \leq \theta_{ub}$, $-\theta \leq -\theta_{lb}$.
# Afterwards, we consider the tightened constraints
#
# $$
# h(x,u) + \alpha \sqrt{C(x,u) P C(x,u)} \leq 0,
# $$
#
# see problem definition https://www.sciencedirect.com/science/article/pii/S0947358023000912, eq. (9).

# +
# satisfaction probability for chance constraints
prob_x = 0.9
prob_tighten = norm.ppf(prob_x)

# noise
# uncertainty dynamics
sigma_theta = (0.0001 / 360.0) * 2 * np.pi
sigma_omega = (0.0001 / 360.0) * 2 * np.pi
w_theta = 0.03
w_omega = 0.03
Sigma_x0 = np.array([[sigma_theta**2, 0], [0, sigma_omega**2]])
Sigma_W = np.array([[w_theta**2, 0], [0, w_omega**2]])
# -

# ### Nominal OCP with lower bounds

# +
# without gp model
ocp_zoro_nogp = export_ocp_nominal(N, T, only_lower_bounds=True)

# OCP is discretized inside zoRO solver, so we need to choose DISCRETE integration here
ocp_zoro_nogp.solver_options.integrator_type = "DISCRETE"
ocp_zoro_nogp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp_zoro_nogp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp_zoro_nogp.solver_options.nlp_solver_type = "SQP_RTI"
# -

# ### Nominal model with lower bounds
#
# The tightenings $\sqrt{\ldots}$ can thereby be automatically defined using the `tighten_model_constraints()` function.

# +
# zoro_solver_nogp = ZoroAcados(ocp_zoro_nogp, sim, prob_x, Sigma_x0, Sigma_W+Sigma_GP_prior)
from zero_order_gpmpc.controllers.zoro_acados_utils import (
    generate_h_tightening_funs_SX,
    only_upper_bounds_expr,
    tighten_model_constraints,
)

# make constraints one-sided
ocp_model = export_simplependulum_ode_model(only_lower_bounds=True)

# tighten constraints
idh_tight = np.array([0])  # lower constraint on theta (theta >= 0)

(
    ocp_model_tightened,
    h_jac_x_fun,
    h_tighten_fun,
    h_tighten_jac_x_fun,
    h_tighten_jac_sig_fun,
) = tighten_model_constraints(ocp_model, idh_tight, prob_x)

ocp_zoro_nogp.model = ocp_model_tightened
ocp_zoro_nogp.dims.nh = ocp_model_tightened.con_h_expr.shape[0]
ocp_zoro_nogp.dims.np = ocp_model_tightened.p.shape[0]
ocp_zoro_nogp.parameter_values = np.zeros((ocp_zoro_nogp.dims.np,))
# -

# ## Simulator object
#
# To automatically discretize the model (and obtain sensitivities of the discrete-time model) within the zero-order implementation, we create the `AcadosSimSolver` object to pass to the solver.

# +
# integrator for nominal model
sim = AcadosSim()

sim.model = ocp_zoro_nogp.model
sim.parameter_values = ocp_zoro_nogp.parameter_values
sim.solver_options.integrator_type = "ERK"

# set prediction horizon
sim.solver_options.T = dT

# acados_ocp_solver = AcadosOcpSolver(ocp_init, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator = AcadosSimSolver(
    sim, json_file="acados_sim_" + sim.model.name + ".json"
)
# -

# ## Create zero-order solver (without GP)
#
# Everything comes together in the `ZoroAcados` constructor:
#
# - nominal OCP `ocp_zoro_nogp`,
# - simulator `AcadosSim` object `sim`,
# - satisfaction probability `prob_x`,
# - initial uncertainty `Sigma_x0`,
# - process noise `Sigma_W`,
# - Jacobian of tightened constraints w.r.t. covariances $P$, `h_tighten_jac_sig_fun`

# +
# solve with zoRO (no GP model, only process noise)
zoro_solver_nogp = ZoroAcados(
    ocp_zoro_nogp,
    sim,
    prob_x,
    Sigma_x0,
    Sigma_W,
    h_tightening_jac_sig_fun=h_tighten_jac_sig_fun,
)

for i in range(N):
    zoro_solver_nogp.ocp_solver.set(i, "x", X_init[i, :])
    zoro_solver_nogp.ocp_solver.set(i, "u", U_init[i, :])
zoro_solver_nogp.ocp_solver.set(N, "x", X_init[N, :])

zoro_solver_nogp.solve()
X_nogp, U_nogp, P_nogp = zoro_solver_nogp.get_solution()
# -

# ### Plot open-loop prediction
#
# Ellipsoids denote the Gaussian confidence intervals, note that one (terminal) ellipsoid "sticks out" since we have omitted the terminal constraint here.

# +
fig, ax = base_plot(lb_theta=lb_theta)

plot_data_nogp = EllipsoidTubeData2D(
    center_data=X_nogp,
    ellipsoid_data=np.array(P_nogp),
    # ellipsoid_data = None
)
add_plot_trajectory(
    ax, plot_data_nom, color_fun=plt.cm.Blues, prob_tighten=prob_tighten
)
add_plot_trajectory(
    ax, plot_data_nogp, color_fun=plt.cm.Oranges, prob_tighten=prob_tighten
)
# -

# # GP training
#
# We use a model with different parameters to emulate the real-world model and obtain some training data. Also create simulator object for real-world model to evaluate our results later (not used in solver).

# +
# generate training data for GP with augmented model

# "real model"
model_actual = export_simplependulum_ode_model()
model_actual.f_expl_expr = model_actual.f_expl_expr + cas.vertcat(
    cas.DM(0), -0.5 * cas.sin((model_actual.x[0]) ** 2)
)
model_actual.f_impl_expr = model_actual.xdot - model_actual.f_expl_expr
model_actual.name = model_actual.name + "_actual"

# acados integrator
sim_actual = AcadosSim()
sim_actual.model = model_actual
sim_actual.solver_options.integrator_type = "ERK"

# set prediction horizon
sim_actual.solver_options.T = dT

# acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator_actual = AcadosSimSolver(
    sim_actual, json_file="acados_sim_" + model_actual.name + ".json"
)
# -

# ## Generate training data
#
# We generate training data (one-step ahead residuals `y_train` for starting point `x_train`) here by running robustified (cautious) solver without GP.

# +
random_seed = 123
N_sim_per_x0 = 1
N_x0 = 10
x0_rand_scale = 0.1

x_train, x0_arr = generate_train_inputs_zoro(
    zoro_solver_nogp,
    x0,
    N_sim_per_x0,
    N_x0,
    random_seed=random_seed,
    x0_rand_scale=x0_rand_scale,
)

y_train = generate_train_outputs_at_inputs(
    x_train, acados_integrator, acados_integrator_actual, Sigma_W
)
# -

# ## Hyper-parameter training for GP model
#
# Optimize hyper-parameters of GP model (kernel function parameters, ...)

# +
x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)
nout = y_train.shape[1]

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout)
gp_model = BatchIndependentMultitaskGPModel(x_train_tensor, y_train_tensor, likelihood)

# +
training_iterations = 500
rng_seed = 456

gp_model, likelihood = train_gp_model(
    gp_model, torch_seed=rng_seed, training_iterations=training_iterations
)

# EVAL MODE
gp_model.eval()
likelihood.eval()
# -

# ## Plot GP predictions
#
# We plot GP predictions along the predicted trajectory of the robustified solver by projecting the multivariate plot down to a line.

# +
num_samples = 5
use_likelihood = False

num_points_between_samples = 30
t_lin = np.linspace(0, 1, num_points_between_samples, endpoint=False)

x_plot_waypts = np.hstack((X_nogp[1:, :], U_nogp))
x_plot = []
for i in range(x_plot_waypts.shape[0] - 1):
    x_plot += [
        x_plot_waypts[i, :] + (x_plot_waypts[i + 1, :] - x_plot_waypts[i, :]) * t
        for t in t_lin
    ]
x_plot = np.vstack(x_plot)

gp_data = gp_data_from_model_and_path(
    gp_model, likelihood, x_plot, num_samples=num_samples, use_likelihood=use_likelihood
)
plot_gp_data([gp_data], marker_size_lim=[1, 15])
# -

# We can also plot the derivative of the GP. Note that the projected Jacobian is not smooth since our path is not smooth either (jump projection direction = jump in Jacobian); however, the actual Jacobian should be smooth here (squared exponential kernel).

gp_derivative_data = gp_derivative_data_from_model_and_path(
    gp_model, likelihood, x_plot, num_samples=0
)
plot_gp_data([gp_derivative_data], marker_size_lim=[5, 20], plot_train_data=False)

# Compare with plotting along a slice of the dimension. Since we generated training data along the path of the robustified controller, the GP looks pretty untrained along a slice of the coordinates.

# +
# plot along axis
x_dim_lims = np.array([[0, np.pi], [-2, 1], [-2, 2]])
x_dim_slice = np.array([1 * np.pi, 0, 0])
x_dim_plot = 2
x_grid = generate_grid_points(x_dim_lims, x_dim_slice, x_dim_plot, num_points=800)

gp_grid_data = gp_data_from_model_and_path(
    gp_model, likelihood, x_grid, num_samples=num_samples, use_likelihood=use_likelihood
)
fig, ax = plot_gp_data([gp_grid_data], marker_size_lim=[5, 50])

y_lim_0 = ax[0].get_ylim()
y_lim_1 = ax[1].get_ylim()
# -

# Jacobian... not much going on away from the data points (this is good!)

# +
gp_derivative_grid_data = gp_derivative_data_from_model_and_path(
    gp_model, likelihood, x_grid, num_samples=0
)
fig, ax = plot_gp_data(
    [gp_derivative_grid_data], marker_size_lim=[5, 50], plot_train_data=False
)

ax[0].set_ylim(*y_lim_0)
ax[1].set_ylim(*y_lim_1)
plt.draw()
# -

# # Zero-Order GP-MPC
#
# We can add the GP model to the solver by simply adding it as an argument to the `ZoroAcados` function. Therefore we copy (important!) the robustified controller and then instantiate another solver object.

# +
from copy import deepcopy

ocp_zoro = deepcopy(ocp_zoro_nogp)

zoro_solver = ZoroAcados(
    ocp_zoro,
    sim,
    prob_x,
    Sigma_x0,
    Sigma_W,
    h_tightening_jac_sig_fun=h_tighten_jac_sig_fun,
    gp_model=gp_model,
)

for i in range(N):
    zoro_solver.ocp_solver.set(i, "x", X_init[i, :])
    zoro_solver.ocp_solver.set(i, "u", U_init[i, :])
zoro_solver.ocp_solver.set(N, "x", X_init[N, :])

zoro_solver.solve()
X, U, P = zoro_solver.get_solution()
# -

# ## Trained GP vs. no GP
#
# Plotting trained GP against no-GP, we see that the GP-MPC controller still has higher uncertainties than the robustified one. This is because the GP covariance, albeit being small where data points are, is *added* to the process noise, i.e., we make the controller even more cautious.

# +
fig, ax = base_plot(lb_theta=lb_theta)

plot_data_gp = EllipsoidTubeData2D(
    center_data=X,
    ellipsoid_data=np.array(P),
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data_nogp, color_fun=plt.cm.Oranges)
add_plot_trajectory(ax, plot_data_gp, color_fun=plt.cm.Reds)
# -

# We can get a more insightful comparison by comparing the *conditioned* GP (i.e., with optimized hyper-parameters and data), to the *unconditioned* GP (i.e., with the same, optimized hyper-parameters and no data).
#
# As you can see, the hyper-parameter optimization also provides us with an estimate of the measurement noise for the data points, which corersponds to added process noise in our model (epistemic uncertainty).

# +
ocp_zoro_gpprior = deepcopy(ocp_zoro_nogp)

gp_model.eval()
y_test = torch.Tensor(np.array([[1, 1, 1]]))
Sigma_GP_prior = gp_model.covar_module(y_test).numpy()

Sigma_GP_prior = np.diag(Sigma_GP_prior.flatten())
Sigma_GP_prior
# -

Sigma_W

# Especially in the velocity component, this really makes a difference!

Sigma_W + Sigma_GP_prior

# Now we solve again with *unconditioned* GP and compare...

# +
# zoro_solver_gpprior = ZoroAcados(ocp_zoro_gpprior, sim, prob_x, Sigma_x0, Sigma_W+Sigma_GP_prior)
zoro_solver_gpprior = ZoroAcados(
    ocp_zoro_gpprior,
    sim,
    prob_x,
    Sigma_x0,
    Sigma_W + Sigma_GP_prior,
    h_tightening_jac_sig_fun=h_tighten_jac_sig_fun,
)

for i in range(N):
    zoro_solver_gpprior.ocp_solver.set(i, "x", X_init[i, :])
    zoro_solver_gpprior.ocp_solver.set(i, "u", U_init[i, :])
zoro_solver_gpprior.ocp_solver.set(N, "x", X_init[N, :])

# +
zoro_solver_gpprior.solve()

X_gpprior, U_gpprior, P_gpprior = zoro_solver_gpprior.get_solution()
# -

# ## Zero-Order GP-MPC with unconditioned GP vs. conditioned GP
#
# Now we see that the training indeed paid off, and reduced uncertainty where we obtained data, i.e., we can go closer to the boundary while still being cautious.

# +

fig, ax = base_plot(lb_theta=lb_theta)

plot_data_gpprior = EllipsoidTubeData2D(
    center_data=X_gpprior,
    ellipsoid_data=np.array(P_gpprior),
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data_gp, color_fun=plt.cm.Reds)
add_plot_trajectory(ax, plot_data_gpprior, color_fun=plt.cm.Blues)
