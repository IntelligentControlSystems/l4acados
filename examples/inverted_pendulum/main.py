import sys, os
import argparse

sys.path += ["../../external/"]

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
import copy
import re

# zoRO imports
import zero_order_gpmpc
from zero_order_gpmpc.controllers import (
    ZoroAcados,
    ZoroAcadosCustomUpdate,
    ZeroOrderGPMPC,
)
from inverted_pendulum_model_acados import (
    export_simplependulum_ode_model,
    export_ocp_nominal,
)
from utils import base_plot, add_plot_trajectory, EllipsoidTubeData2D

# gpytorch_utils
from gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_acados,
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
from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
    GPyTorchResidualModel,
)


def setup_sim_from_ocp(ocp):
    # integrator for nominal model
    sim = AcadosSim()

    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values

    for opt_name in dir(ocp.solver_options):
        if (
            opt_name in dir(sim.solver_options)
            and re.search(r"__.*?__", opt_name) is None
        ):
            set_value = getattr(ocp.solver_options, opt_name)

            if opt_name == "sim_method_jac_reuse":
                set_value = array_to_int(set_value)

            print(f"Setting {opt_name} to {set_value}")
            setattr(sim.solver_options, opt_name, set_value)

    sim.solver_options.T = ocp.solver_options.Tsim
    sim.solver_options.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim.solver_options.newton_tol = ocp.solver_options.sim_method_newton_tol
    sim.solver_options.num_stages = array_to_int(
        ocp.solver_options.sim_method_num_stages
    )
    sim.solver_options.num_steps = array_to_int(ocp.solver_options.sim_method_num_steps)

    return sim


def array_to_int(arr):
    value = copy.deepcopy(arr)
    if type(value) is list or type(value) is np.ndarray:
        assert all(value == value[0])
        value = value[0]

    return int(value)


def get_solution(ocp_solver, x0, N, nx, nu):
    # get initial values
    X = np.zeros((N + 1, nx))
    U = np.zeros((N, nu))

    # xcurrent = x0
    X[0, :] = x0

    # solve
    status = ocp_solver.solve()

    if status != 0:
        raise Exception("acados ocp_solver returned status {}. Exiting.".format(status))

    # get data
    for i in range(N):
        X[i, :] = ocp_solver.get(i, "x")
        U[i, :] = ocp_solver.get(i, "u")

    X[N, :] = ocp_solver.get(N, "x")
    return X, U


def simulate_solution(sim_solver, x0, N, nx, nu, U):
    # get initial values
    X = np.zeros((N + 1, nx))

    # xcurrent = x0
    X[0, :] = x0

    # simulate
    for i in range(N):
        sim_solver.set("x", X[i, :])
        sim_solver.set("u", U[i, :])
        status = sim_solver.solve()
        if status != 0:
            raise Exception(
                "acados sim_solver returned status {}. Exiting.".format(status)
            )
        X[i + 1, :] = sim_solver.get("x")

    return X


def get_gp_model(ocp_solver, sim_solver, sim_solver_actual, x0, Sigma_W):
    random_seed = 123
    N_sim_per_x0 = 1
    N_x0 = 10
    x0_rand_scale = 0.1

    x_train, x0_arr = generate_train_inputs_acados(
        ocp_solver,
        x0,
        N_sim_per_x0,
        N_x0,
        random_seed=random_seed,
        x0_rand_scale=x0_rand_scale,
    )

    y_train = generate_train_outputs_at_inputs(
        x_train, sim_solver, sim_solver_actual, Sigma_W
    )


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description="A foo that bars")

    parser.add_argument("-solver", type=str, default="zero_order_gpmpc")
    args = parser.parse_args()

    solver_name = args.solver

    # build C code again?
    build_c_code = True

    # discretization
    N = 30
    T = 5
    dT = T / N

    # constraints
    x0 = np.array([np.pi, 0])
    nx = 2
    nu = 1

    # uncertainty
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

    # GP training params
    random_seed = 123
    N_sim_per_x0 = 1
    N_x0 = 10
    x0_rand_scale = 0.1

    # nominal OCP
    ocp_init_model_name = "simplependulum_ode_init"
    ocp_init = export_ocp_nominal(N, T, model_name=ocp_init_model_name)
    ocp_init.solver_options.nlp_solver_type = "SQP"
    acados_ocp_init_solver = AcadosOcpSolver(
        ocp_init, json_file="acados_ocp_" + ocp_init_model_name + ".json"
    )
    X_init, U_init = get_solution(acados_ocp_init_solver, x0, N, nx, nu)

    # nominal sim
    sim = setup_sim_from_ocp(ocp_init)
    acados_integrator = AcadosSimSolver(
        sim, json_file="acados_sim_" + sim.model.name + ".json"
    )

    # actual sim
    model_actual = export_simplependulum_ode_model(
        model_name=sim.model.name + "_actual", add_residual_dynamics=True
    )
    sim_actual = setup_sim_from_ocp(ocp_init)
    sim_actual.model = model_actual
    acados_integrator_actual = AcadosSimSolver(
        sim_actual, json_file="acados_sim_" + sim_actual.model.name + ".json"
    )

    # GP training
    x_train, x0_arr = generate_train_inputs_acados(
        acados_ocp_init_solver,
        x0,
        N_sim_per_x0,
        N_x0,
        random_seed=random_seed,
        x0_rand_scale=x0_rand_scale,
    )

    y_train = generate_train_outputs_at_inputs(
        x_train, acados_integrator, acados_integrator_actual, Sigma_W
    )

    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    nout = y_train.shape[1]

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout)
    gp_model = BatchIndependentMultitaskGPModel(
        x_train_tensor, y_train_tensor, likelihood
    )

    training_iterations = 200
    rng_seed = 456

    gp_model, likelihood = train_gp_model(
        gp_model, torch_seed=rng_seed, training_iterations=training_iterations
    )

    gp_model.eval()
    likelihood.eval()

    residual_model = GPyTorchResidualModel(gp_model)

    # compare different solvers
    if solver_name == "zero_order_gpmpc":
        zoro_solver = ZeroOrderGPMPC(
            ocp_init,
            sim,
            prob_x,
            Sigma_x0,
            Sigma_W,
            h_tightening_idx=[0],
            gp_model=residual_model,
            use_cython=False,
            path_json_ocp=f"{solver_name}_ocp_solver_config.json",
            path_json_sim=f"{solver_name}_sim_solver_config.json",
            build_c_code=True,
        )
    elif solver_name == "zoro_acados_custom_update":
        zoro_solver = ZoroAcadosCustomUpdate(
            ocp_init,
            sim,
            prob_x,
            Sigma_x0,
            Sigma_W,
            h_tightening_idx=[0],
            gp_model=gp_model,
            use_cython=False,
            path_json_ocp=f"{solver_name}_ocp_solver_config_cupdate.json",
            path_json_sim=f"{solver_name}_sim_solver_config_cupdate.json",
        )
    elif solver_name == "zoro_acados":
        # zoro_solver_nogp = ZoroAcados(ocp_zoro_nogp, sim, prob_x, Sigma_x0, Sigma_W+Sigma_GP_prior)
        from zero_order_gpmpc.controllers.zoro_acados_utils import (
            tighten_model_constraints,
        )

        # tighten constraints
        idh_tight = np.array([0])  # lower constraint on theta (theta >= 0)

        (
            ocp_model_tightened,
            h_jac_x_fun,
            h_tighten_fun,
            h_tighten_jac_x_fun,
            h_tighten_jac_sig_fun,
        ) = tighten_model_constraints(ocp_init, idh_tight, prob_x)

        ocp_init.model = ocp_model_tightened
        ocp_init.dims.nh = ocp_model_tightened.con_h_expr.shape[0]
        ocp_init.dims.np = ocp_model_tightened.p.shape[0]
        ocp_init.parameter_values = np.zeros((ocp_init.dims.np,))

        zoro_solver = ZoroAcados(
            ocp_init,
            sim,
            prob_x,
            Sigma_x0,
            Sigma_W,
            h_tightening_jac_sig_fun=h_tighten_jac_sig_fun,
            gp_model=gp_model,
            path_json_ocp=f"{solver_name}_ocp_solver_config_cupdate.json",
            path_json_sim=f"{solver_name}_sim_solver_config_cupdate.json",
        )

    # initializte with nominal solution
    for i in range(N):
        zoro_solver.ocp_solver.set(i, "x", X_init[i, :])
        zoro_solver.ocp_solver.set(i, "u", U_init[i, :])
    zoro_solver.ocp_solver.set(N, "x", X_init[N, :])

    zoro_solver.solve()
    X, U, P = zoro_solver.get_solution()

    # save data
    data_dict = {
        "X": X,
        "U": U,
        "P": P,
    }
    np.save(f"solve_data_{solver_name}.npy", data_dict)
