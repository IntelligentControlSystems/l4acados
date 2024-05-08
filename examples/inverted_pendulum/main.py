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

# + metadata={}
# %load_ext autoreload
# %autoreload 1
# %aimport run_example

# + metadata={}
from run_example import solve_pendulum
from utils import base_plot, EllipsoidTubeData2D, add_plot_trajectory
import matplotlib.pyplot as plt

# -

# ## Inverted pendulum model
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

# + metadata={}
X_zoro_acados, U_zoro_acados, P_zoro_acados = solve_pendulum("zoro_acados")
X_zoro_cupdate, U_zoro_cupdate, P_zoro_cupdate = solve_pendulum(
    "zoro_acados_custom_update"
)
X_zero_order_gpmpc, U_zero_order_gpmpc, P_zero_order_gpmpc = solve_pendulum(
    "zero_order_gpmpc"
)
# -

# ## Plot results

# + metadata={}
lb_theta = 0.0
fig, ax = base_plot(lb_theta=lb_theta)

plot_data_zoro_acados = EllipsoidTubeData2D(
    center_data=X_zoro_acados, ellipsoid_data=P_zoro_acados
)
plot_data_zoro_cupdate = EllipsoidTubeData2D(
    center_data=X_zoro_cupdate, ellipsoid_data=P_zoro_cupdate
)
plot_data_zero_order_gpmpc = EllipsoidTubeData2D(
    center_data=X_zero_order_gpmpc, ellipsoid_data=P_zero_order_gpmpc
)
add_plot_trajectory(ax, plot_data_zoro_acados, color_fun=plt.cm.Purples, linewidth=5)
add_plot_trajectory(ax, plot_data_zoro_cupdate, color_fun=plt.cm.Oranges, linewidth=3)
add_plot_trajectory(ax, plot_data_zero_order_gpmpc, color_fun=plt.cm.Blues, linewidth=1)

plt.title("All controllers should give the same result")

plt.show()
# -
