import sys, os

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


if __name__ == "__main__":
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
