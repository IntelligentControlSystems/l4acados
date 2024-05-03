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


def setup_ocp(N, T, only_lower_bounds=True):
    ocp = export_ocp_nominal(N, T, only_lower_bounds=only_lower_bounds)
    ocp.solver_options.nlp_solver_type = "SQP"
    return ocp


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
            if opt_name == "sim_method_jac_reuse":
                set_value = int(getattr(ocp.solver_options, opt_name)[0])
            else:
                set_value = getattr(ocp.solver_options, opt_name)
            print(f"Setting {opt_name} to {set_value}")
            setattr(sim.solver_options, opt_name, set_value)

    sim.solver_options.T = ocp.solver_options.Tsim
    sim.solver_options.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim.solver_options.newton_tol = ocp.solver_options.sim_method_newton_tol

    assert all(
        ocp.solver_options.sim_method_num_stages
        == ocp.solver_options.sim_method_num_stages[0]
    )
    sim.solver_options.num_stages = int(ocp.solver_options.sim_method_num_stages[0])
    assert all(
        ocp.solver_options.sim_method_num_steps
        == ocp.solver_options.sim_method_num_steps[0]
    )
    sim.solver_options.num_steps = int(ocp.solver_options.sim_method_num_steps[0])

    return sim


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
