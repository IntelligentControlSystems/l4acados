import sys, os, shutil
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
    setup_sim_from_ocp,
)
from inverted_pendulum_model_acados import (
    export_simplependulum_ode_model,
    export_ocp_nominal,
)
from utils import *

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


def solve_pendulum(solver_name):
    # clear existing solver data
    os.system("rm -r c_generated_code/*")
    os.system("rm ./acados_*.json")
    os.system("rm ./jit_*.c")
    os.system("rm ./tmp_casadi_*")

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
    ocp_init_model_name = "simplependulum_ode_ocp_init"
    ocp_init = export_ocp_nominal(N, T, model_name=ocp_init_model_name)
    ocp_init.solver_options.nlp_solver_type = "SQP"
    acados_ocp_init_solver = AcadosOcpSolver(
        ocp_init, json_file="acados_ocp_" + ocp_init_model_name + ".json"
    )
    X_init, U_init = get_solution(acados_ocp_init_solver, x0)

    # actual sim
    sim_model_name = "simplependulum_ode_sim"
    model_actual = export_simplependulum_ode_model(
        model_name=sim_model_name + "_actual", add_residual_dynamics=True
    )
    sim_actual = setup_sim_from_ocp(ocp_init)
    sim_actual.model = model_actual
    acados_integrator_actual = AcadosSimSolver(
        sim_actual, json_file="acados_sim_" + sim_actual.model.name + ".json"
    )

    if solver_name == "zoro_acados":
        # modify OCP for backwards compatibility with zoro_acados
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
        ) = tighten_model_constraints(ocp_init.model, idh_tight, prob_x)

        ocp_init.model = ocp_model_tightened
        ocp_init.dims.nh = ocp_model_tightened.con_h_expr.shape[0]
        ocp_init.dims.np = ocp_model_tightened.p.shape[0]
        ocp_init.parameter_values = np.zeros((ocp_init.dims.np,))

    sim = setup_sim_from_ocp(ocp_init)
    acados_integrator = AcadosSimSolver(
        sim, json_file="acados_sim_" + sim.model.name + ".json"
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

        init_ocp_solver(zoro_solver.ocp_solver, X_init, U_init)

        zoro_solver.solve()
        X, U = zoro_solver.get_solution()
        P_arr = zoro_solver.covariances_array

        P = []
        for i in range(N + 1):
            P.append(
                np.array(
                    [
                        [P_arr[3 * i], P_arr[3 * i + 2]],
                        [P_arr[3 * i + 2], P_arr[3 * i + 1]],
                    ]
                )
            )
        P = np.array(P)

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
            path_json_ocp=f"{solver_name}_ocp_solver_config.json",
            path_json_sim=f"{solver_name}_sim_solver_config.json",
        )

        init_ocp_solver(zoro_solver.ocp_solver, X_init, U_init)

        zoro_solver.solve()
        X, U, P_arr = zoro_solver.get_solution()

        P = []
        for i in range(N + 1):
            P.append(
                np.array(
                    [
                        [P_arr[3 * i], P_arr[3 * i + 2]],
                        [P_arr[3 * i + 2], P_arr[3 * i + 1]],
                    ]
                )
            )
        P = np.array(P)

    elif solver_name == "zoro_acados":
        zoro_solver = ZoroAcados(
            ocp_init,
            sim,
            prob_x,
            Sigma_x0,
            Sigma_W,
            h_tightening_jac_sig_fun=h_tighten_jac_sig_fun,
            gp_model=gp_model,
            use_cython=False,
            path_json_ocp=f"{solver_name}_ocp_solver_config.json",
            path_json_sim=f"{solver_name}_sim_solver_config.json",
        )

        init_ocp_solver(zoro_solver.ocp_solver, X_init, U_init)

        zoro_solver.solve()
        X, U, P = zoro_solver.get_solution()
        P = np.array(P)

    else:
        raise ValueError(f"Unknown solver name: {solver_name}")

    return X, U, P


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(
        description="Solve the pendulum OCP with a given solver."
    )

    parser.add_argument("-solver", type=str, default="zero_order_gpmpc")
    args = parser.parse_args()

    solver_name = args.solver

    X, U, P = solve_pendulum(solver_name)

    # save data
    data_dict = {
        "X": X,
        "U": U,
        "P": P,
    }
    np.save(f"solve_data_{solver_name}.npy", data_dict)
