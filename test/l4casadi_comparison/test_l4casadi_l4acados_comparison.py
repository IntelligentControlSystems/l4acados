import l4casadi as l4c
import numpy as np
import time
import zero_order_gpmpc as zogp
from typing import Optional, Union
import torch
import casadi as cs
from zero_order_gpmpc.controllers.residual_learning_mpc import ResidualLearningMPC
from zero_order_gpmpc.models.pytorch_models import PyTorchResidualModel
from zero_order_gpmpc.models.pytorch_models.pytorch_feature_selector import (
    FeatureSelector,
)
from zero_order_gpmpc.controllers.zoro_acados_utils import setup_sim_from_ocp
import argparse
import os, shutil, re
import subprocess
import matplotlib.pyplot as plt

from l4casadi_comparison.l4casadi_with_acados import (
    run,
    MultiLayerPerceptron,
    DoubleIntegratorWithLearnedDynamics,
    MPC,
    train_nn,
)


def init_l4casadi(
    nn_model: torch.nn.Module,
    N: int,
    device="cpu",
):
    learned_dyn_model = l4c.L4CasADi(
        nn_model,
        name="learned_dyn",
        device=device,
    )

    model = DoubleIntegratorWithLearnedDynamics(learned_dyn_model)

    mpc_obj = MPC(
        model=model.model(),
        N=N,
        external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
        external_shared_lib_name=learned_dyn_model.name,
    )
    solver = mpc_obj.solver

    ocp = mpc_obj.ocp()
    return solver


def init_l4acados(
    nn_model: torch.nn.Module,
    N: int,
    device="cpu",
    use_cython=False,
):
    feature_selector = FeatureSelector([1, 0, 0], device=device)
    residual_model = PyTorchResidualModel(
        nn_model,
        feature_selector,
    )
    B_proj = (1.0 / N) * np.array([[1.0], [1.0]])  # NOTE: Ts = 1.0 hard coded
    model_new = DoubleIntegratorWithLearnedDynamics(None, name="wr_new")
    mpc_obj_nolib = MPC(
        model=model_new.model(),
        N=N,
    )
    solver_nolib = mpc_obj_nolib.solver
    ocp_nolib = mpc_obj_nolib.ocp()
    sim_nolib = setup_sim_from_ocp(ocp_nolib)

    solver_l4acados = ResidualLearningMPC(
        ocp=ocp_nolib,
        B=B_proj,
        residual_model=residual_model,
        use_cython=use_cython,
    )

    return solver_l4acados


def run_timing_experiment(N, solver, solve_call, solve_steps=1e3):
    x = []
    u = []
    xt = np.array([1.0, 0.0])
    T = 1.0
    ts = T / N
    opt_times = []

    for i in range(solve_steps):
        # t = np.linspace(i * ts, (i + 1) * ts, N) # TODO: this ts should be T IMO, maybe with lower frequency?
        t = np.linspace(
            i * T, (i + 1) * T, N
        )  # TODO: this ts should be T IMO, maybe with lower frequency?
        yref = np.sin(0.1 * t + np.pi / 2)
        for t, ref in enumerate(yref):
            solver.set(t, "yref", np.array([ref]))
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        # now = time.time()
        elapsed = solve_call()

        xt = solver.get(1, "x")

        opt_times.append(elapsed)
        x.append(xt)

        print(f"Running timing experiment: {i}/{solve_steps}")

    return x, opt_times


def time_fun_call(fun):
    now = time.perf_counter()
    fun()
    return time.perf_counter() - now


def delete_file_by_pattern(dir_path, pattern):
    for f in os.listdir(dir_path):
        if re.search(pattern, f):
            os.remove(os.path.join(dir_path, f))


def run(
    N,
    solve_steps,
    device="cpu",
    save_data=False,
    save_dir="data",
    nn_params_path=None,
):

    nn_fun = MultiLayerPerceptron(
        n_inputs=1, hidden_layers=2, hidden_size=64, n_outputs=1
    )
    if nn_params_path is not None:
        nn_fun.load_state_dict(torch.load(nn_params_path))
    nn_fun.eval()

    solver_l4casadi = init_l4casadi(
        nn_fun,
        N,
        device=device,
    )
    x_l4casadi, opt_times_l4casadi = run_timing_experiment(
        N,
        solver_l4casadi,
        lambda: time_fun_call(solver_l4casadi.solve),
        solve_steps=solve_steps,
    )

    shutil.rmtree("c_generated_code")
    shutil.rmtree("_l4c_generated")
    delete_file_by_pattern("./", r".*[ocp|sim].*\.json")

    solver_l4acados = init_l4acados(
        nn_fun,
        N,
        device=device,
        use_cython=False,
    )
    x_l4acados, opt_times_l4acados = run_timing_experiment(
        N,
        solver_l4acados.ocp_solver,
        lambda: time_fun_call(lambda: solver_l4acados.solve()),
        solve_steps=solve_steps,
    )

    shutil.rmtree("c_generated_code")
    delete_file_by_pattern("./", r".*[ocp|sim].*\.json")

    del solver_l4casadi, solver_l4acados

    return x_l4casadi, opt_times_l4casadi, x_l4acados, opt_times_l4acados


def test_l4acados_equal_l4casadi():
    # check if nn_params.pt is in the root directory
    test_l4casadi_dir = os.path.dirname(os.path.abspath(__file__))
    nn_params_path = os.path.join("nn_params.pt")

    if not os.path.exists(nn_params_path):
        train_nn(nn_params_path=nn_params_path)

    x_l4casadi, opt_times_l4casadi, x_l4acados, opt_times_l4acados = run(
        N=10,
        solve_steps=100,
        device="cpu",
        save_data=False,
        nn_params_path=nn_params_path,
    )

    x_l4casadi = np.array(x_l4casadi)
    x_l4acados = np.array(x_l4acados)

    (
        x_l4casadi_zero,
        opt_times_l4casadi_zero,
        x_l4acados_zero,
        opt_times_l4acados_zero,
    ) = run(
        N=10,
        solve_steps=100,
        device="cpu",
        save_data=False,
        nn_params_path=None,
    )

    x_l4casadi_zero = np.array(x_l4casadi_zero)
    x_l4acados_zero = np.array(x_l4acados_zero)

    print(f"Normdiff casadi: {np.linalg.norm(x_l4casadi - x_l4casadi_zero)}")
    print(f"Normdiff acados: {np.linalg.norm(x_l4acados - x_l4acados_zero)}")
    print(f"Normdiff casadi vs acados: {np.linalg.norm(x_l4casadi - x_l4acados)}")
    print(
        f"Normdiff casadi vs acados zero: {np.linalg.norm(x_l4casadi_zero - x_l4acados_zero)}"
    )

    assert not np.allclose(x_l4casadi, x_l4casadi_zero)
    assert not np.allclose(x_l4acados, x_l4acados_zero)

    assert np.allclose(x_l4casadi_zero, x_l4acados_zero, atol=1e-5, rtol=1e-4)
    assert np.allclose(x_l4casadi, x_l4acados, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    test_l4acados_equal_l4casadi()
