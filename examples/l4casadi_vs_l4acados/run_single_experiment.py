from acados import (
    run,
    MultiLayerPerceptron,
    NaiveMultiLayerPerceptron,
    DoubleIntegratorWithLearnedDynamics,
    MPC,
)
import l4casadi as l4c
import numpy as np
import time
import l4acados as l4a
from typing import Optional, Union
import torch
import casadi as cs
from l4acados.controllers import ResidualLearningMPC
from l4acados.models import ResidualModel, PyTorchResidualModel, PyTorchFeatureSelector
from l4acados.controllers.zoro_acados_utils import setup_sim_from_ocp
import argparse
import os, shutil, re
import subprocess


def init_l4casadi(
    N: int,
    hidden_layers: int,
    batch_dim: int = 1,
    batched: bool = True,
    device="cpu",
    num_threads_acados_openmp=1,
    use_naive_l4casadi: bool = False,
):
    n_inputs = 2
    n_outputs = 1
    hidden_size = 512

    if use_naive_l4casadi:
        # mlp = l4c.naive.MultiLayerPerceptron(
        #     n_inputs, hidden_size, n_outputs, hidden_layers
        # )
        mlp = NaiveMultiLayerPerceptron(
            n_inputs=n_inputs,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            n_outputs=n_outputs,
        )
    else:
        mlp = MultiLayerPerceptron(
            n_inputs=n_inputs,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            n_outputs=n_outputs,
        )

    learned_dyn_model = l4c.L4CasADi(
        mlp,
        batched=batched,
        name="learned_dyn",
        device=device,
    )
    if use_naive_l4casadi:
        learned_dyn_model_shared_lib_dir = None
    else:
        learned_dyn_model_shared_lib_dir = learned_dyn_model.shared_lib_dir

    model = DoubleIntegratorWithLearnedDynamics(
        learned_dyn_model, batched=batched, batch_dim=batch_dim
    )

    mpc_obj = MPC(
        model=model.model(),
        N=N,
        external_shared_lib_dir=learned_dyn_model_shared_lib_dir,
        external_shared_lib_name=learned_dyn_model.name,
        num_threads_acados_openmp=num_threads_acados_openmp,
    )
    solver = mpc_obj.solver

    if num_threads_acados_openmp > 1:
        assert (
            solver.acados_lib_uses_omp == True
        ), "Acados not compiled with OpenMP, cannot use multiple threads."

    ocp = mpc_obj.ocp()
    return solver


def init_l4acados(
    N: int,
    hidden_layers: int,
    batch_dim: int = 1,
    batched: bool = True,
    device="cpu",
    use_cython=False,
    num_threads_acados_openmp=1,
):
    feature_selector = PyTorchFeatureSelector([1, 1, 0], device=device)
    residual_model = PyTorchResidualModel(
        MultiLayerPerceptron(hidden_layers=hidden_layers).to(device),
        feature_selector,
    )
    B_proj = np.ones((1, batch_dim))
    model_new = DoubleIntegratorWithLearnedDynamics(None, name="wr_new")
    mpc_obj_nolib = MPC(
        model=model_new.model(),
        N=N,
        num_threads_acados_openmp=num_threads_acados_openmp,
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
        t = np.linspace(
            i * ts, i * ts + T, N
        )  # TODO: this ts should be T IMO, maybe with lower frequency?
        yref = np.sin(2 * np.pi * t + np.pi / 2)
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
    hidden_layers,
    solve_steps,
    device="cpu",
    save_data=False,
    save_dir="data",
    num_threads: int = -1,
    num_threads_acados_openmp: int = 1,
    build_acados: bool = True,
):

    if build_acados:
        if num_threads_acados_openmp >= 1:
            build_acados_with_openmp = "ON"
        else:
            build_acados_with_openmp = "OFF"

        subprocess.check_call(
            [
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "build_acados_num_threads.sh",
                ),
                str(num_threads_acados_openmp),
                build_acados_with_openmp,
            ]
        )

    if device == "cuda":
        num_threads = 1
    elif num_threads == -1:
        num_threads = os.cpu_count() // 2
    torch.set_num_threads(num_threads)

    # standard L4CasADi
    solver_l4casadi = init_l4casadi(
        N,
        hidden_layers,
        device=device,
        num_threads_acados_openmp=num_threads_acados_openmp,
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

    # Naive L4CasADi
    solver_l4casadi_naive = init_l4casadi(
        N,
        hidden_layers,
        device=device,
        num_threads_acados_openmp=num_threads_acados_openmp,
        use_naive_l4casadi=True,
    )
    x_l4casadi_naive, opt_times_l4casadi_naive = run_timing_experiment(
        N,
        solver_l4casadi_naive,
        lambda: time_fun_call(solver_l4casadi_naive.solve),
        solve_steps=solve_steps,
    )

    shutil.rmtree("c_generated_code")
    delete_file_by_pattern("./", r".*[ocp|sim].*\.json")

    solver_l4acados = init_l4acados(
        N,
        hidden_layers,
        device=device,
        use_cython=True,
        num_threads_acados_openmp=num_threads_acados_openmp,
    )
    x_l4acados, opt_times_l4acados = run_timing_experiment(
        N,
        solver_l4acados.ocp_solver,
        lambda: time_fun_call(lambda: solver_l4acados.solve()),
        solve_steps=solve_steps,
    )

    shutil.rmtree("c_generated_code")
    delete_file_by_pattern("./", r".*[ocp|sim].*\.json")

    if save_data:
        print("Saving data")
        np.savez(
            os.path.join(
                save_dir,
                f"l4casadi_vs_l4acados_N{N}_layers{hidden_layers}_steps{solve_steps}_{device}_threads{num_threads}_acados_{num_threads_acados_openmp}.npz",
            ),
            x_l4casadi=x_l4casadi,
            opt_times_l4casadi=opt_times_l4casadi,
            x_l4casadi_naive=x_l4casadi_naive,
            opt_times_l4casadi_naive=opt_times_l4casadi_naive,
            x_l4acados=x_l4acados,
            opt_times_l4acados=opt_times_l4acados,
        )

    del solver_l4casadi, solver_l4casadi_naive, solver_l4acados

    return (
        x_l4casadi,
        opt_times_l4casadi,
        x_l4casadi_naive,
        opt_times_l4casadi_naive,
        x_l4acados,
        opt_times_l4acados,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--solve_steps", type=int, default=1000)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--num_threads_acados_openmp", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--build_acados", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    args_dict = vars(args)

    print_str = "Run_single_experiment: "
    for arg_name, arg_value in args_dict.items():
        print_str += f"{arg_name}={arg_value}, "
    print(f"{print_str}\n")

    run(
        args.N,
        args.hidden_layers,
        args.solve_steps,
        device=args.device,
        num_threads=args.num_threads,
        num_threads_acados_openmp=args.num_threads_acados_openmp,
        build_acados=args.build_acados,
        save_data=True,
    )

# python run_single_experiment.py --N 10 --hidden_layers 20 --solve_steps 100 --num_threads 1 --num_threads_acados_openmp 14 --device cpu --(no-)build_acados
