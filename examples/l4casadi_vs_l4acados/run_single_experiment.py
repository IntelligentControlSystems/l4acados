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

ts = 0.05


def init_l4casadi(
    N: int,
    hidden_layers: int,
    hidden_size: int,
    batch_dim: int = 1,
    batched: bool = True,
    device="cpu",
    num_threads_acados_openmp=1,
    use_naive_l4casadi: bool = False,
):
    n_inputs = 2
    n_outputs = 1

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
        T=ts * N,
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
    hidden_size: int,
    batch_dim: int = 1,
    device="cpu",
    use_cython=False,
    num_threads_acados_openmp=1,
):
    feature_selector = PyTorchFeatureSelector([1, 1, 0], device=device)
    residual_model = PyTorchResidualModel(
        MultiLayerPerceptron(hidden_layers=hidden_layers).to(device),
        feature_selector,
        measure_to_tensor_time=True,
    )
    B_proj = np.ones((1, batch_dim))
    model_new = DoubleIntegratorWithLearnedDynamics(None, name="wr_new")
    mpc_obj_nolib = MPC(
        model=model_new.model(),
        N=N,
        T=ts * N,
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


def get_timings(solver):
    timings_dict = {
        "time_preparation": solver.get_stats("time_preparation"),
        "time_feedback": solver.get_stats("time_feedback"),
    }
    return timings_dict


def get_timings_l4acados_with_to_tensor_time(l4acados_solver):
    timings_dict = get_timings(l4acados_solver)
    timings_dict["time_to_tensor"] = l4acados_solver.residual_model.to_tensor_time
    timings_dict["time_residual_model"] = l4acados_solver.time_residual[0]
    timings_dict["time_nominal_model"] = l4acados_solver.time_nominal[0]
    l4acados_solver.residual_model.to_tensor_time = 0.0  # reset after getting timings
    return timings_dict


def run_timing_experiment(
    N,
    init_solver_call,
    solve_call=lambda solver: 0.0,
    get_timings_call=lambda solver: {},
    solve_steps=1e3,
):
    x = []
    u = []
    xt = np.array([1.0, 0.0])
    T = ts * N
    # T = 1.0
    # ts = T / N
    solver = init_solver_call()

    opt_times = []
    opt_times_preparation = []
    opt_times_feedback = []
    results_dict = {
        "time_total": [],
        "time_preparation": [],
        "time_feedback": [],
        "state_trajectory": [],
    }
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
        timings_dict = solve_call(solver)
        timings_dict_extra = get_timings_call(solver)

        xt = solver.get(1, "x")

        for tdict in [timings_dict, timings_dict_extra]:
            for key in tdict:
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(tdict[key])

        results_dict["state_trajectory"].append(xt)

        print(f"Running timing experiment: {i}/{solve_steps}")

    del solver
    return results_dict


def time_rti_call_acados(solver):
    start_preparation = time.perf_counter()
    solver.options_set("rti_phase", 1)
    solver.solve()
    time_preparation = time.perf_counter() - start_preparation

    start_feedback = time.perf_counter()
    solver.options_set("rti_phase", 2)
    solver.solve()
    time_feedback = time.perf_counter() - start_feedback

    return {
        "time_preparation_extra": time_preparation,
        "time_feedback_extra": time_feedback,
        "time_total": time_preparation + time_feedback,
    }


def time_fun_call(fun):
    now = time.perf_counter()
    fun()
    return {
        "time_total": time.perf_counter() - now,
    }


def delete_file_by_pattern(dir_path, pattern):
    for f in os.listdir(dir_path):
        if re.search(pattern, f):
            os.remove(os.path.join(dir_path, f))


def run(
    N,
    hidden_layers,
    hidden_size,
    solve_steps,
    device="cpu",
    save_data=False,
    save_dir="data",
    num_threads: int = -1,
    num_threads_acados_openmp: int = 1,
    build_acados: bool = True,
    run_methods=["l4casadi", "l4casadi_naive", "l4acados"],
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
    torch.set_num_interop_threads(num_threads)

    # standard L4CasADi
    if "l4casadi" in run_methods:
        results_dict_l4casadi = run_timing_experiment(
            N,
            lambda: init_l4casadi(
                N,
                hidden_layers,
                hidden_size,
                device=device,
                num_threads_acados_openmp=num_threads_acados_openmp,
                use_naive_l4casadi=False,
            ),
            lambda solver: time_rti_call_acados(solver),
            lambda solver: get_timings(solver),
            solve_steps=solve_steps,
        )

        shutil.rmtree("c_generated_code")
        shutil.rmtree("_l4c_generated")
        delete_file_by_pattern("./", r".*[ocp|sim].*\.json")
    else:
        print("Skipping standard L4CasADi, as it is not requested.")
        results_dict_l4casadi = {}

    # Naive L4CasADi
    if "l4casadi_naive" in run_methods and device == "cpu":
        results_dict_l4casadi_naive = run_timing_experiment(
            N,
            lambda: init_l4casadi(
                N,
                hidden_layers,
                hidden_size,
                device=device,
                num_threads_acados_openmp=num_threads_acados_openmp,
                use_naive_l4casadi=True,
            ),
            lambda solver: time_rti_call_acados(solver),
            lambda solver: get_timings(solver),
            solve_steps=solve_steps,
        )

        shutil.rmtree("c_generated_code")
        delete_file_by_pattern("./", r".*[ocp|sim].*\.json")
    else:
        if device != "cpu":
            print(
                "Skipping Naive L4CasADi for non-CPU devices, as it is not implemented."
            )
        else:
            print("Skipping Naive L4CasADi, as it is not requested.")
        results_dict_l4casadi_naive = {}

    if "l4acados" in run_methods:
        results_dict_l4acados = run_timing_experiment(
            N,
            lambda: init_l4acados(
                N,
                hidden_layers,
                hidden_size,
                device=device,
                use_cython=True,
                num_threads_acados_openmp=num_threads_acados_openmp,
            ),
            lambda solver: time_fun_call(solver.solve),
            lambda solver: get_timings_l4acados_with_to_tensor_time(solver),
            solve_steps=solve_steps,
        )

        shutil.rmtree("c_generated_code")
        delete_file_by_pattern("./", r".*[ocp|sim].*\.json")
    else:
        print("Skipping L4Acados, as it is not requested.")
        results_dict_l4acados = {}

    results_dict_all = {
        **{f"l4casadi_{key}": value for key, value in results_dict_l4casadi.items()},
        **{
            f"l4casadi_naive_{key}": value
            for key, value in results_dict_l4casadi_naive.items()
        },
        **{f"l4acados_{key}": value for key, value in results_dict_l4acados.items()},
    }

    if save_data:
        print("Saving data")
        np.savez(
            os.path.join(
                save_dir,
                f"l4casadi_vs_l4acados_N{N}_layers{hidden_layers}_size{hidden_size}_steps{solve_steps}_{device}_threads{num_threads}_acados{num_threads_acados_openmp}.npz",
            ),
            **results_dict_all,
        )

    return results_dict_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
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
        args.hidden_size,
        args.solve_steps,
        device=args.device,
        num_threads=args.num_threads,
        num_threads_acados_openmp=args.num_threads_acados_openmp,
        build_acados=args.build_acados,
        save_data=True,
    )

# python run_single_experiment.py --N 10 --hidden_layers 20 --hidden_size 512 --solve_steps 100 --num_threads 1 --num_threads_acados_openmp 14 --device cpu --(no-)build_acados
