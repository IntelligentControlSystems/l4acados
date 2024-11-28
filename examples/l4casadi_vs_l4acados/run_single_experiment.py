from acados import run, MultiLayerPerceptron, DoubleIntegratorWithLearnedDynamics, MPC
import l4casadi as l4c
import numpy as np
import time
import l4acados as l4a
from typing import Optional, Union
import torch
import casadi as cs
from l4acados.controllers.residual_learning_mpc import ResidualLearningMPC
from l4acados.models import ResidualModel
from l4acados.models.pytorch_models.pytorch_feature_selector import (
    FeatureSelector,
)
from l4acados.controllers.zoro_acados_utils import setup_sim_from_ocp
import argparse
import os, shutil, re
import subprocess


class PyTorchResidualModel(ResidualModel):
    """Basic PyTorch residual model class.

    Args:
        - model: Any Pytorch torch.nn.Module.
        - feature_selector: Optional feature selector mapping (state, input) -> (NN input dimension). If set to None, then no selection
          is performed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_selector: Optional[FeatureSelector] = None,
        device="cpu",
    ):
        self.model = model

        self._gp_feature_selector = (
            feature_selector if feature_selector is not None else FeatureSelector()
        )

        if device == "cuda":
            self.to_tensor = lambda X: torch.Tensor(X).cuda()
            self.to_numpy = lambda T: T.cpu().detach().numpy()
            self.model.cuda()
        elif device == "cpu":
            self.to_tensor = lambda X: torch.Tensor(X)
            self.to_numpy = lambda T: T.detach().numpy()
            self.model.cpu()
        else:
            raise ValueError(f"Unknown device {device}, should be 'cpu' or 'gpu'")

    def _mean_fun_sum(self, y):
        """Helper function for jacobian computation

        sums up the mean predictions along the first dimension
        (i.e. along the horizon).
        """
        self.evaluate(y, require_grad=True)
        return self.predictions.sum(dim=0)

    def evaluate(self, y, require_grad=False):
        y_tensor = self.to_tensor(y)
        if require_grad:
            self.predictions = self.model(self._gp_feature_selector(y_tensor))
        else:
            with torch.no_grad():
                self.predictions = self.model(self._gp_feature_selector(y_tensor))

        self.current_mean = self.to_numpy(self.predictions)
        return self.current_mean

    def jacobian(self, y):
        y_tensor = self.to_tensor(y)
        mean_dy = torch.autograd.functional.jacobian(self._mean_fun_sum, y_tensor)
        return self.to_numpy(mean_dy)

    def value_and_jacobian(self, y):
        """Computes the necessary values for GPMPC

        Args:
            - x_input: (N, state_dim) tensor

        Returns:
            - mean:  (N, residual_dim) tensor
            - mean_dy:  (residual_dim, N, state_dim) tensor
            - covariance:  (N, residual_dim) tensor
        """
        self.current_mean_dy = self.jacobian(y)

        return self.current_mean, self.current_mean_dy


def init_l4casadi(
    N: int,
    hidden_layers: int,
    batch_dim: int = 1,
    batched: bool = True,
    device="cpu",
    num_threads_acados_openmp=1,
):
    learned_dyn_model = l4c.L4CasADi(
        MultiLayerPerceptron(hidden_layers=hidden_layers),
        batched=batched,
        name="learned_dyn",
        device=device,
    )

    model = DoubleIntegratorWithLearnedDynamics(
        learned_dyn_model, batched=batched, batch_dim=batch_dim
    )

    mpc_obj = MPC(
        model=model.model(),
        N=N,
        external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
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
    feature_selector = FeatureSelector([1, 1, 0], device=device)
    residual_model = PyTorchResidualModel(
        MultiLayerPerceptron(hidden_layers=hidden_layers),
        feature_selector,
        device=device,
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
        # save data
        print("Saving data")
        np.savez(
            os.path.join(
                save_dir,
                f"l4casadi_vs_l4acados_N{N}_layers{hidden_layers}_steps{solve_steps}_{device}_threads{num_threads}_acados_{num_threads_acados_openmp}.npz",
            ),
            x_l4casadi=x_l4casadi,
            opt_times_l4casadi=opt_times_l4casadi,
            x_l4acados=x_l4acados,
            opt_times_l4acados=opt_times_l4acados,
        )

    del solver_l4casadi, solver_l4acados

    return x_l4casadi, opt_times_l4casadi, x_l4acados, opt_times_l4acados


if __name__ == "__main__":
    # parse arguments with argparse
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
