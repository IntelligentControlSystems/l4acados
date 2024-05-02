import os, sys, shutil
from subprocess import check_output
import numpy as np
import casadi as cas

import torch
import gpytorch

from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimSolver,
    AcadosOcpSolver,
    ZoroDescription,
)
from .zoro_acados_utils import *
from .residual_learning_mpc import ResidualLearningMPC
from zero_order_gpmpc.models import ResidualModel

from time import perf_counter
from dataclasses import dataclass


@dataclass
class SolveData:
    n_iter: int
    sol_x: np.ndarray
    sol_u: np.ndarray
    timings_total: float
    timings: dict


class ZeroOrderGPMPC(ResidualLearningMPC):
    def __init__(
        self,
        ocp,
        sim,
        prob_x,
        Sigma_x0,
        Sigma_W,
        B=None,
        gp_model=None,
        h_tightening_idx=[],
        use_cython=True,
        path_json_ocp="zoro_ocp_solver_config.json",
        path_json_sim="zoro_sim_solver_config.json",
        build_c_code=True,
    ):
        # Set up all member variables but don't build the code yet. We still need to setup
        # the custom update.
        super().__init__(
            ocp,
            sim,
            B=B,
            residual_model=gp_model,
            use_cython=use_cython,
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
            build_c_code=False,
        )

        # TODO(@naefjo): figure out proper length
        self.covariances_array = np.zeros(200)
        """Propagated covariances along the horizon"""

        self.prob_x = prob_x
        self.Sigma_x0 = Sigma_x0
        self.Sigma_x0_diag = np.diag(Sigma_x0)
        self.Sigma_W = Sigma_W
        self.Sigma_W_diag = np.diag(Sigma_W)
        self.tighten_idx = h_tightening_idx
        self.setup_custom_update()

        self.build_c_code_done = False
        if build_c_code:
            self.build(
                use_cython=use_cython,
                build_c_code=build_c_code,
                path_json_ocp=path_json_ocp,
                path_json_sim=path_json_sim,
            )

    def solve(self, tol_nlp=1e-6, n_iter_max=30):
        time_total = perf_counter()
        self.init_solve_stats(n_iter_max)

        for i in range(n_iter_max):
            time_iter = perf_counter()
            status_prep = self.preparation(i)
            status_cupd = self.do_custom_update()
            status_feed = self.feedback(i)

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            time_check_termination = perf_counter()

            # self.ocp_solver.print_statistics() # encapsulates: stat = self.ocp_solver.get_stats("statistics")
            residuals = self.ocp_solver.get_residuals()
            print("residuals after ", i, "SQP_RTI iterations:\n", residuals)

            self.solve_stats["timings"]["check_termination"][i] += (
                perf_counter() - time_check_termination
            )
            self.solve_stats["timings"]["total"][i] += perf_counter() - time_iter

            if status_feed != 0:
                raise Exception(
                    "acados self.ocp_solver returned status {} in time step {}. Exiting.".format(
                        status_feed, i
                    )
                )

            if max(residuals) < tol_nlp:
                break

        self.solve_stats["n_iter"] = i + 1
        self.solve_stats["timings_total"] = perf_counter() - time_total

    def setup_custom_update(self):
        custom_update_source_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "custom_update_functions"
        )
        template_c_file = "custom_update_function_gpzoro_template.in.c"
        template_h_file = "custom_update_function_gpzoro_template.in.h"
        custom_c_file = "custom_update_function_gpzoro.c"
        custom_h_file = "custom_update_function_gpzoro.h"

        # copy custom update functions into acados
        # Copy custom update functions into acados
        # First find location where acados-template is installed.
        pip_output = check_output(["pip", "show", "acados-template"]).decode().split()
        path_to_acados_template = pip_output[pip_output.index("Location:") + 1]

        # Then concat the full path to the template folder.
        path_acados_custom_update = os.path.join(
            path_to_acados_template,
            "acados_template",
            "custom_update_templates",
        )
        shutil.copy(
            os.path.join(custom_update_source_dir, template_h_file),
            path_acados_custom_update,
        )
        shutil.copy(
            os.path.join(custom_update_source_dir, template_c_file),
            path_acados_custom_update,
        )

        # custom update: disturbance propagation
        self.ocp.solver_options.custom_update_filename = custom_c_file
        self.ocp.solver_options.custom_update_header_filename = custom_h_file

        self.ocp.solver_options.custom_templates = [
            (
                template_c_file,
                custom_c_file,
            ),
            (
                template_h_file,
                custom_h_file,
            ),
        ]

        self.ocp.solver_options.custom_update_copy = False
        """NOTE(@naefjo): As far as I understand you need to set this variable to True if you just
        want to copy an existing custom_update.c/h into the export directory and to False if you want
        to render the custom_udpate files from the template"""

        # zoro stuff
        zoro_description = ZoroDescription()
        zoro_description.backoff_scaling_gamma = norm.ppf(self.prob_x)
        zoro_description.P0_mat = self.Sigma_x0
        zoro_description.fdbk_K_mat = np.zeros((self.nu, self.nx))
        zoro_description.unc_jac_G_mat = self.B
        """G in (nx, nw) describes how noise affects dynamics. I.e. x+ = ... + G@w"""
        zoro_description.W_mat = self.Sigma_W
        """W in (nw, nw) describes the covariance of the noise on the system"""

        zoro_description.idx_uh_t = self.tighten_idx
        self.ocp.zoro_description = zoro_description

    def do_custom_update(self) -> None:
        """performs the acados custom update and propagates the covariances for the constraint tightening

        The array which is passed to the custom update function consists of an input array and
        an output array [cov_in, cov_out], where
        cov_in = [Sigma_x0, Sigma_w, [Sigma_GP_i forall i in (0, N-1)]] and
        cov_out = [-1*ones(3 * (N + 1)))] is a placeholder for the positional covariances used for
        visualization.

        Note that the function currently only supports setting the diagonal elements of the covariance matrices
        in the solver.
        """
        if not self.has_residual_model:
            return

        covariances_in = np.concatenate(
            (
                self.Sigma_x0_diag,
                self.Sigma_W_diag,
                # self.residual_model.current_variance is updated with value_and_jacobian() call in preparation phase
                self.residual_model.current_variance.flatten(),
            )
        )
        covariances_in_len = covariances_in.size
        out_arr = np.concatenate((covariances_in, -1.0 * np.ones(3 * (self.N + 1))))
        self.ocp_solver.custom_update(out_arr)
        assert np.all(
            out_arr[:covariances_in_len] == covariances_in
        ), "do_custom_update: elements in the input covariances changed"
        self.covariances_array = out_arr[covariances_in_len:]

        return
