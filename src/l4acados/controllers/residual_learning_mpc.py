import numpy as np

from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
from .zoro_acados_utils import (
    transform_ocp,
    setup_sim_from_ocp,
    get_solve_opts_from_ocp,
)
from l4acados.models import ResidualModel
from typing import Tuple


class ResidualLearningMPC:
    def __init__(
        self,
        ocp: AcadosOcp,
        B: np.ndarray = None,
        residual_model: ResidualModel = None,
        build_c_code: bool = True,
        use_cython: bool = True,
        path_json_ocp: str = "residual_lbmpc_ocp_solver_config.json",
        path_json_sim: str = "residual_lbmpc_sim_solver_config.json",
    ) -> None:
        """
        ocp: AcadosOcp for nominal problem
        sim: AcadosSim for nominal model
        B: Residual Jacobian mapping from resiudal dimension to state dimension. If None,
            it is assumed that redsidual_dim == state_dim
        residual_model: ResidualModel class
        build_c_code: Whether the solver should be built. Note, if you do not build it here,
            you will have to call the build function before calling the solver.
        The following args only apply if build_c_code == True:
        use_cython: Whether Acados' cython solver interface should be used. You probably
            want this enabled.
        path_json_ocp: Name of the json file where the resulting ocp will be dumped
        path_json_sim: Name of the json file where the resulting sim will be dumped
        """

        ocp.make_consistent()

        # optional argument
        if B is None:
            B = np.eye(ocp.dims.nx)
        self.B = B

        # transform OCP to linear-params-model
        self.ocp, self.ocp_opts = transform_ocp(ocp, use_cython)
        self.sim = setup_sim_from_ocp(ocp)
        self.ocp_opts_tol_arr = np.array(
            [
                self.ocp_opts["nlp_solver_tol_stat"],
                self.ocp_opts["nlp_solver_tol_eq"],
                self.ocp_opts["nlp_solver_tol_ineq"],
                self.ocp_opts["nlp_solver_tol_comp"],
            ]
        )

        # get dimensions
        self.nx = self.ocp.dims.nx
        self.nu = self.ocp.dims.nu
        self.np_nonlin = ocp.dims.np
        self.np_linmdl = self.ocp.dims.np
        self.N = self.ocp.dims.N
        self.nw = B.shape[1]

        # allocation
        self.x_hat_all = np.zeros((self.N + 1, self.nx))
        self.u_hat_all = np.zeros((self.N, self.nu))
        self.y_hat_all = np.zeros((self.N, self.nx + self.nu))

        self.residual_fun = np.zeros((self.N, self.nw))
        self.residual_jac = np.zeros((self.nw, self.N, self.nx + self.nu))
        self.p_hat_nonlin = np.array([ocp.parameter_values for _ in range(self.N + 1)])
        self.p_hat_linmdl = np.array(
            [self.ocp.parameter_values for _ in range(self.N + 1)]
        )
        self.nlp_residuals = np.zeros((self.ocp_opts["nlp_solver_max_iter"], 4))
        self.num_iter = 0

        self.has_residual_model = False
        if residual_model is not None:
            self.has_residual_model = True
            self.residual_model = residual_model

        self.build(
            use_cython=use_cython,
            build_c_code=build_c_code,
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
        )

        self.init_last_iterate()

    def build(
        self,
        use_cython=False,
        build_c_code=True,
        path_json_ocp="residual_lbmpc_ocp_solver_config.json",
        path_json_sim="residual_lbmpc_sim_solver_config.json",
    ) -> None:
        """
        build_c_code: Whether the solver should be built. If set to false, the solver
            will simply be read from the json files.
        use_cython: Whether Acados' cython solver interface should be used. You probably
            want this enabled.
        path_json_ocp: Name of the json file where the resulting ocp will be dumped
            (or was dumped if build_c_code == False)
        path_json_sim: Name of the json file where the resulting sim will be dumped
            (or was dumped if build_c_code == False)
        """
        self.use_cython = use_cython
        if use_cython:
            if build_c_code:
                AcadosOcpSolver.generate(self.ocp, json_file=path_json_ocp)
                AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
                AcadosSimSolver.generate(self.sim, json_file=path_json_sim)
                AcadosSimSolver.build(self.sim.code_export_directory, with_cython=True)

            self.ocp_solver = AcadosOcpSolver.create_cython_solver(path_json_ocp)
            self.sim_solver = AcadosSimSolver.create_cython_solver(path_json_sim)
        else:
            self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=path_json_ocp)
            self.sim_solver = AcadosSimSolver(self.sim, json_file=path_json_sim)

    def solve(self, acados_sqp_mode=False):
        status_feed = 0
        self.num_iter = 0
        for i in range(self.ocp_opts["nlp_solver_max_iter"]):
            self.preparation()

            if acados_sqp_mode:
                self.store_last_iterate()

            status_feed = self.feedback()
            if status_feed != 0:
                raise Exception(
                    "acados self.ocp_solver returned status {} in time step {}. Exiting.".format(
                        status_feed, i
                    )
                )
            self.num_iter += 1

            # ------------------- Check termination --------------------
            if self.ocp.solver_options.rti_log_residuals:
                self.nlp_residuals[i, :] = self.get_initial_residuals()

                if np.all(self.nlp_residuals[i, :] < self.ocp_opts_tol_arr):
                    if acados_sqp_mode:
                        # restore previous iterate for which residuals are valid
                        self.load_last_iterate()
                    break

        return status_feed

    def preparation(self):
        # ------------------- Query nodes --------------------
        # preparation rti_phase (solve() AFTER setting params to get right Jacobians)
        self.ocp_solver.options_set("rti_phase", 1)

        # get linearization points for all stages
        for stage in range(self.N):
            # current stage values
            self.x_hat_all[stage, :] = self.ocp_solver.get(stage, "x")
            self.u_hat_all[stage, :] = self.ocp_solver.get(stage, "u")
            self.y_hat_all[stage, :] = np.hstack(
                (self.x_hat_all[stage, :], self.u_hat_all[stage, :])
            ).reshape((1, self.nx + self.nu))

        self.x_hat_all[self.N, :] = self.ocp_solver.get(self.N, "x")

        # ------------------- Sensitivities --------------------
        if self.has_residual_model:
            self.residual_fun, self.residual_jac = (
                self.residual_model.value_and_jacobian(self.y_hat_all)
            )

        # ------------------- Update stages --------------------
        for stage in range(self.N):
            # set parameters (linear matrices and offset)
            # ------------------- Integrate --------------------
            self.sim_solver.set("x", self.x_hat_all[stage, :])
            self.sim_solver.set("u", self.u_hat_all[stage, :])
            self.sim_solver.set("p", self.p_hat_nonlin[stage, :])
            status_integrator = self.sim_solver.solve()

            A_nom = self.sim_solver.get("Sx")
            B_nom = self.sim_solver.get("Su")
            x_nom = self.sim_solver.get("x")

            # ------------------- Build linear model --------------------
            A_total = A_nom + self.B @ self.residual_jac[:, stage, 0 : self.nx]
            B_total = (
                B_nom
                + self.B @ self.residual_jac[:, stage, self.nx : self.nx + self.nu]
            )

            f_hat = (
                x_nom
                + self.B @ self.residual_fun[stage, :]
                - A_total @ self.x_hat_all[stage, :]
                - B_total @ self.u_hat_all[stage, :]
            )

            # ------------------- Set sensitivities --------------------
            A_reshape = np.reshape(A_total, (self.nx**2), order="F")
            B_reshape = np.reshape(B_total, (self.nx * self.nu), order="F")

            self.p_hat_linmdl[stage, :] = np.hstack(
                (A_reshape, B_reshape, f_hat, self.p_hat_nonlin[stage, :])
            )
            self.ocp_solver.set(stage, "p", self.p_hat_linmdl[stage, :])

        # use stage N for linear part of last stage (unused anyway)
        self.p_hat_linmdl[self.N, :] = np.hstack(
            (A_reshape, B_reshape, f_hat, self.p_hat_nonlin[self.N, :])
        )
        self.ocp_solver.set(self.N, "p", self.p_hat_linmdl[self.N, :])

        # feedback rti_phase
        # ------------------- Phase 1 --------------------
        status = self.ocp_solver.solve()

    def feedback(self):
        # ------------------- Solve QP --------------------
        self.ocp_solver.options_set("rti_phase", 2)
        status = self.ocp_solver.solve()
        return status

    def get_solution(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((self.N + 1, self.nx))
        U = np.zeros((self.N, self.nu))

        # get data
        for i in range(self.N):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")

        X[self.N, :] = self.ocp_solver.get(self.N, "x")

        return X, U

    def init_last_iterate(self):
        npi = len(self.ocp_solver.get(0, "pi"))
        nlam_0 = len(self.ocp_solver.get(0, "lam"))
        if self.N > 1:
            nlam = len(self.ocp_solver.get(1, "lam"))
        else:
            nlam = 0
        nlam_e = len(self.ocp_solver.get(self.N, "lam"))

        self.pi_hat_all = np.zeros((self.N, self.nx))
        self.lam_hat_all_0 = np.zeros((nlam_0,))
        self.lam_hat_all = np.zeros((self.N, nlam))
        self.lam_hat_all_e = np.zeros((nlam_e,))

        self.x_hat_all_lastiter = np.zeros((self.N + 1, self.nx))
        self.u_hat_all_lastiter = np.zeros((self.N, self.nu))
        self.pi_hat_all_lastiter = np.zeros((self.N, npi))
        self.lam_hat_all_lastiter_0 = np.zeros((nlam_0,))
        self.lam_hat_all_lastiter = np.zeros((self.N, nlam))
        self.lam_hat_all_lastiter_e = np.zeros((nlam_e,))

    def load_last_iterate(self):
        # TODO: remove this once cython support is enabled
        if self.use_cython:
            self.ocp_solver.set(0, "lam", self.lam_hat_all_lastiter_0[:])
            for stage in range(self.N):
                self.ocp_solver.set(stage, "x", self.x_hat_all_lastiter[stage, :])
                self.ocp_solver.set(stage, "u", self.u_hat_all_lastiter[stage, :])
                self.ocp_solver.set(stage, "pi", self.pi_hat_all_lastiter[stage, :])
                if stage > 0:
                    self.ocp_solver.set(
                        stage, "lam", self.lam_hat_all_lastiter[stage, :]
                    )
            self.ocp_solver.set(self.N, "x", self.x_hat_all_lastiter[self.N, :])
            self.ocp_solver.set(self.N, "lam", self.lam_hat_all_lastiter_e[:])
        else:
            self.ocp_solver.load_iterate_from_flat_obj(self.last_iterate)

    def store_last_iterate(self):
        # TODO: remove this once cython support is enabled
        if self.use_cython:
            self.lam_hat_all_lastiter_0[:] = self.ocp_solver.get(0, "lam")
            for stage in range(self.N):
                self.x_hat_all_lastiter[stage, :] = self.ocp_solver.get(stage, "x")
                self.u_hat_all_lastiter[stage, :] = self.ocp_solver.get(stage, "u")
                self.pi_hat_all_lastiter[stage, :] = self.ocp_solver.get(stage, "pi")
                if stage > 0:
                    self.lam_hat_all_lastiter[stage, :] = self.ocp_solver.get(
                        stage, "lam"
                    )
            self.x_hat_all_lastiter[self.N, :] = self.ocp_solver.get(self.N, "x")
            self.lam_hat_all_lastiter_e[:] = self.ocp_solver.get(self.N, "lam")
        else:
            self.last_iterate = self.ocp_solver.store_iterate_to_flat_obj()

    # ------------------- Forward OCP solver functions -------------------
    def dump_last_qp_to_json(self, *args, **kwargs) -> None:
        self.ocp_solver.dump_last_qp_to_json(*args, **kwargs)

    def get(self, stage: int, var: str) -> np.ndarray:
        if var == "p":
            assert 0 <= stage <= self.N
            return self.p_hat_nonlin[stage, :]  # params are set in preparation phase
        else:
            return self.ocp_solver.get(stage, var)

    def get_initial_residuals(self) -> np.ndarray:
        return self.ocp_solver.get_initial_residuals()

    def get_residuals(self, recompute=False, ignore_warning=False) -> np.ndarray:
        if ignore_warning:
            return self.ocp_solver.get_residuals(recompute, ignore_warning)
        raise ValueError(
            "Only logging of available residuals is supported. Use get_initial_residuals() instead. See https://github.com/acados/acados/pull/1346."
        )

    def get_stats(self, stat: str):
        if stat == "res_stat_all":
            return self.nlp_residuals[: self.num_iter, 0]
        if stat == "res_eq_all":
            return self.nlp_residuals[: self.num_iter, 1]
        if stat == "res_ineq_all":
            return self.nlp_residuals[: self.num_iter, 2]
        if stat == "res_comp_all":
            return self.nlp_residuals[: self.num_iter, 3]
        return self.ocp_solver.get_stats(stat)

    def set(self, stage: int, var: str, value: np.ndarray) -> None:
        if var == "p":
            assert 0 <= stage <= self.N
            self.p_hat_nonlin[stage, :] = value
        else:
            self.ocp_solver.set(stage, var, value)

    def cost_set(self, stage: int, cost: str, value: np.ndarray) -> None:
        self.ocp_solver.cost_set(stage, cost, value)

    def constraints_set(self, stage: int, constraint: str, value: np.ndarray) -> None:
        self.ocp_solver.constraints_set(stage, constraint, value)

    def options_set(self, option: str, value) -> None:
        self.ocp_solver.options_set(option, value)

    def print_statistics(self) -> None:
        if self.ocp_opts["nlp_solver_type"] == "SQP":
            print("iter    res_stat        res_eq          res_ineq        res_comp")
            for i in range(self.num_iter):
                print(
                    f"{i:<7d} {self.nlp_residuals[i, 0]:12e}    {self.nlp_residuals[i, 1]:12e}    {self.nlp_residuals[i, 2]:12e}    {self.nlp_residuals[i, 3]:11e}"
                )
            return
        self.ocp_solver.print_statistics()

    def solve_for_x0(
        self, x0_bar, fail_on_nonzero_status=True, print_stats_on_failure=True
    ) -> np.ndarray:
        self.ocp_solver.set(0, "lbx", x0_bar)
        self.ocp_solver.set(0, "ubx", x0_bar)

        status = self.solve()

        if status != 0:
            if print_stats_on_failure:
                self.ocp_solver.print_statistics()
            if fail_on_nonzero_status:
                raise Exception(f"acados acados_ocp_solver returned status {status}")
            elif print_stats_on_failure:
                print(f"Warning: acados acados_ocp_solver returned status {status}")

        u0 = self.ocp_solver.get(0, "u")
        return u0
