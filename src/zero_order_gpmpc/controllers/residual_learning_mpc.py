import numpy as np

from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
from .zoro_acados_utils import (
    transform_ocp,
    setup_sim_from_ocp,
    get_solve_opts_from_ocp,
)
from zero_order_gpmpc.models import ResidualModel


class ResidualLearningMPC:
    def __init__(
        self,
        ocp: AcadosOcp,
        B: np.ndarray = None,
        residual_model: ResidualModel = None,
        build_c_code: bool = True,
        use_cython: bool = True,
        path_json_ocp: str = "zoro_ocp_solver_config.json",
        path_json_sim: str = "zoro_sim_solver_config.json",
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

        # optional argument
        if B is None:
            B = np.eye(ocp.dims.nx)

        # transform OCP to linear-params-model
        self.B = B
        self.sim = setup_sim_from_ocp(ocp)
        self.ocp = transform_ocp(ocp)
        self.ocp_opts = get_solve_opts_from_ocp(ocp)
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

        self.has_residual_model = False
        if residual_model is not None:
            self.has_residual_model = True
            self.residual_model = residual_model

        self.build_c_code_done = False
        if build_c_code:
            self.build(
                use_cython=use_cython,
                build_c_code=build_c_code,
                path_json_ocp=path_json_ocp,
                path_json_sim=path_json_sim,
            )

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

        self.build_c_code_done = True

    def solve(self):
        for i in range(self.ocp_opts["nlp_solver_max_iter"]):
            self.preparation()
            status_feed = self.feedback()

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            residuals = self.ocp_solver.get_residuals()

            if status_feed != 0:
                raise Exception(
                    "acados self.ocp_solver returned status {} in time step {}. Exiting.".format(
                        status_feed, i
                    )
                )

            if np.all(residuals < self.ocp_opts_tol_arr):
                break

    def preparation(self):
        # ------------------- Query nodes --------------------
        # preparation rti_phase (solve() AFTER setting params to get right Jacobians)
        self.ocp_solver.options_set("rti_phase", 1)

        # get sensitivities for all stages
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

    def get_solution(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((self.N + 1, self.nx))
        U = np.zeros((self.N, self.nu))

        # get data
        for i in range(self.N):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")

        X[self.N, :] = self.ocp_solver.get(self.N, "x")

        return X, U
