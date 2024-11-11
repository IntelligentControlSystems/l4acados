# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#
import numpy as np
import scipy.linalg
from casadi import vertcat
from zero_order_gpmpc.controllers import ResidualLearningMPC
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from acados_python_examples.getting_started.pendulum_model import (
    export_pendulum_ode_model,
)
from acados_python_examples.getting_started.utils import plot_pendulum


def create_ocp(x0, Fmax, N_horizon, Tf, RTI=False, model_name="pendulum_ode"):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model(model_name=model_name)
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-2])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])

    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_newton_iter = 10
    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-6

    if RTI:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.rti_log_residuals = 1
        ocp.solver_options.rti_log_only_available_residuals = 1
    else:
        ocp.solver_options.nlp_solver_type = "SQP"
        # NOTE: globalization currently not supported by l4acados
        # ocp.solver_options.globalization = (
        #     "MERIT_BACKTRACKING"  # turns on globalization
        # )
        ocp.solver_options.nlp_solver_max_iter = 150
        # ocp.solver_options.store_iterates = True

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    return ocp


def setup_acados(ocp, json_file):
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file=json_file)
    return acados_ocp_solver, acados_integrator


def run(use_RTI=False, use_l4acados=False, plot=True, max_sqp_iter=150):

    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    Fmax = 80

    Tf = 0.8
    N_horizon = 40

    model_name = "pendulum_ode"
    solve_kwargs = {}
    if use_l4acados:
        model_name += f"_l4a"
        if not use_RTI:
            solve_kwargs = {"acados_sqp_mode": True}
    if use_RTI:
        max_sqp_iter = 1
        model_name += f"_rti_{max_sqp_iter}"
    else:
        model_name += f"_sqp_{max_sqp_iter}"

    ocp = create_ocp(x0, Fmax, N_horizon, Tf, use_RTI)
    ocp.solver_options.nlp_solver_max_iter = max_sqp_iter

    solver_json = "acados_ocp_" + ocp.model.name + ".json"

    if use_l4acados:
        solver_json_ocp = "l4acados_ocp_" + ocp.model.name + ".json"
        solver_json_sim = "l4acados_sim_" + ocp.model.name + ".json"
        l4acados_solver = ResidualLearningMPC(
            ocp,
            use_cython=False,
            path_json_ocp=solver_json_ocp,
            path_json_sim=solver_json_sim,
        )
        ocp_solver = l4acados_solver
        integrator = AcadosSimSolver(ocp, json_file=solver_json)
    else:
        ocp_solver, integrator = setup_acados(ocp, solver_json)

    nx = ocp.dims.nx
    nu = ocp.dims.nu

    Nsim = 10
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0, :] = x0

    t_total = np.zeros((Nsim,))
    nlp_residuals = np.zeros((Nsim, max_sqp_iter + 1, 4))

    # do some initial iterations to start with a good initial guess
    # if use_RTI or max_sqp_iter == 1:
    #     num_iter_initial = 5
    #     for _ in range(num_iter_initial):
    #         ocp_solver.solve_for_x0(x0_bar=x0, fail_on_nonzero_status=False)
    #         print(ocp_solver.get_residuals())

    # closed loop
    for i in range(Nsim):
        # set initial state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        if use_RTI and not use_l4acados:
            status = ocp_solver.solve(**solve_kwargs)
        else:
            status = ocp_solver.solve(**solve_kwargs)

        t_total[i] = ocp_solver.get_stats("time_tot")
        simU[i, :] = ocp_solver.get(0, "u")

        if use_RTI:
            res_stat = ocp_solver.get_stats("res_stat_all")[[0]]
            res_eq = ocp_solver.get_stats("res_eq_all")[[0]]
            res_ineq = ocp_solver.get_stats("res_ineq_all")[[0]]
            res_comp = ocp_solver.get_stats("res_comp_all")[[0]]
        else:
            res_stat = ocp_solver.get_stats("res_stat_all")
            res_eq = ocp_solver.get_stats("res_eq_all")
            res_ineq = ocp_solver.get_stats("res_ineq_all")
            res_comp = ocp_solver.get_stats("res_comp_all")

        nlp_residuals[i, : len(res_stat), 0] = res_stat
        nlp_residuals[i, : len(res_eq), 1] = res_eq
        nlp_residuals[i, : len(res_ineq), 2] = res_ineq
        nlp_residuals[i, : len(res_comp), 3] = res_comp
        # nlp_residuals[i, :] = ocp_solver.get_initial_residuals()

        # simulate system
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # scale to milliseconds
    t_total *= 1000
    print(
        f"Computation time (total) in ms: min {np.min(t_total):.3f} median {np.median(t_total):.3f} max {np.max(t_total):.3f}"
    )

    timings = {"total": t_total}

    # plot results
    model = ocp.model

    if plot:
        plot_pendulum(
            np.linspace(0, (Tf / N_horizon) * Nsim, Nsim + 1),
            Fmax,
            simU,
            simX,
            latexify=False,
            time_label=model.t_label,
            x_labels=model.x_labels,
            u_labels=model.u_labels,
            plt_show=False,
        )

    del ocp_solver

    return simX, simU, timings, nlp_residuals


def test_minimal_example_closed_loop():
    simX_l4acados, simU_l4acados, timings_l4acados, nlp_res_l4acados = run(
        use_RTI=False, use_l4acados=True, plot=True
    )
    simX_acados, simU_acados, timings_acados, nlp_res_acados = run(
        use_RTI=False, use_l4acados=False, plot=True
    )
    (
        simX_l4acados_1sqp,
        simU_l4acados_1sqp,
        timings_l4acados_1sqp,
        nlp_res_l4acados_1sqp,
    ) = run(use_RTI=False, use_l4acados=True, plot=True, max_sqp_iter=1)
    (
        simX_acados_1sqp,
        simU_acados_1sqp,
        timings_acados_1sqp,
        nlp_res_acados_1sqp,
    ) = run(use_RTI=False, use_l4acados=False, plot=True, max_sqp_iter=1)
    simX_l4acados_rti, simU_l4acados_rti, timings_l4acados_rti, nlp_res_l4acados_rti = (
        run(use_RTI=True, use_l4acados=True, plot=True)
    )
    simX_acados_rti, simU_acados_rti, timings_acados_rti, nlp_res_acados_rti = run(
        use_RTI=True, use_l4acados=False, plot=True
    )
    # plt.show()

    atol = 1e-10
    rtol = 1e-6
    assert np.allclose(simX_l4acados, simX_acados, atol=atol, rtol=rtol)
    assert np.allclose(simU_l4acados, simU_acados, atol=atol, rtol=rtol)
    assert np.allclose(simX_l4acados_rti, simX_acados_rti, atol=atol, rtol=rtol)
    assert np.allclose(simU_l4acados_rti, simU_acados_rti, atol=atol, rtol=rtol)

    # acados check: *initial* residuals after 1 sqp iteration vs. after RTI
    assert np.allclose(
        nlp_res_acados_1sqp[:, 0, :], nlp_res_acados_rti[:, 0, :], atol=atol, rtol=rtol
    )
    assert np.allclose(nlp_res_l4acados, nlp_res_acados, atol=atol, rtol=rtol)
    assert np.allclose(nlp_res_l4acados_rti, nlp_res_acados_rti, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_minimal_example_closed_loop()
