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
import sys, os
import numpy as np
import scipy.linalg
from casadi import vertcat
from zero_order_gpmpc.controllers import ResidualLearningMPC
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from .pendulum_model import export_pendulum_ode_model
from .utils import plot_pendulum


def create_ocp(x0, Fmax, N_horizon, Tf, RTI=False):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()
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

    if RTI:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 1
    else:
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # turns on globalization
        )
        ocp.solver_options.nlp_solver_max_iter = 150

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    return ocp


def setup_acados(ocp, json_file):
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file=json_file)
    return acados_ocp_solver, acados_integrator


def run(use_RTI=False, use_l4acados=False, plot=True):

    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    Fmax = 80

    Tf = 0.8
    N_horizon = 40

    ocp = create_ocp(x0, Fmax, N_horizon, Tf, use_RTI)
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

    Nsim = 100
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0, :] = x0

    t_preparation = np.zeros((Nsim))
    t_feedback = np.zeros((Nsim))
    t_total = np.zeros((Nsim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar=x0)

    # closed loop
    for i in range(Nsim):

        if use_RTI and not use_l4acados:
            # preparation phase
            ocp_solver.options_set("rti_phase", 1)
            status = ocp_solver.solve()
            t_preparation[i] = ocp_solver.get_stats("time_tot")

            # set initial state
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp_solver.options_set("rti_phase", 2)
            status = ocp_solver.solve()
            t_feedback[i] = ocp_solver.get_stats("time_tot")

            simU[i, :] = ocp_solver.get(0, "u")

        else:
            # solve ocp and get next control input
            simU[i, :] = ocp_solver.solve_for_x0(x0_bar=simX[i, :])

            t_total[i] = ocp_solver.get_stats("time_tot")

        # simulate system
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    if use_RTI:
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        t_total = t_preparation + t_feedback
        print(
            f"Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}"
        )
        print(
            f"Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}"
        )

    # scale to milliseconds
    t_total *= 1000
    print(
        f"Computation time (total) in ms: min {np.min(t_total):.3f} median {np.median(t_total):.3f} max {np.max(t_total):.3f}"
    )

    if use_RTI:
        timings = {
            "preparation": t_preparation,
            "feedback": t_feedback,
            "total": t_total,
        }
    else:
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

    ocp_solver = None

    return simX, simU, timings


def test_minimal_example_closed_loop(show_plots=False):
    simX_l4acados, simU_l4acados, timings_l4acados = run(
        use_RTI=False, use_l4acados=True, plot=True
    )
    simX_acados, simU_acados, timings_acados = run(
        use_RTI=False, use_l4acados=False, plot=True
    )
    simX_l4acados_rti, simU_l4acados_rti, timings_l4acados_rti = run(
        use_RTI=True, use_l4acados=True, plot=True
    )
    simX_acados_rti, simU_acados_rti, timings_acados_rti = run(
        use_RTI=True, use_l4acados=False, plot=True
    )

    if show_plots:
        plt.show()

    atol = 1e-5
    rtol = 1e-3
    assert np.allclose(simX_l4acados, simX_acados, atol=atol, rtol=rtol)
    assert np.allclose(simU_l4acados, simU_acados, atol=atol, rtol=rtol)
    assert np.allclose(simX_l4acados_rti, simX_acados_rti, atol=atol, rtol=rtol)
    assert np.allclose(simU_l4acados_rti, simU_acados_rti, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_minimal_example_closed_loop()
