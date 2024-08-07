# %%
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import DM, SX, MX, vertcat, sin, cos, Function
from scipy.linalg import block_diag


# %%
def export_simplependulum_ode_model(
    noise=False,
    only_lower_bounds=False,
    add_residual_dynamics=False,
    model_name="simplependulum_ode",
):
    # set up states & controls
    theta = SX.sym("theta")
    dtheta = SX.sym("dtheta")

    x = vertcat(theta, dtheta)
    u = SX.sym("u")

    # xdot
    theta_dot = SX.sym("theta_dot")
    dtheta_dot = SX.sym("dtheta_dot")
    xdot = vertcat(theta_dot, dtheta_dot)

    # parameters
    p = []
    if noise:
        w = SX.sym("w", 2, 1)
        p += [w]
        p = vertcat(*p)

    # dynamics
    f_expl = vertcat(dtheta, -sin(theta) + u)
    if noise:
        f_expl += w
    if add_residual_dynamics:
        f_expl += vertcat(DM(0), -0.5 * sin(theta**2))

    f_impl = xdot - f_expl

    # constraints
    if only_lower_bounds:
        con_h_expr = vertcat(theta, -theta)  # for lower bound  # for upper bound
    else:
        con_h_expr = vertcat(
            theta,  # theta
        )

    # acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.con_h_expr = con_h_expr
    # model.con_h_expr_e = con_h_expr
    model.name = model_name

    return model


def export_ocp_nominal(N, T, ocp_opts=None, only_lower_bounds=False, **model_kwargs):
    # constraints
    x0 = np.array([np.pi, 0])
    lb_u = -2.0
    ub_u = 2.0
    lb_theta = (0.0 / 360.0) * 2 * np.pi
    ub_theta = (200.0 / 360.0) * 2 * np.pi

    # cost
    cost_theta = 5
    cost_omega = 1
    cost_fac_e = 4
    Q = np.diagflat(np.array([cost_theta, cost_omega]))
    R = np.array(1)

    model = export_simplependulum_ode_model(
        only_lower_bounds=only_lower_bounds, **model_kwargs
    )

    # generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()

    ocp.model = model
    ocp.dims.N = N

    # dimensions
    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.nh = model.con_h_expr.shape[0]
    ocp.dims.np = model.p.shape[0] if isinstance(model.p, SX) else 0

    # cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = cost_fac_e * Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx : nx + nu, :] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = np.array([lb_u])
    ocp.constraints.ubu = np.array([ub_u])
    ocp.constraints.idxbu = np.array(range(nu))

    # TODO: automate this for arbitrary models (and tightened-constraint indices)
    if only_lower_bounds:
        inf_num = 1e6
        ocp.constraints.lh = np.array([lb_theta, -ub_theta])
        ocp.constraints.uh = np.array([inf_num, inf_num])
    else:
        ocp.constraints.lh = np.array([lb_theta])
        ocp.constraints.uh = np.array([ub_theta])

    # ocp.constraints.lh_e = ocp.constraints.lh
    # ocp.constraints.uh_e = ocp.constraints.uh

    # terminal constraints
    ocp.constraints.C_e = ocp.constraints.C
    ocp.constraints.lg_e = ocp.constraints.lg
    ocp.constraints.ug_e = ocp.constraints.ug

    ocp.constraints.x0 = x0

    # solver options

    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.integrator_type = 'DISCRETE' # 'IRK'

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.hessian_approx = 'EXACT'

    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.nlp_solver_type = 'SQP' # , SQP_RTI

    ocp.solver_options.tf = T
    ocp.solver_options.Tsim = T / N

    ocp.solver_options.tol = 1e-6
    ocp.solver_options.nlp_solver_max_iter = 30

    return ocp
