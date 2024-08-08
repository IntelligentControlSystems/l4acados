from casadi import SX, MX, vertcat
from acados_template import AcadosModel, AcadosSim
import numpy as np
from copy import deepcopy
import re


def export_linear_model(x, u, p):
    nx = x.shape[0]
    nu = u.shape[0]

    if isinstance(x, MX):
        VARX = MX
    elif isinstance(x, SX):
        VARX = SX
    else:
        raise ValueError("x must be of type SX or MX")

    nparam = p.shape[0] if isinstance(p, VARX) else 0

    # linear dynamics for every stage
    A = VARX.sym("A", nx, nx)
    B = VARX.sym("B", nx, nu)
    w = VARX.sym("w", nx, 1)
    xdot = VARX.sym("xdot", nx, 1)

    f_expl = A @ x + B @ u + w
    f_impl = xdot - f_expl

    # parameters
    p_lin = vertcat(
        A.reshape((nx**2, 1)), B.reshape((nx * nu, 1)), w, p  # (P_vec, p_nom)
    )

    # acados model
    model = AcadosModel()
    model.disc_dyn_expr = f_expl
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p_lin
    model.name = f"linear_model_with_params_nx{nx}_nu{nu}_np{nparam}"

    return model


def transform_ocp(ocp_input):
    ocp = deepcopy(ocp_input)

    original_nparam = ocp.dims.np

    model_lin = export_linear_model(ocp.model.x, ocp.model.u, ocp.model.p)
    ocp.model.disc_dyn_expr = model_lin.disc_dyn_expr
    ocp.model.f_impl_expr = model_lin.f_impl_expr
    ocp.model.f_expl_expr = model_lin.f_expl_expr
    ocp.model.x = model_lin.x
    ocp.model.xdot = model_lin.xdot
    ocp.model.u = model_lin.u
    ocp.model.p = model_lin.p
    ocp.model.name = model_lin.name
    ocp.dims.np = model_lin.p.shape[0]

    ocp_parameter_values = ocp.parameter_values
    ocp.parameter_values = np.zeros((model_lin.p.shape[0],))
    if original_nparam > 0:
        ocp.parameter_values[-original_nparam:] = ocp_parameter_values

    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    return ocp


def setup_sim_from_ocp(ocp):
    # integrator for nominal model
    sim = AcadosSim()

    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values

    for opt_name in dir(ocp.solver_options):
        if (
            opt_name in dir(sim.solver_options)
            and re.search(r"__.*?__", opt_name) is None
        ):
            set_value = getattr(ocp.solver_options, opt_name)

            if opt_name == "sim_method_jac_reuse":
                set_value = array_to_int(set_value)

            print(f"Setting {opt_name} to {set_value}")
            setattr(sim.solver_options, opt_name, set_value)

    if ocp.solver_options.Tsim is not None and ocp.solver_options.tf is not None:
        assert np.isclose(
            ocp.solver_options.tf / ocp.dims.N, ocp.solver_options.Tsim
        ), "Both tf and Tsim are set in ocp, it should hold that (tf / N) == Tsim."

    if ocp.solver_options.Tsim is not None:
        sim.solver_options.T = ocp.solver_options.Tsim
    else:
        sim.solver_options.T = ocp.solver_options.tf / ocp.dims.N

    sim.solver_options.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim.solver_options.newton_tol = ocp.solver_options.sim_method_newton_tol
    sim.solver_options.num_stages = array_to_int(
        ocp.solver_options.sim_method_num_stages
    )
    sim.solver_options.num_steps = array_to_int(ocp.solver_options.sim_method_num_steps)
    sim.code_export_directory = ocp.code_export_directory

    return sim


def get_solve_opts_from_ocp(ocp):
    solve_opts = {}
    for opt_name in dir(ocp.solver_options):
        if re.search(r"__.*?__", opt_name) is not None:
            continue
        if re.search(r"_AcadosOcpOptions__.*?", opt_name) is not None:
            continue

        try:
            set_value = getattr(ocp.solver_options, opt_name)
        except Exception as e:
            print(f"Error getting attribute: {e}")
            set_value = None

        print(f"Getting: {opt_name} = {set_value}")
        solve_opts[opt_name] = set_value

    return solve_opts


def array_to_int(arr):
    value = deepcopy(arr)
    if type(value) is list or type(value) is np.ndarray:
        assert all(value == value[0])
        value = value[0]

    return int(value)
