from casadi import SX, MX, vertcat
from acados_template import AcadosModel, AcadosSim
import torch
import gpytorch
import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import re
import copy

timings_names_default = [
    "build_lin_model",
    "query_nodes",
    "get_gp_sensitivities",
    "integrate_acados",
    "integrate_acados_python",
    "integrate_get",
    "integrate_set",
    "set_sensitivities",
    "set_sensitivities_reshape",
    "propagate_covar",
    "get_backoffs",
    "get_backoffs_htj_sig",
    "get_backoffs_htj_sig_matmul",
    "get_backoffs_add",
    "set_tightening",
    "phase_one",
    "check_termination",
    "solve_qp",
    "solve_qp_acados",
    # "total",
]

timings_names_raw = [
    "build_lin_model",
    "query_nodes",
    "get_gp_sensitivities",
    "integrate_acados",
    # "integrate_acados_python",
    "integrate_get",
    "integrate_set",
    "set_sensitivities",
    "set_sensitivities_reshape",
    "propagate_covar",
    # "get_backoffs",
    "get_backoffs_htj_sig",
    "get_backoffs_htj_sig_matmul",
    "get_backoffs_add",
    "set_tightening",
    "phase_one",
    "check_termination",
    # "solve_qp",
    "solve_qp_acados",
]

timings_names_backoffs = [
    "get_backoffs_htj_sig",
    "get_backoffs_htj_sig_matmul",
    "get_backoffs_add",
]


def set_ocp_options(ocp, ocp_opts):
    for key, value in ocp_opts.items():
        if isinstance(value, dict):
            set_ocp_options(getattr(ocp, key), value)
        elif hasattr(ocp, key):
            setattr(ocp, key, value)
            print(f"Set attribute {key} <= {value} of {ocp}")
        else:
            print(f"NOT FOUND: attribute {key} of {ocp}")

    return ocp


def export_linear_model(x, u, p):
    nx = x.shape[0]
    nu = u.shape[0]
    nparam = p.shape[0] if isinstance(p, SX) else 0

    # linear dynamics for every stage
    A = SX.sym("A", nx, nx)
    B = SX.sym("B", nx, nu)
    w = SX.sym("w", nx, 1)
    xdot = SX.sym("xdot", nx, 1)

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


def get_total_timings_per_task(solve_data: list):
    timings_per_task = {}
    for s in solve_data:
        for t_key, t_arr in s.timings.items():
            if not any([t_key == key for key in timings_per_task]):
                timings_per_task[t_key] = []
            timings_per_task[t_key] += [np.sum(t_arr)]
    return timings_per_task


def get_total_timings(solve_data: list, timings_names=timings_names_default):
    timings = []
    for s in solve_data:
        t_total = 0.0
        for t_key, t_arr in s.timings.items():
            if any([t_key == key for key in timings_names]):
                t_total += np.sum(t_arr)
        timings += [t_total]
    return np.array(timings)


def get_total_iter(solve_data: list):
    n_iter = []
    for s in solve_data:
        n_iter += [s.n_iter]
    return np.array(n_iter)


def plot_timings(solve_data: list, timings_names=timings_names_default):
    # plot timings as bar plot
    fig, ax_bars = plt.subplots()

    n_sim = len(solve_data)

    # x = list(solve_data[0].timings.keys())
    x = np.arange(0.5, len(solve_data) + 0.5, 1)
    y_old = np.zeros((n_sim,))
    for i, key in enumerate(timings_names):
        y = np.zeros((n_sim,))
        for j, s in enumerate(solve_data):
            y[j] = np.sum(s.timings[key])

        ax_bars.bar(x, y, bottom=y_old)
        if y_old is None:
            y_old = y
        else:
            y_old += y

    n_iter = []
    for j, s in enumerate(solve_data):
        n_iter += [s.n_iter]

    plt.legend(timings_names, bbox_to_anchor=(1.1, 1), loc="upper left")

    ax_iter = ax_bars.twinx()
    x_stairs = np.hstack((0.0, x + 0.5))
    ax_iter.stairs(n_iter, x_stairs)
    ax_iter.set_ylabel("n iter")

    plt.show()


def sym_mat2vec(mat):
    nx = mat.shape[0]

    if isinstance(mat, np.ndarray):
        i, j = np.triu_indices(nx, m=nx)
        return mat[i, j]
    elif isinstance(mat, torch.Tensor):
        i, j = torch.triu_indices(nx, nx)
        return mat[i, j]
    elif isinstance(mat, cas.DM):
        mat_np = np.array(mat)
        i, j = np.triu_indices(nx, m=nx)
        return cas.DM(mat_np[i, j])
    else:
        i, j = np.triu_indices(nx, m=nx)
        return mat[i, j]


def vec2sym_mat(vec, nx):

    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx, nx))
        i, j = np.triu_indices(nx, m=nx)
    elif isinstance(vec, torch.Tensor):
        mat = torch.zeros((nx, nx), device=vec.device)
        i, j = torch.triu_indices(nx, nx)
    else:
        mat = SX.zeros(nx, nx)
        i, j = np.triu_indices(nx, m=nx)

    mat[i, j] = vec
    mat.T[i, j] = vec

    return mat


def sym_mat2vec_old(mat):
    nx = mat.shape[0]

    if isinstance(mat, np.ndarray):
        vec = np.zeros((int((nx + 1) * nx / 2),))
    else:
        vec = cas.SX.zeros(int((nx + 1) * nx / 2))

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        vec[start_mat:end_mat] = mat[i:, i]
        start_mat += nx - i

    return vec


def vec2sym_mat_old(vec, nx):
    # nx = (vec.shape[0])

    if isinstance(vec, np.ndarray):
        mat = np.zeros((nx, nx))
    else:
        mat = cas.SX.zeros(nx, nx)

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        aux = vec[start_mat:end_mat]
        mat[i, i:] = aux.T
        mat[i:, i] = aux
        start_mat += nx - i

    return mat


def generate_gp_funs(gp_model, covar_jac=False, B=None):
    if gp_model.train_inputs[0].device.type == "cuda":
        to_tensor = lambda X: torch.Tensor(X).cuda()
        to_numpy = lambda T: T.cpu().numpy()
    else:
        to_tensor = lambda X: torch.Tensor(X)
        to_numpy = lambda T: T.numpy()

    if B is not None:
        B_tensor = to_tensor(B)

    def mean_fun_sum(y):
        with gpytorch.settings.fast_pred_var():
            return gp_model(y).mean.sum(dim=0)

    def covar_fun(y):
        with gpytorch.settings.fast_pred_var():
            return gp_model(y).variance

    def get_mean_dy(y, create_graph=False):
        with gpytorch.settings.fast_pred_var():
            mean_dy = torch.autograd.functional.jacobian(
                mean_fun_sum, y, create_graph=create_graph
            )
        return mean_dy

    def gp_sensitivities(y):
        # evaluate GP part (GP jacobians)
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(to_tensor(y), requires_grad=True)
            # DERIVATIVE
            mean_dy = to_numpy(
                torch.autograd.functional.jacobian(mean_fun_sum, y_tensor)
            )

            with torch.no_grad():
                predictions = gp_model(
                    y_tensor
                )  # only model (we want to find true function)
                mean = to_numpy(predictions.mean)
                variance = to_numpy(predictions.variance)

        return mean, mean_dy, variance

    def P_propagation_with_y(y, P_vec, A_nom, create_graph=False):
        variance = covar_fun(y)
        mean_dy = get_mean_dy(y, create_graph=create_graph)
        # P_vec_tensor = to_tensor(P_vec)
        # P_next = P_propagation(P, B @ A_GP, B, torch.diag(variance[0,:]))

        # no diag needed for variance
        nx_nom = B_tensor.shape[0]
        nx_vec = int((nx_nom + 1) * nx_nom / 2)

        N = y.shape[0]
        P_vec_prop = torch.zeros((N, nx_vec))
        for i in range(N):
            A_GP = mean_dy[:, i, 0:nx_nom]
            P = vec2sym_mat(P_vec[i, :], nx_nom)
            A_prop = A_nom[i, :, :] + B_tensor @ A_GP
            P_prop = (
                A_prop @ P @ A_prop.T
                + B_tensor @ torch.diag(variance[i, :]) @ B_tensor.T
            )
            P_vec_prop[i, :] = sym_mat2vec(P_prop)

        return P_vec_prop

    def gp_sensitivities_with_prop(y, P, A_nom, A_nom_dy):
        # evaluate GP part (GP jacobians)
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(to_tensor(y), requires_grad=True)
            P_tensor = torch.autograd.Variable(to_tensor(P), requires_grad=True)
            A_tensor = torch.autograd.Variable(to_tensor(A_nom), requires_grad=True)
            A_dy_tensor = torch.autograd.Variable(
                to_tensor(A_nom_dy), requires_grad=False
            )
            # DERIVATIVE
            mean_dy = to_numpy(
                torch.autograd.functional.jacobian(mean_fun_sum, y_tensor)
            )
            prop_dy_partial = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda y: P_propagation_with_y(
                        y, P_tensor, A_tensor, create_graph=True
                    ).sum(dim=0),
                    y_tensor,
                )
            )
            prop_dA = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda A: P_propagation_with_y(
                        y_tensor, P_tensor, A, create_graph=True
                    ).sum(dim=0),
                    A_tensor,
                )
            )
            prop_dP = to_numpy(
                torch.autograd.functional.jacobian(
                    lambda P: P_propagation_with_y(
                        y_tensor, P, A_tensor, create_graph=True
                    ).sum(dim=0),
                    P_tensor,
                )
            )
            prop_dA_dy = np.transpose(
                np.diagonal(
                    np.tensordot(prop_dA, A_nom_dy, ([2, 3], [1, 2])), axis1=1, axis2=2
                ),
                [0, 2, 1],
            )
            prop_dy = prop_dy_partial + prop_dA_dy

            with torch.no_grad():
                predictions = gp_model(
                    y_tensor
                )  # only model (we want to find true function)
                mean = to_numpy(predictions.mean)
                prop = to_numpy(P_propagation_with_y(y_tensor, P_tensor, A_tensor))

        return mean, mean_dy, prop, prop_dy, prop_dP

    if covar_jac:
        return gp_sensitivities_with_prop
    else:
        return gp_sensitivities


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


def P_propagation(P, A, B, W):
    #  P_i+1 = A P A^T +  B*W*B^T
    return A @ P @ A.T + B @ W @ B.T


def generate_h_tighten_jac_sig_from_h_tighten(h_tight, x, u, p, sig):
    # JIT options
    jit_options = {"flags": ["-O3"], "verbose": True}
    fun_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}

    h_tighten_jac_sig = cas.jacobian(h_tight, sig)

    p_sig = cas.vertcat(sig, p)
    h_tighten_jac_sig_fun = cas.Function(
        "h_tighten_jac_sig", [x, u, p_sig], [h_tighten_jac_sig], fun_options
    )

    return h_tighten_jac_sig_fun


def generate_h_tightening_funs_SX(h, x, u, p, idh_tight):
    """
    h: cas.MX or cas.SX expression for contraints to be tightened depending on x
    x: cas.MX or cas.SX variable
    """
    # JIT options
    jit_options = {"flags": ["-O3"], "verbose": True}
    fun_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}

    # dims
    nx = x.shape[0]
    nu = u.shape[0]
    np = p.shape[0] if isinstance(p, cas.SX) else 0
    nh = h.shape[0]

    # variables for uncertainty and gradients
    sig_vec = cas.SX.sym("sig_vec", int((nx + 1) * nx / 2))
    sig = vec2sym_mat_old(sig_vec, nx)

    h_fun = cas.Function("h", [x, u, p], [h])
    h_jac_x = cas.jacobian(h, x)

    h_tighten_stack = cas.SX.zeros(nh, 1)
    for i in idh_tight:
        h_tighten_stack[i, :] = cas.sqrt(cas.dot(h_jac_x[i, :] @ sig, h_jac_x[i, :]))

    h_tighten_jac_x = cas.jacobian(h_tighten_stack, x)
    h_tighten_jac_sig = cas.jacobian(h_tighten_stack, sig_vec)

    p_sig = cas.vertcat(sig_vec, p)

    h_jac_x_fun = cas.Function("h_jac_x", [x, u, p_sig], [h_jac_x], fun_options)
    h_tighten_fun = cas.Function(
        "h_tighten", [x, u, p_sig], [h_tighten_stack], fun_options
    )
    h_tighten_jac_x_fun = cas.Function(
        "h_tighten_jac_x", [x, u, p_sig], [h_tighten_jac_x], fun_options
    )
    h_tighten_jac_sig_fun = cas.Function(
        "h_tighten_jac_sig", [x, u, p_sig], [h_tighten_jac_sig], fun_options
    )

    return p_sig, h_jac_x_fun, h_tighten_fun, h_tighten_jac_x_fun, h_tighten_jac_sig_fun


def only_upper_bounds_expr(h):
    h_only_upper = cas.vertcat(-h, h)  # lower  # upper
    idx_lh = np.arange(h.shape[0])
    idx_uh = len(idx_lh) + idx_lh
    return h_only_upper, idx_lh, idx_uh


def tighten_model_constraints(model, idh_tight, prob_x):
    h = model.con_h_expr
    x = model.x
    u = model.u
    p = model.p

    p_sig, h_jac_x_fun, h_tighten_fun, h_tighten_jac_x_fun, h_tighten_jac_sig_fun = (
        generate_h_tightening_funs_SX(h, x, u, p, idh_tight)
    )
    model.p = p_sig

    prob_tighten = norm.ppf(prob_x)
    for ih in idh_tight:
        model.con_h_expr[ih] = (
            model.con_h_expr[ih] - prob_tighten * h_tighten_fun(x, u, p_sig)[ih, :]
        )

    return model, h_jac_x_fun, h_tighten_fun, h_tighten_jac_x_fun, h_tighten_jac_sig_fun


def propagate(P0, Afun, B, Wfun, y_all, N_sim):
    P_return = [np.zeros(P0.shape)] * (N_sim + 1)
    P_return[0] = P0
    for i in range(N_sim):
        y = y_all[i, :]
        P_return[i + 1] = P_propagation(P_return[i], Afun(y), B, Wfun(y))
    return P_return


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

    sim.solver_options.T = ocp.solver_options.Tsim
    sim.solver_options.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim.solver_options.newton_tol = ocp.solver_options.sim_method_newton_tol
    sim.solver_options.num_stages = array_to_int(
        ocp.solver_options.sim_method_num_stages
    )
    sim.solver_options.num_steps = array_to_int(ocp.solver_options.sim_method_num_steps)

    return sim


def array_to_int(arr):
    value = copy.deepcopy(arr)
    if type(value) is list or type(value) is np.ndarray:
        assert all(value == value[0])
        value = value[0]

    return int(value)
