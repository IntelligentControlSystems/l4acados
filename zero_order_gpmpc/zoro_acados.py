import os, sys
import numpy as np
import casadi as cas

import torch
import gpytorch

from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
from .zoro_acados_utils import *

from time import perf_counter
from dataclasses import dataclass

@dataclass
class ZoroAcadosData:
    n_iter: int
    sol_x: np.ndarray
    sol_u: np.ndarray 
    sol_P: np.ndarray
    timings_total: float
    timings: dict

class ZoroAcados():
    def __init__(self, ocp, sim, prob_x, Sigma_x0, Sigma_W, 
        B=None, 
        gp_model=None, 
        use_cython=True, 
        h_tightening_jac_sig_fun=None,
        h_tightening_idx=[],
        path_json_ocp="zoro_ocp_solver_config.json",
        path_json_sim="zoro_sim_solver_config.json"
    ):
        """
        ocp: AcadosOcp for nominal problem
        sim: AcadosSim for nominal model
        gp_model: GPyTorch GP model
        """
        # get dimensions
        nx = ocp.dims.nx
        nu = ocp.dims.nu
        nparam = ocp.dims.np
        N = ocp.dims.N
        T = ocp.solver_options.tf

        self.nx = nx
        self.nu = nu
        self.nparam = nparam
        self.nparam_model = nparam - int((self.nx+1)*self.nx/2)
        self.N = N
        self.T = T
        self.sim = sim

        # constants
        if B is None:
            B = np.eye(self.nx)
        
        self.nw = B.shape[1]
        self.B = B
        self.Sigma_x0 = Sigma_x0
        self.Sigma_W = Sigma_W

        # allocation
        self.x_hat_all = np.zeros((N+1, nx))
        self.u_hat_all = np.zeros((N, nu))
        self.y_hat_all = np.zeros((N,nx+nu))
        self.P_bar_all = [None] * (N+1)
        self.P_bar_all_vec = [None] * (N+1)
        self.P_bar_old_vec = None
        self.P_bar_all[0] = Sigma_x0
        self.P_bar_all_vec[0] = sym_mat2vec(Sigma_x0)
        self.mean = np.zeros((N, self.nw))
        self.mean_dy = np.zeros((self.nw, N, nx+nu))
        self.var = np.zeros((N,self.nw))

        # TODO: allow for more general model structures (other params than just vectorized covariances)
        if h_tightening_jac_sig_fun is None:
            # TODO: general solution (problem is concatenated paramteres in uncertain model, cannot compute jacobian w.r.t. subset of variables)
            h_jac_x_fun, h_tighten_fun, h_tighten_jac_x_fun, h_tighten_jac_sig_fun = generate_h_tightening_funs_SX(ocp.model.con_h_expr, ocp.model.x, ocp.model.u, ocp.model.p, h_tightening_idx)
            self.h_tightening_jac_sig_fun = h_tighten_jac_sig_fun
            # self.h_tightening_jac_sig_fun = generate_h_tighten_jac_sig_from_h_tighten(ocp.model.con_h_expr, ocp.model.x, ocp.model.u, ocp.model.p)
        # else:
        
        self.h_tightening_jac_sig_fun = h_tightening_jac_sig_fun

        self.ocp = transform_ocp(ocp)
        self.nparam_zoro = self.ocp.dims.np
        self.p_hat_model = np.zeros((N,self.nparam_model))
        self.p_hat_model_with_Pvec = np.zeros((N+1,self.nparam))
        self.p_hat_all = np.zeros((N,self.nparam_zoro))

        self.p_hat_model_with_Pvec[0,:int((self.nx+1)*self.nx/2)] = self.P_bar_all_vec[0]

        if use_cython:
            AcadosOcpSolver.generate(self.ocp, json_file = path_json_ocp)
            AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
            self.ocp_solver = AcadosOcpSolver.create_cython_solver(path_json_ocp)

            AcadosSimSolver.generate(self.sim, json_file = path_json_sim)
            AcadosSimSolver.build(self.sim.code_export_directory, with_cython=True)
            self.sim_solver = AcadosSimSolver.create_cython_solver(path_json_sim)
        else:
            self.ocp_solver = AcadosOcpSolver(self.ocp, json_file = path_json_ocp)
            self.sim_solver = AcadosSimSolver(self.sim, json_file = path_json_sim)
        
        if gp_model is None:
            self.has_gp_model = False
        else:
            self.has_gp_model = True
            self.gp_model = gp_model
            self.gp_sensitivities = generate_gp_funs(gp_model)

        # timings
        self.solve_stats_default = {
            "n_iter": 0,
            "timings_total": 0.0,
            "timings": {
                "build_lin_model": 0.0,
                "query_nodes": 0.0,
                "get_gp_sensitivities": 0.0,
                "integrate_acados": 0.0,
                "integrate_acados_python": 0.0,
                "integrate_get": 0.0,
                "integrate_set": 0.0,
                "set_sensitivities": 0.0,
                "set_sensitivities_reshape": 0.0,
                "propagate_covar": 0.0,
                "get_backoffs": 0.0,
                "get_backoffs_htj_sig": 0.0,
                "get_backoffs_htj_sig_matmul": 0.0,
                "get_backoffs_add": 0.0,
                "set_tightening": 0.0,
                "phase_one": 0.0,
                "check_termination": 0.0,
                "solve_qp": 0.0,
                "solve_qp_acados": 0.0,
                "total": 0.0,
            }
        }
        self.solve_stats = self.solve_stats_default.copy()

    def solve(self, tol_nlp=1e-6, n_iter_max=30):
        time_total = perf_counter()
        self.init_solve_stats(n_iter_max)

        nx = self.nx
        nu = self.nu
        nparam = self.nparam
        nparam_zoro = self.nparam_zoro
        nw = self.nw
        N = self.N

        for i in range(n_iter_max):
            time_iter = perf_counter()
            # ------------------- Query nodes --------------------
            time_query_nodes = perf_counter()
            # preparation rti_phase (solve() AFTER setting params to get right Jacobians)
            self.ocp_solver.options_set('rti_phase', 1)

            # get sensitivities for all stages
            for stage in range(self.N):
                # current stage values
                self.x_hat_all[stage,:] = self.ocp_solver.get(stage,"x")   
                self.u_hat_all[stage,:] = self.ocp_solver.get(stage,"u")   
                self.y_hat_all[stage,:] = np.hstack((self.x_hat_all[stage,:],self.u_hat_all[stage,:])).reshape((1,nx+nu))   

            self.solve_stats["timings"]["query_nodes"][i] += perf_counter() - time_query_nodes
            
            # ------------------- GP Sensitivities --------------------
            time_get_gp_sensitivities = perf_counter()

            if self.has_gp_model:
                torch.cuda.synchronize()
                self.mean, self.mean_dy, self.var = self.gp_sensitivities(self.y_hat_all)
                torch.cuda.synchronize()

            self.solve_stats["timings"]["get_gp_sensitivities"][i] += perf_counter() - time_get_gp_sensitivities
            
            # ------------------- Update stages --------------------
            for stage in range(self.N):
                # set parameters (linear matrices and offset)
                # ------------------- Integrate --------------------
                time_integrate_set = perf_counter()
                self.sim_solver.set("x", self.x_hat_all[stage,:])
                self.sim_solver.set("u", self.u_hat_all[stage,:])
                self.sim_solver.set("p", self.p_hat_model_with_Pvec[stage,:])
                self.solve_stats["timings"]["integrate_set"][i] += perf_counter() - time_integrate_set

                time_integrate_acados_python = perf_counter()
                status_integrator = self.sim_solver.solve()
                self.solve_stats["timings"]["integrate_acados_python"][i] += perf_counter() - time_integrate_acados_python
                self.solve_stats["timings"]["integrate_acados"][i] += self.sim_solver.get("time_tot")

                time_integrate_get = perf_counter()
                A_nom = self.sim_solver.get("Sx")
                B_nom = self.sim_solver.get("Su")
                x_nom = self.sim_solver.get("x")
                self.solve_stats["timings"]["integrate_get"][i] += perf_counter() - time_integrate_get

                # ------------------- Build linear model --------------------
                time_build_lin_model = perf_counter()

                A_total = A_nom + self.B @ self.mean_dy[:,stage,0:nx]
                B_total = B_nom + self.B @ self.mean_dy[:,stage,nx:nx+nu]

                f_hat = x_nom + self.B @ self.mean[stage,:] \
                    - A_total @ self.x_hat_all[stage,:] - B_total @ self.u_hat_all[stage,:]

                self.solve_stats["timings"]["build_lin_model"][i] += perf_counter() - time_build_lin_model
                
                # ------------------- Propagate --------------------
                time_propagate_covar = perf_counter()

                # delta-Values
                if self.P_bar_old_vec is None:
                    # i == 0
                    dP_bar_vec = cas.DM.zeros((int((nx+1)*nx/2),1))
                else:
                    dP_bar_vec = self.P_bar_all_vec[stage] - self.P_bar_old_vec

                self.P_bar_all[stage+1] = P_propagation(self.P_bar_all[stage], A_total, self.B, self.Sigma_W + np.diag(self.var[stage,:]))
                
                self.P_bar_old_vec = self.P_bar_all_vec[stage+1] # used in next iter
                self.P_bar_all_vec[stage+1] = sym_mat2vec(self.P_bar_all[stage+1])
                self.p_hat_model_with_Pvec[stage+1,:int((self.nx+1)*self.nx/2)] = self.P_bar_all_vec[stage+1]

                self.solve_stats["timings"]["propagate_covar"][i] += perf_counter() - time_propagate_covar

                # ------------------- Set sensitivities --------------------
                time_set_sensitivities_reshape = perf_counter()

                A_reshape = np.reshape(A_total,(nx**2),order="F")
                B_reshape = np.reshape(B_total,(nx*nu),order="F")
                
                self.solve_stats["timings"]["set_sensitivities_reshape"][i] += perf_counter() - time_set_sensitivities_reshape
                time_set_sensitivities = perf_counter()

                self.p_hat_all[stage,:] = np.hstack((
                    A_reshape,
                    B_reshape,
                    f_hat,
                    self.P_bar_all_vec[stage],
                    self.p_hat_model[stage,:]
                ))
                self.ocp_solver.set(stage, "p", self.p_hat_all[stage,:])
                self.solve_stats["timings"]["set_sensitivities"][i] += perf_counter() - time_set_sensitivities

                # ------------------- Compute back off --------------------
                time_get_backoffs = perf_counter()

                time_get_backoffs_htj_sig = perf_counter()
                p_sig = np.hstack((
                    self.P_bar_all_vec[stage], 
                    self.p_hat_model[i,:]
                ))
                htj_sig = self.h_tightening_jac_sig_fun(self.x_hat_all[stage,:], self.u_hat_all[stage,:], p_sig)          
                self.solve_stats["timings"]["get_backoffs_htj_sig"][i] += perf_counter() - time_get_backoffs_htj_sig
                
                time_get_backoffs_htj_sig_matmul = perf_counter()
                htj_sig_matmul = htj_sig @ dP_bar_vec         
                self.solve_stats["timings"]["get_backoffs_htj_sig_matmul"][i] += perf_counter() - time_get_backoffs_htj_sig_matmul

                time_get_backoffs_add = perf_counter()
                tightening = htj_sig_matmul

                lh = cas.DM(self.ocp.constraints.lh) + tightening 

                time_now = perf_counter()
                self.solve_stats["timings"]["get_backoffs_add"][i] = time_now - time_get_backoffs_add
                self.solve_stats["timings"]["get_backoffs"][i] += time_now - time_get_backoffs
                
                # ------------------- Set tightening --------------------
                time_set_tightening = perf_counter()

                # set constraints
                if stage > 0:
                    self.ocp_solver.constraints_set(stage,"lh",lh.full().flatten())

                self.solve_stats["timings"]["set_tightening"][i] += perf_counter() - time_set_tightening
                # self.solve_stats["timings"]["set_tightening_raw"][i] += t_set_C + t_set_lg + t_set_ug

            # feedback rti_phase
            # self.ocp_solver.options_set('rti_phase', 1)
            # ------------------- Phase 1 --------------------
            time_phase_one = perf_counter()
            status = self.ocp_solver.solve()
            self.solve_stats["timings"]["phase_one"][i] += perf_counter() - time_phase_one

            # ------------------- Solve QP --------------------
            time_solve_qp = perf_counter()

            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()
                
            self.solve_stats["timings"]["solve_qp"][i] += perf_counter() - time_solve_qp
            self.solve_stats["timings"]["solve_qp_acados"][i] += self.ocp_solver.get_stats("time_tot")

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            time_check_termination = perf_counter()
            
            # self.ocp_solver.print_statistics() # encapsulates: stat = self.ocp_solver.get_stats("statistics")
            residuals = self.ocp_solver.get_residuals()
            print("residuals after ", i, "SQP_RTI iterations:\n", residuals)

            self.solve_stats["timings"]["check_termination"][i] += perf_counter() - time_check_termination
            self.solve_stats["timings"]["total"][i] += perf_counter() - time_iter

            if status != 0:
                raise Exception('acados self.ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

            if max(residuals) < tol_nlp:
                break
        
        self.solve_stats["n_iter"] = i + 1
        self.solve_stats["timings_total"] = perf_counter() - time_total

    def set_model_params(self, i, p_model):
        self.p_hat_model[i,:] = p_model
        self.p_hat_model_with_Pvec[i,-self.nparam_model:] = p_model
        # print(f"Set model params (i={i}): p_hat_model={self.p_hat_model[i,:]},\nwith_Pvec={self.p_hat_model_with_Pvec[i,:]}")

    def get_solution(self):
        X = np.zeros((self.N+1, self.nx))
        U = np.zeros((self.N, self.nu))

        # get data
        for i in range(self.N):
            X[i,:] = self.ocp_solver.get(i, "x")
            U[i,:] = self.ocp_solver.get(i, "u")

        X[self.N,:] = self.ocp_solver.get(self.N, "x")

        return X,U,self.P_bar_all

    def print_solve_stats(self):
        n_iter = self.solve_stats["n_iter"]

        time_other = 0.0
        for key, t_arr in self.solve_stats["timings"].items():
            for i in range(n_iter):
                t_sum = np.sum(t_arr[0:n_iter])
                t_avg = t_sum / n_iter
                t_max = np.max(t_arr[0:n_iter])
                t_min = np.min(t_arr[0:n_iter])
                if key != "integrate_acados":
                    if key != "total":
                        time_other -= t_sum
                    else:
                        time_other += t_sum
            print(f"{key:20s}: {1000*t_sum:8.3f}ms ({n_iter} calls), {1000*t_avg:8.3f}/{1000*t_max:8.3f}/{1000*t_min:8.3f}ms (avg/max/min per call)")

        key = "other"
        t = time_other
        print("----------------------------------------------------------------")
        print(f"{key:20s}: {1000*t:8.3f}ms ({n_iter} calls), {1000*t/n_iter:8.3f}ms (1 call)")
    
    def init_solve_stats(self, max_iter):
        self.solve_stats = self.solve_stats_default.copy()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = np.zeros((max_iter,))
    
    def get_solve_stats(self):
        X,U,P = self.get_solution()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = self.solve_stats["timings"][k][0:self.solve_stats["n_iter"]]

        return ZoroAcadosData(
            self.solve_stats["n_iter"], 
            X, 
            U, 
            P, 
            self.solve_stats["timings_total"],
            self.solve_stats["timings"].copy()
        )
