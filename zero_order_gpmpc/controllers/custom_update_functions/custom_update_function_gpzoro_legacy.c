/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// This is a template based custom_update function
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "custom_update_function.h"
#include "acados_solver_pacejka_model.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados/utils/mem.h"

#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blasfeo_api.h"

typedef struct custom_memory
{
  // covariance matrics
  struct blasfeo_dmat* uncertainty_matrix_buffer;  // shape = (N+1, nx, nx)
  // covariance matrix of the additive disturbance
  struct blasfeo_dmat W_mat;          // shape = (nw, nw)
  struct blasfeo_dmat unc_jac_G_mat;  // shape = (nx, nw)
  struct blasfeo_dmat temp_GW_mat;    // shape = (nx, nw)
  struct blasfeo_dmat GWG_mat;        // shape = (nx, nx)

  // covariance matrix of the additive GP disturbance
  struct blasfeo_dmat W_gp_mat;  // shape = (nw, nw)
  // struct blasfeo_dmat temp_GW_gp_mat;    // shape = (nx, nw)
  // struct blasfeo_dmat GW_gpG_mat;        // shape = (nx, nx)
  // struct blasfeo_dmat GW_gpG_p_GWG_mat;  // shape = (nx, nx)
  // // sensitivity matrices
  struct blasfeo_dmat A_mat;  // shape = (nx, nx)
  struct blasfeo_dmat B_mat;  // shape = (nx, nu)
  // matrix in linear constraints
  struct blasfeo_dmat Cg_mat;           // shape = (ng, nx)
  struct blasfeo_dmat Dg_mat;           // shape = (ng, nu)
  struct blasfeo_dmat Cg_e_mat;         // shape = (ng_e, nx)
  struct blasfeo_dmat dummy_Dgh_e_mat;  // shape = (ngh_e_max, nu)
  // matrix in nonlinear constraints
  struct blasfeo_dmat Ch_mat;    // shape = (nh, nx)
  struct blasfeo_dmat Dh_mat;    // shape = (nh, nu)
  struct blasfeo_dmat Ch_e_mat;  // shape = (nh_e, nx)
  // feedback gain matrix
  struct blasfeo_dmat K_mat;  // shape = (nu, nx)
  // AK = A - B@K
  struct blasfeo_dmat AK_mat;  // shape = (nx, nx)
  // A@P_k
  struct blasfeo_dmat temp_AP_mat;  // shape = (nx, nx)
  // K@P_k, K@P_k@K^T
  struct blasfeo_dmat temp_KP_mat;   // shape = (nu, nx)
  struct blasfeo_dmat temp_KPK_mat;  // shape = (nu, nu)
  // C + D @ K, (C + D @ K) @ P_k
  struct blasfeo_dmat temp_CaDK_mat;    // shape = (ngh_me_max, nx)
  struct blasfeo_dmat temp_CaDKmP_mat;  // shape = (ngh_me_max, nx)
  struct blasfeo_dmat temp_beta_mat;    // shape = (ngh_me_max, ngh_me_max)

  double* d_A_mat;      // shape = (nx, nx)
  double* d_B_mat;      // shape = (nx, nu)
  double* d_Cg_mat;     // shape = (ng, nx)
  double* d_Dg_mat;     // shape = (ng, nu)
  double* d_Cg_e_mat;   // shape = (ng_e, nx)
  double* d_Cgh_mat;    // shape = (ng+nh, nx)
  double* d_Dgh_mat;    // shape = (ng+nh, nu)
  double* d_Cgh_e_mat;  // shape = (ng_e+nh_e, nx)
  double* d_state_vec;
  // upper and lower bounds on state variables
  double* d_lbx;    // shape = (nbx,)
  double* d_ubx;    // shape = (nbx,)
  double* d_lbx_e;  // shape = (nbx_e,)
  double* d_ubx_e;  // shape = (nbx_e,)
  // tightened upper and lower bounds on state variables
  double* d_lbx_tightened;    // shape = (nbx,)
  double* d_ubx_tightened;    // shape = (nbx,)
  double* d_lbx_e_tightened;  // shape = (nbx_e,)
  double* d_ubx_e_tightened;  // shape = (nbx_e,)
  // upper and lower bounds on control inputs
  double* d_lbu;  // shape = (nbu,)
  double* d_ubu;  // shape = (nbu,)
  // tightened upper and lower bounds on control inputs
  double* d_lbu_tightened;  // shape = (nbu,)
  double* d_ubu_tightened;  // shape = (nbu,)
  // upper and lower bounds on polytopic constraints
  double* d_lg;    // shape = (ng,)
  double* d_ug;    // shape = (ng,)
  double* d_lg_e;  // shape = (ng_e,)
  double* d_ug_e;  // shape = (ng_e,)
  // tightened lower bounds on polytopic constraints
  double* d_lg_tightened;    // shape = (ng,)
  double* d_ug_tightened;    // shape = (ng,)
  double* d_lg_e_tightened;  // shape = (ng_e,)
  double* d_ug_e_tightened;  // shape = (ng_e,)
  // upper and lower bounds on nonlinear constraints
  double* d_lh;    // shape = (nh,)
  double* d_uh;    // shape = (nh,)
  double* d_lh_e;  // shape = (nh_e,)
  double* d_uh_e;  // shape = (nh_e,)
  // tightened upper and lower bounds on nonlinear constraints
  double* d_lh_tightened;    // shape = (nh,)
  double* d_uh_tightened;    // shape = (nh,)
  double* d_lh_e_tightened;  // shape = (nh_e,)
  double* d_uh_e_tightened;  // shape = (nh_e,)

  int* idxbx;    // shape = (nbx,)
  int* idxbu;    // shape = (nbu,)
  int* idxbx_e;  // shape = (nbx_e,)

  void* raw_memory;  // Pointer to allocated memory, to be used for freeing
} custom_memory;

static int int_max(int num1, int num2)
{
  return (num1 > num2) ? num1 : num2;
}

static int custom_memory_calculate_size(ocp_nlp_config* nlp_config, ocp_nlp_dims* nlp_dims)
{
  int N = nlp_dims->N;
  int nx = 9;
  int nu = 3;
  int nw = 9;

  int ng = 0;
  int nh = 1;
  int nbx = 9;
  int nbu = 3;

  int ng_e = 0;
  int nh_e = 0;
  int ngh_e_max = int_max(ng_e, nh_e);
  int ngh_me_max = int_max(ngh_e_max, int_max(ng, nh));
  int nbx_e = 0;

  assert(0 <= nbx);
  assert(0 <= nbx);
  assert(0 <= nbu);
  assert(0 <= nbu);
  assert(0 <= ng);
  assert(0 <= ng);
  assert(0 <= nh);
  assert(1 <= nh);
  assert(0 <= nbx_e);
  assert(0 <= nbx_e);
  assert(0 <= ng_e);
  assert(0 <= ng_e);
  assert(0 <= nh_e);
  assert(0 <= nh_e);

  acados_size_t size = sizeof(custom_memory);
  size += nbx * sizeof(int);
  /* blasfeo structs */
  size += (N + 1) * sizeof(struct blasfeo_dmat);
  /* blasfeo mem: mat */
  size += (N + 1) * blasfeo_memsize_dmat(nx, nx);        // uncertainty_matrix_buffer
  size += blasfeo_memsize_dmat(nw, nw);                  // W_mat
  size += 2 * blasfeo_memsize_dmat(nx, nw);              // unc_jac_G_mat, temp_GW_mat
  size += 4 * blasfeo_memsize_dmat(nx, nx);              // GWG_mat, A_mat, AK_mat, temp_AP_mat
  size += blasfeo_memsize_dmat(nx, nu);                  // B_mat
  size += 2 * blasfeo_memsize_dmat(nu, nx);              // K_mat, temp_KP_mat
  size += blasfeo_memsize_dmat(nu, nu);                  // temp_KPK_mat
  size += blasfeo_memsize_dmat(ng, nx);                  // Cg_mat
  size += blasfeo_memsize_dmat(ng, nu);                  // Dg_mat
  size += blasfeo_memsize_dmat(ng_e, nx);                // Cg_e_mat
  size += blasfeo_memsize_dmat(ngh_e_max, nu);           // dummy_Dgh_e_mat
  size += blasfeo_memsize_dmat(nh, nx);                  // Ch_mat
  size += blasfeo_memsize_dmat(nh, nu);                  // Dh_mat
  size += blasfeo_memsize_dmat(nh_e, nx);                // Ch_e_mat
  size += 2 * blasfeo_memsize_dmat(ngh_me_max, nx);      // temp_CaDK_mat, temp_CaDKmP_mat
  size += blasfeo_memsize_dmat(ngh_me_max, ngh_me_max);  // temp_beta_mat

  // NOTE(@naefjo): GP matrices stuffs:
  size += blasfeo_memsize_dmat(nw, nw);  // W_gp_mat
  // size += blasfeo_memsize_dmat(nx, nw);  // GW_gp_mat
  // size += blasfeo_memsize_dmat(nx, nx);  // GW_gpG_mat
  // size += blasfeo_memsize_dmat(nx, nx);  // GW_gpG_p_GWG_mat

  /* blasfeo mem: vec */
  /* Arrays */
  size += nx * nx * sizeof(double);                       // d_A_mat
  size += nx * nu * sizeof(double);                       // d_B_mat
  size += (ng + ng_e) * nx * sizeof(double);              // d_Cg_mat, d_Cg_e_mat
  size += (ng)*nu * sizeof(double);                       // d_Dg_mat
  size += (nh + nh_e + ng + ng_e) * nx * sizeof(double);  // d_Cgh_mat, d_Cgh_e_mat
  size += (nh + ng) * nu * sizeof(double);                // d_Dgh_mat
  // d_state_vec
  size += nx * sizeof(double);
  // constraints and tightened constraints
  size += 4 * (nbx + nbu + ng + nh) * sizeof(double);
  size += 4 * (nbx_e + ng_e + nh_e) * sizeof(double);
  size += (nbx + nbu + nbx_e) * sizeof(int);  // idxbx, idxbu, idxbx_e

  size += 1 * 8;  // initial alignment
  make_int_multiple_of(64, &size);
  size += 1 * 64;

  return size;
}

static custom_memory* custom_memory_assign(ocp_nlp_config* nlp_config, ocp_nlp_dims* nlp_dims, void* raw_memory)
{
  int N = nlp_dims->N;
  int nx = 9;
  int nu = 3;
  int nw = 9;

  int ng = 0;
  int nh = 1;
  int nbx = 9;
  int nbu = 3;

  int ng_e = 0;
  int nh_e = 0;
  int ngh_e_max = int_max(ng_e, nh_e);
  int ngh_me_max = int_max(ngh_e_max, int_max(ng, nh));
  int nbx_e = 0;

  char* c_ptr = (char*)raw_memory;
  custom_memory* mem = (custom_memory*)c_ptr;
  c_ptr += sizeof(custom_memory);

  align_char_to(8, &c_ptr);
  assign_and_advance_blasfeo_dmat_structs(N + 1, &mem->uncertainty_matrix_buffer, &c_ptr);

  align_char_to(64, &c_ptr);

  for (int ii = 0; ii <= N; ii++)
  {
    assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->uncertainty_matrix_buffer[ii], &c_ptr);
  }
  // Disturbance Dynamics
  assign_and_advance_blasfeo_dmat_mem(nw, nw, &mem->W_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nw, &mem->unc_jac_G_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nw, &mem->temp_GW_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->GWG_mat, &c_ptr);
  // GP Disturbance Dynamics
  assign_and_advance_blasfeo_dmat_mem(nw, nw, &mem->W_gp_mat, &c_ptr);
  // assign_and_advance_blasfeo_dmat_mem(nx, nw, &mem->temp_GW_gp_mat, &c_ptr);
  // assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->GW_gpG_mat, &c_ptr);
  // assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->GW_gpG_p_GWG_mat, &c_ptr);
  // System Dynamics
  assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->A_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nu, &mem->B_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ng, nx, &mem->Cg_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ng, nu, &mem->Dg_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ng_e, nx, &mem->Cg_e_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ngh_e_max, nu, &mem->dummy_Dgh_e_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nh, nx, &mem->Ch_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nh, nu, &mem->Dh_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nh_e, nx, &mem->Ch_e_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nu, nx, &mem->K_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->AK_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nx, nx, &mem->temp_AP_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nu, nx, &mem->temp_KP_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(nu, nu, &mem->temp_KPK_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ngh_me_max, nx, &mem->temp_CaDK_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ngh_me_max, nx, &mem->temp_CaDKmP_mat, &c_ptr);
  assign_and_advance_blasfeo_dmat_mem(ngh_me_max, ngh_me_max, &mem->temp_beta_mat, &c_ptr);

  assign_and_advance_double(nx * nx, &mem->d_A_mat, &c_ptr);
  assign_and_advance_double(nx * nu, &mem->d_B_mat, &c_ptr);
  assign_and_advance_double(ng * nx, &mem->d_Cg_mat, &c_ptr);
  assign_and_advance_double(ng * nu, &mem->d_Dg_mat, &c_ptr);
  assign_and_advance_double(ng_e * nx, &mem->d_Cg_e_mat, &c_ptr);
  assign_and_advance_double((ng + nh) * nx, &mem->d_Cgh_mat, &c_ptr);
  assign_and_advance_double((ng + nh) * nu, &mem->d_Dgh_mat, &c_ptr);
  assign_and_advance_double((ng_e + nh_e) * nx, &mem->d_Cgh_e_mat, &c_ptr);
  assign_and_advance_double(nx, &mem->d_state_vec, &c_ptr);
  assign_and_advance_double(nbx, &mem->d_lbx, &c_ptr);
  assign_and_advance_double(nbx, &mem->d_ubx, &c_ptr);
  assign_and_advance_double(nbx_e, &mem->d_lbx_e, &c_ptr);
  assign_and_advance_double(nbx_e, &mem->d_ubx_e, &c_ptr);
  assign_and_advance_double(nbx, &mem->d_lbx_tightened, &c_ptr);
  assign_and_advance_double(nbx, &mem->d_ubx_tightened, &c_ptr);
  assign_and_advance_double(nbx_e, &mem->d_lbx_e_tightened, &c_ptr);
  assign_and_advance_double(nbx_e, &mem->d_ubx_e_tightened, &c_ptr);
  assign_and_advance_double(nbu, &mem->d_lbu, &c_ptr);
  assign_and_advance_double(nbu, &mem->d_ubu, &c_ptr);
  assign_and_advance_double(nbu, &mem->d_lbu_tightened, &c_ptr);
  assign_and_advance_double(nbu, &mem->d_ubu_tightened, &c_ptr);
  assign_and_advance_double(ng, &mem->d_lg, &c_ptr);
  assign_and_advance_double(ng, &mem->d_ug, &c_ptr);
  assign_and_advance_double(ng_e, &mem->d_lg_e, &c_ptr);
  assign_and_advance_double(ng_e, &mem->d_ug_e, &c_ptr);
  assign_and_advance_double(ng, &mem->d_lg_tightened, &c_ptr);
  assign_and_advance_double(ng, &mem->d_ug_tightened, &c_ptr);
  assign_and_advance_double(ng_e, &mem->d_lg_e_tightened, &c_ptr);
  assign_and_advance_double(ng_e, &mem->d_ug_e_tightened, &c_ptr);
  assign_and_advance_double(nh, &mem->d_lh, &c_ptr);
  assign_and_advance_double(nh, &mem->d_uh, &c_ptr);
  assign_and_advance_double(nh_e, &mem->d_lh_e, &c_ptr);
  assign_and_advance_double(nh_e, &mem->d_uh_e, &c_ptr);
  assign_and_advance_double(nh, &mem->d_lh_tightened, &c_ptr);
  assign_and_advance_double(nh, &mem->d_uh_tightened, &c_ptr);
  assign_and_advance_double(nh_e, &mem->d_lh_e_tightened, &c_ptr);
  assign_and_advance_double(nh_e, &mem->d_uh_e_tightened, &c_ptr);

  assign_and_advance_int(nbx, &mem->idxbx, &c_ptr);
  assign_and_advance_int(nbu, &mem->idxbu, &c_ptr);
  assign_and_advance_int(nbx_e, &mem->idxbx_e, &c_ptr);

  assert((char*)raw_memory + custom_memory_calculate_size(nlp_config, nlp_dims) >= c_ptr);
  mem->raw_memory = raw_memory;

  return mem;
}

static void* custom_memory_create(pacejka_model_solver_capsule* capsule)
{
  // printf("\nin custom_memory_create_function\n");

  ocp_nlp_dims* nlp_dims = pacejka_model_acados_get_nlp_dims(capsule);
  ocp_nlp_config* nlp_config = pacejka_model_acados_get_nlp_config(capsule);
  acados_size_t bytes = custom_memory_calculate_size(nlp_config, nlp_dims);

  void* ptr = acados_calloc(1, bytes);

  custom_memory* custom_mem = custom_memory_assign(nlp_config, nlp_dims, ptr);
  custom_mem->raw_memory = ptr;

  return custom_mem;
}

static void custom_val_init_function(ocp_nlp_dims* nlp_dims, ocp_nlp_in* nlp_in, ocp_nlp_solver* nlp_solver,
                                     custom_memory* custom_mem)
{
  int N = nlp_dims->N;
  int nx = 9;
  int nu = 3;
  int nw = 9;

  int ng = 0;
  int nh = 1;
  int nbx = 9;
  int nbu = 3;

  int ng_e = 0;
  int nh_e = 0;
  int ngh_e_max = int_max(ng_e, nh_e);
  int nbx_e = 0;

  /* Get the state constraint bounds */
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "idxbx", custom_mem->idxbx);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "idxbx", custom_mem->idxbx_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lbx", custom_mem->d_lbx);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ubx", custom_mem->d_ubx);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lbx", custom_mem->d_lbx_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "ubx", custom_mem->d_ubx_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "idxbu", custom_mem->idxbu);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lbu", custom_mem->d_lbu);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ubu", custom_mem->d_ubu);
  // Get the Jacobians and the bounds of the linear constraints
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lg", custom_mem->d_lg);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ug", custom_mem->d_ug);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lg", custom_mem->d_lg_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "ug", custom_mem->d_ug_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "C", custom_mem->d_Cg_mat);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "D", custom_mem->d_Dg_mat);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "C", custom_mem->d_Cg_e_mat);
  blasfeo_pack_dmat(ng, nx, custom_mem->d_Cg_mat, ng, &custom_mem->Cg_mat, 0, 0);
  blasfeo_pack_dmat(ng, nu, custom_mem->d_Dg_mat, ng, &custom_mem->Dg_mat, 0, 0);
  blasfeo_pack_dmat(ng_e, nx, custom_mem->d_Cg_e_mat, ng_e, &custom_mem->Cg_e_mat, 0, 0);
  blasfeo_dgese(ngh_e_max, nu, 0., &custom_mem->dummy_Dgh_e_mat, 0, 0);  // fill with zeros
  // NOTE: fixed lower and upper bounds of nonlinear constraints
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lh", custom_mem->d_lh);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "uh", custom_mem->d_uh);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lh", custom_mem->d_lh_e);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "uh", custom_mem->d_uh_e);

  /* Initilize tightened constraints*/
  // NOTE: tightened constraints are only initialized once
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lbx", custom_mem->d_lbx_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ubx", custom_mem->d_ubx_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lbx", custom_mem->d_lbx_e_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "ubx", custom_mem->d_ubx_e_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lbu", custom_mem->d_lbu_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ubu", custom_mem->d_ubu_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lg", custom_mem->d_lg_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "ug", custom_mem->d_ug_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lg", custom_mem->d_lg_e_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "ug", custom_mem->d_ug_e_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "lh", custom_mem->d_lh_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, 1, "uh", custom_mem->d_uh_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "lh", custom_mem->d_lh_e_tightened);
  ocp_nlp_constraints_model_get(nlp_solver->config, nlp_dims, nlp_in, N, "uh", custom_mem->d_uh_e_tightened);

  /* Initialize the W matrix */
  // blasfeo_dgese(nw, nw, 0., &custom_mem->W_mat, 0, 0);
  blasfeo_dgein1(0.00018861, &custom_mem->W_mat, 0, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 0, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 0);
  blasfeo_dgein1(0.00010332, &custom_mem->W_mat, 1, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 1, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 1);
  blasfeo_dgein1(0.00220326, &custom_mem->W_mat, 2, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 2, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 2);
  blasfeo_dgein1(0.00010145, &custom_mem->W_mat, 3, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 3, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 3);
  blasfeo_dgein1(0.00010113, &custom_mem->W_mat, 4, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 4, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 4);
  blasfeo_dgein1(0.00273745, &custom_mem->W_mat, 5, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 5, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 5);
  blasfeo_dgein1(0.00010108, &custom_mem->W_mat, 6, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 6, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 6);
  blasfeo_dgein1(0.00010109, &custom_mem->W_mat, 7, 7);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 7, 8);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 0);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 1);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 2);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 3);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 4);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 5);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 6);
  blasfeo_dgein1(0.0, &custom_mem->W_mat, 8, 7);
  blasfeo_dgein1(0.00010109, &custom_mem->W_mat, 8, 8);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 0, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 0, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 0);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 1, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 1, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 1);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 2, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 2, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 2);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 3, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 3, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 3);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 4, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 4, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 4);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 5, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 5, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 5);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 6, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 6, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 6);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 7, 7);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 7, 8);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 0);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 1);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 2);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 3);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 4);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 5);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 6);
  blasfeo_dgein1(0.0, &custom_mem->unc_jac_G_mat, 8, 7);
  blasfeo_dgein1(1.0, &custom_mem->unc_jac_G_mat, 8, 8);

  // // NOTE: if G is changing this is not in init!
  // // temp_GW_mat = unc_jac_G_mat * W_mat
  // blasfeo_dgemm_nn(nx, nw, nw, 1.0, &custom_mem->unc_jac_G_mat, 0, 0, &custom_mem->W_mat, 0, 0, 0.0,
  //                  &custom_mem->temp_GW_mat, 0, 0, &custom_mem->temp_GW_mat, 0, 0);
  // // GWG_mat = temp_GW_mat * unc_jac_G_mat^T
  // blasfeo_dgemm_nt(nx, nx, nw, 1.0, &custom_mem->temp_GW_mat, 0, 0, &custom_mem->unc_jac_G_mat, 0, 0, 0.0,
  //                  &custom_mem->GWG_mat, 0, 0, &custom_mem->GWG_mat, 0, 0);

  // NOTE(@naefjo): Initialize Gaussian Process covariance matrix to 0.
  blasfeo_dgese(nw, nw, 0.0, &custom_mem->W_gp_mat, 0, 0);
  // blasfeo_dgese(nw, nw, 0.0, &GW_gpG_p_GWG_mat, 0, 0);

  /* Initialize the uncertainty_matrix_buffer[0] */
  blasfeo_dgein1(0.001, &custom_mem->uncertainty_matrix_buffer[0], 0, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 0, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 0);
  blasfeo_dgein1(0.001, &custom_mem->uncertainty_matrix_buffer[0], 1, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 1, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 1);
  blasfeo_dgein1(0.001, &custom_mem->uncertainty_matrix_buffer[0], 2, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 2, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 2);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 3, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 3, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 3);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 4, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 4, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 4);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 5, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 5, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 5);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 6, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 6, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 6);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 7, 7);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 7, 8);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 0);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 1);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 2);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 3);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 4);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 5);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 6);
  blasfeo_dgein1(0.0, &custom_mem->uncertainty_matrix_buffer[0], 8, 7);
  blasfeo_dgein1(1e-6, &custom_mem->uncertainty_matrix_buffer[0], 8, 8);

  /* Initialize the feedback gain matrix */
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 0);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 1);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 2);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 3);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 4);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 5);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 6);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 7);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 0, 8);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 0);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 1);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 2);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 3);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 4);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 5);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 6);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 7);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 1, 8);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 0);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 1);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 2);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 3);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 4);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 5);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 6);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 7);
  blasfeo_dgein1(0.0, &custom_mem->K_mat, 2, 8);
}

int custom_update_init_function(pacejka_model_solver_capsule* capsule)
{
  capsule->custom_update_memory = custom_memory_create(capsule);
  ocp_nlp_in* nlp_in = pacejka_model_acados_get_nlp_in(capsule);

  ocp_nlp_dims* nlp_dims = pacejka_model_acados_get_nlp_dims(capsule);
  ocp_nlp_solver* nlp_solver = pacejka_model_acados_get_nlp_solver(capsule);
  custom_val_init_function(nlp_dims, nlp_in, nlp_solver, capsule->custom_update_memory);
  return 1;
}

static void compute_gh_beta(struct blasfeo_dmat* K_mat, struct blasfeo_dmat* C_mat, struct blasfeo_dmat* D_mat,
                            struct blasfeo_dmat* CaDK_mat, struct blasfeo_dmat* CaDKmP_mat,
                            struct blasfeo_dmat* beta_mat, struct blasfeo_dmat* P_mat, int n_cstr, int nx, int nu)
{
  // (C+DK)@P@(C^T+K^TD^T)
  // CaDK_mat = C_mat + D_mat @ K_mat
  blasfeo_dgemm_nn(n_cstr, nx, nu, 1.0, D_mat, 0, 0, K_mat, 0, 0, 1.0, C_mat, 0, 0, CaDK_mat, 0, 0);
  // CaDKmP_mat = CaDK_mat @ P_mat
  blasfeo_dgemm_nn(n_cstr, nx, nx, 1.0, CaDK_mat, 0, 0, P_mat, 0, 0, 0.0, CaDKmP_mat, 0, 0, CaDKmP_mat, 0, 0);
  // NOTE: here we also compute cross-terms which are not needed.
  //       Only diag(beta_mat) is used later
  // beta_mat = CaDKmP_mat @ CaDK_mat^T
  // blasfeo_dgemm_nt(n_cstr, n_cstr, nx, 1.0, CaDKmP_mat, 0, 0,
  //                     CaDK_mat, 0, 0, 0.0,
  //                     beta_mat, 0, 0, beta_mat, 0, 0);
  for (int ii = 0; ii < n_cstr; ii++)
  {
    blasfeo_dgemm_nt(1, 1, nx, 1.0, CaDKmP_mat, ii, 0, CaDK_mat, ii, 0, 0.0, beta_mat, ii, ii, beta_mat, ii, ii);
  }
}

static void compute_KPK(struct blasfeo_dmat* K_mat, struct blasfeo_dmat* temp_KP_mat, struct blasfeo_dmat* temp_KPK_mat,
                        struct blasfeo_dmat* P_mat, int nx, int nu)
{
  // K @ P_k @ K^T
  // temp_KP_mat = K_mat @ P_mat
  blasfeo_dgemm_nn(nu, nx, nx, 1.0, K_mat, 0, 0, P_mat, 0, 0, 0.0, temp_KP_mat, 0, 0, temp_KP_mat, 0, 0);
  // temp_KPK_mat = temp_KP_mat @ K_mat^T
  blasfeo_dgemm_nt(nu, nu, nx, 1.0, temp_KP_mat, 0, 0, K_mat, 0, 0, 0.0, temp_KPK_mat, 0, 0, temp_KPK_mat, 0, 0);
}

static void compute_next_P_matrix(struct blasfeo_dmat* P_mat, struct blasfeo_dmat* P_next_mat,
                                  struct blasfeo_dmat* A_mat, struct blasfeo_dmat* B_mat, struct blasfeo_dmat* K_mat,
                                  struct blasfeo_dmat* W_mat, struct blasfeo_dmat* AK_mat,
                                  struct blasfeo_dmat* temp_AP_mat, int nx, int nu)
{
  // TODO: exploit symmetry of P, however, only blasfeo_dtrmm_rlnn is implemented in high-performance BLAFEO variant.
  // AK_mat = -B@K + A
  blasfeo_dgemm_nn(nx, nx, nu, -1.0, B_mat, 0, 0, K_mat, 0, 0, 1.0, A_mat, 0, 0, AK_mat, 0, 0);
  // temp_AP_mat = AK_mat @ P_k
  blasfeo_dgemm_nn(nx, nx, nx, 1.0, AK_mat, 0, 0, P_mat, 0, 0, 0.0, temp_AP_mat, 0, 0, temp_AP_mat, 0, 0);
  // P_{k+1} = temp_AP_mat @ AK_mat^T + GWG_mat
  blasfeo_dgemm_nt(nx, nx, nx, 1.0, temp_AP_mat, 0, 0, AK_mat, 0, 0, 1.0, W_mat, 0, 0, P_next_mat, 0, 0);
}

// static void reset_P0_matrix(ocp_nlp_dims* nlp_dims, struct blasfeo_dmat* P_mat, double* data)
// {
//   int nx = nlp_dims->nx[0];
//   blasfeo_pack_dmat(nx, nx, data, nx, P_mat, 0, 0);
// }

/**
 * @brief Computes the adjusted GWG based on the uncertainties provided by the Gaussian Process.
 */
static void compute_GWG(ocp_nlp_solver* solver, custom_memory* custom_mem, double* data, const int current_stage)
{
  ocp_nlp_dims* nlp_dims = solver->dims;

  // int N = nlp_dims->N;
  const int nx = nlp_dims->nx[0];
  const int nw = 9;

  // NOTE(@naefjo): update GP covariance terms for current stage
  for (int i = 0; i < nw; ++i)
  {
    blasfeo_dgein1(data[nw * current_stage + i], &custom_mem->W_gp_mat, i, i);
  }

  // NOTE(@naefjo): Combine covariances of process noise w and of GP.
  blasfeo_dgead(nw, nw, 1.0, &custom_mem->W_mat, 0, 0, &custom_mem->W_gp_mat, 0, 0);

  // // NOTE(@naefjo): Compute G@W_gp@G^T term coming from GP
  // temp_GW_mat = unc_jac_G_mat * W_gp_mat
  blasfeo_dgemm_nn(nx, nw, nw, 1.0, &custom_mem->unc_jac_G_mat, 0, 0, &custom_mem->W_gp_mat, 0, 0, 0.0,
                   &custom_mem->temp_GW_mat, 0, 0, &custom_mem->temp_GW_mat, 0, 0);
  // GWG_mat = temp_GW_gp_mat * unc_jac_G_mat^T
  blasfeo_dgemm_nt(nx, nx, nw, 1.0, &custom_mem->temp_GW_mat, 0, 0, &custom_mem->unc_jac_G_mat, 0, 0, 0.0,
                   &custom_mem->GWG_mat, 0, 0, &custom_mem->GWG_mat, 0, 0);
}

// NOTE(@naefjo): modified function signature
static void uncertainty_propagate_and_update(ocp_nlp_solver* solver, ocp_nlp_in* nlp_in, ocp_nlp_out* nlp_out,
                                             custom_memory* custom_mem, double* data, const int data_out_start_idx,
                                             const int data_len)
{
  ocp_nlp_config* nlp_config = solver->config;
  ocp_nlp_dims* nlp_dims = solver->dims;

  int N = nlp_dims->N;
  int nx = nlp_dims->nx[0];
  int nw = 9;
  int nu = nlp_dims->nu[0];
  int nx_sqr = nx * nx;
  int nbx = 9;
  int nbu = 3;
  int ng = 0;
  int nh = 1;
  int ng_e = 0;
  int nh_e = 0;
  int nbx_e = 0;
  double backoff_scaling_gamma = 0.6744897501960817;

  // NOTE(@naefjo): Save uncertainty data
  double cov_x = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[0], 0, 0);
  double cov_y = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[0], 1, 1);
  double cov_xy = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[0], 0, 1);

  data[data_out_start_idx + 0] = cov_x;
  data[data_out_start_idx + 1] = cov_y;
  data[data_out_start_idx + 2] = cov_xy;

  // First Stage
  // NOTE: lbx_0 and ubx_0 should not be tightened.
  // NOTE: lg_0 and ug_0 are not tightened.
  // NOTE: lh_0 and uh_0 are not tightened.
  // Middle Stages
  // constraint tightening: for next stage based on dynamics of ii stage
  // P[ii+1] = (A-B@K) @ P[ii] @ (A-B@K).T + G@W@G.T
  for (int ii = 0; ii < N - 1; ii++)
  {
    // get and pack: A, B
    ocp_nlp_get_at_stage(nlp_config, nlp_dims, solver, ii, "A", custom_mem->d_A_mat);
    blasfeo_pack_dmat(nx, nx, custom_mem->d_A_mat, nx, &custom_mem->A_mat, 0, 0);
    ocp_nlp_get_at_stage(nlp_config, nlp_dims, solver, ii, "B", custom_mem->d_B_mat);
    blasfeo_pack_dmat(nx, nu, custom_mem->d_B_mat, nx, &custom_mem->B_mat, 0, 0);

    compute_GWG(solver, custom_mem, data, ii);

    compute_next_P_matrix(&(custom_mem->uncertainty_matrix_buffer[ii]),
                          &(custom_mem->uncertainty_matrix_buffer[ii + 1]), &custom_mem->A_mat, &custom_mem->B_mat,
                          &custom_mem->K_mat, &custom_mem->GWG_mat, &custom_mem->AK_mat, &custom_mem->temp_AP_mat, nx,
                          nu);

    // state constraints
    // nonlinear constraints: h
    // Get C_{k+1} and D_{k+1}
    ocp_nlp_get_at_stage(solver->config, nlp_dims, solver, ii + 1, "C", custom_mem->d_Cgh_mat);
    ocp_nlp_get_at_stage(solver->config, nlp_dims, solver, ii + 1, "D", custom_mem->d_Dgh_mat);
    // NOTE: the d_Cgh_mat is column-major, the first ng rows are the Jacobians of the linear constraints
    blasfeo_pack_dmat(nh, nx, custom_mem->d_Cgh_mat + ng, ng + nh, &custom_mem->Ch_mat, 0, 0);
    blasfeo_pack_dmat(nh, nu, custom_mem->d_Dgh_mat + ng, ng + nh, &custom_mem->Dh_mat, 0, 0);

    compute_gh_beta(&custom_mem->K_mat, &custom_mem->Ch_mat, &custom_mem->Dh_mat, &custom_mem->temp_CaDK_mat,
                    &custom_mem->temp_CaDKmP_mat, &custom_mem->temp_beta_mat,
                    &custom_mem->uncertainty_matrix_buffer[ii + 1], nh, nx, nu);

    // printf("temp_CaDKmP_mat k = %d", ii);
    // blasfeo_print_dmat(nh, nx, &custom_mem->temp_CaDKmP_mat, 0, 0);

    // TODO: eval hessian(h) -> H_hess (nh*(nx+nu)**2)
    // temp_Kt_hhess = h_i_hess[:nx, :] + K^T * h_i_hess[nx:nx+nu, :]
    // tempCD = temp_CaDKmP_mat * temp_Kt_hhess
    // acados_CD += tempCD
    // += for upper or -= for lower bound

    // NOTE(@naefjo): Save uncertainty data
    cov_x = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[ii + 1], 0, 0);
    cov_y = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[ii + 1], 1, 1);
    cov_xy = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[ii + 1], 0, 1);
    data[data_out_start_idx + 3 * (ii + 1)] = cov_x;
    data[data_out_start_idx + 3 * (ii + 1) + 1] = cov_y;
    data[data_out_start_idx + 3 * (ii + 1) + 2] = cov_xy;

    // only works if 1 bound is trivially satisfied.
    custom_mem->d_uh_tightened[0] =
        custom_mem->d_uh[0] - backoff_scaling_gamma * sqrt(blasfeo_dgeex1(&custom_mem->temp_beta_mat, 0, 0));

    // NOTE(@naefjo): make sure tightened bound does not become an empty set
    printf("constr at stage %d, %f\n", ii + 1, custom_mem->d_uh_tightened[0]);
    if (custom_mem->d_uh_tightened[0] < 0.0)
    {
      custom_mem->d_uh_tightened[0] = 0.0;
    }

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ii + 1, "uh", custom_mem->d_uh_tightened);
  }

  // Last stage
  // get and pack: A, B
  ocp_nlp_get_at_stage(nlp_config, nlp_dims, solver, N - 1, "A", custom_mem->d_A_mat);
  blasfeo_pack_dmat(nx, nx, custom_mem->d_A_mat, nx, &custom_mem->A_mat, 0, 0);
  ocp_nlp_get_at_stage(nlp_config, nlp_dims, solver, N - 1, "B", custom_mem->d_B_mat);
  blasfeo_pack_dmat(nx, nu, custom_mem->d_B_mat, nx, &custom_mem->B_mat, 0, 0);

  compute_GWG(solver, custom_mem, data, N - 1);

  // AK_mat = -B*K + A
  compute_next_P_matrix(&(custom_mem->uncertainty_matrix_buffer[N - 1]), &(custom_mem->uncertainty_matrix_buffer[N]),
                        &custom_mem->A_mat, &custom_mem->B_mat, &custom_mem->K_mat, &custom_mem->GWG_mat,
                        &custom_mem->AK_mat, &custom_mem->temp_AP_mat, nx, nu);

  // NOTE(@naefjo): Save uncertainty data
  cov_x = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[N], 0, 0);
  cov_y = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[N], 1, 1);
  cov_xy = blasfeo_dgeex1(&custom_mem->uncertainty_matrix_buffer[N], 0, 1);
  data[data_out_start_idx + 3 * ((N - 1) + 1)] = cov_x;
  data[data_out_start_idx + 3 * ((N - 1) + 1) + 1] = cov_y;
  data[data_out_start_idx + 3 * ((N - 1) + 1) + 2] = cov_xy;

  // state constraints nlbx_e_t
}

int custom_update_function(pacejka_model_solver_capsule* capsule, double* data, int data_len)
{
  custom_memory* custom_mem = (custom_memory*)capsule->custom_update_memory;
  ocp_nlp_config* nlp_config = pacejka_model_acados_get_nlp_config(capsule);
  ocp_nlp_dims* nlp_dims = pacejka_model_acados_get_nlp_dims(capsule);
  ocp_nlp_in* nlp_in = pacejka_model_acados_get_nlp_in(capsule);
  ocp_nlp_out* nlp_out = pacejka_model_acados_get_nlp_out(capsule);
  ocp_nlp_solver* nlp_solver = pacejka_model_acados_get_nlp_solver(capsule);
  void* nlp_opts = pacejka_model_acados_get_nlp_opts(capsule);

  // Get the first index of the data array which is designated for output data
  const int nw = 9;
  const int N = nlp_dims->N;
  int data_out_start_idx = nw * N;
  // NOTE(@naefjo): value should be correct but loop does not add any significant computation time
  // so I'll leave it for now. Might prevent some bugs...
  for (int i = 0; i < data_len; ++i)
  {
    if (data[i] < 0.0)
    {
      data_out_start_idx = i;
      break;
    }
  }

  // NOTE(@naefjo): modified function signature
  uncertainty_propagate_and_update(nlp_solver, nlp_in, nlp_out, custom_mem, data, data_out_start_idx, data_len);

  return 1;
}

int custom_update_terminate_function(pacejka_model_solver_capsule* capsule)
{
  custom_memory* mem = capsule->custom_update_memory;

  free(mem->raw_memory);
  return 1;
}

// useful prints for debugging

/*
printf("A_mat:\n");
blasfeo_print_exp_dmat(nx, nx, &custom_mem->A_mat, 0, 0);
printf("B_mat:\n");
blasfeo_print_exp_dmat(nx, nu, &custom_mem->B_mat, 0, 0);
printf("K_mat:\n");
blasfeo_print_exp_dmat(nu, nx, &custom_mem->K_mat, 0, 0);
printf("AK_mat:\n");
blasfeo_print_exp_dmat(nx, nx, &custom_mem->AK_mat, 0, 0);
printf("temp_AP_mat:\n");
blasfeo_print_exp_dmat(nx, nx, &custom_mem->temp_AP_mat, 0, 0);
printf("W_mat:\n");
blasfeo_print_exp_dmat(nx, nx, &custom_mem->W_mat, 0, 0);
printf("P_k+1:\n");
blasfeo_print_exp_dmat(nx, nx, &(custom_mem->uncertainty_matrix_buffer[ii+1]), 0, 0);*/
