# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: zero-order-gpmpc-package-3.10
#     language: python
#     name: python3
# ---

# +
import acados
import importlib

importlib.reload(acados)
# -

from acados import run, MultiLayerPerceptron, DoubleIntegratorWithLearnedDynamics, MPC
import l4casadi as l4c
import numpy as np
import time
import l4acados as l4a
from typing import Optional, Union
import torch
import casadi as cs

from l4acados.controllers.residual_learning_mpc import ResidualLearningMPC
from l4acados.models import ResidualModel, PyTorchFeatureSelector
from l4acados.controllers.zoro_acados_utils import setup_sim_from_ocp

import copy

from run_single_experiment import *

N = 20
ts = 1.0 / N
batch_dim = 1
hidden_layers = 5
warmup_iter = 100
solve_steps = 1000
num_threads = 1
# device = "cpu"
device = "cuda"
num_threads_acados_openmp = 4

x_l4casadi, opt_times_l4casadi, x_l4acados, opt_times_l4acados = run(
    N,
    hidden_layers,
    solve_steps,
    device=device,
    num_threads_acados_openmp=num_threads_acados_openmp,
)

import matplotlib.pyplot as plt

opt_times_l4casadi_avg = np.cumsum(opt_times_l4casadi[warmup_iter:]) / np.arange(
    1, len(opt_times_l4casadi[warmup_iter:]) + 1
)
opt_times_l4acados_avg = np.cumsum(opt_times_l4acados[warmup_iter:]) / np.arange(
    1, len(opt_times_l4acados[warmup_iter:]) + 1
)

h_l4casadi = plt.plot(
    np.arange(warmup_iter, len(opt_times_l4casadi)),
    opt_times_l4casadi_avg,
    label="l4casadi",
)
h_l4acados = plt.plot(
    np.arange(warmup_iter, len(opt_times_l4casadi)),
    opt_times_l4acados_avg,
    label="l4acados",
)
plt.plot(
    opt_times_l4casadi,
    label="l4casadi",
    color=h_l4casadi[0].get_color(),
    alpha=0.3,
)
plt.plot(
    opt_times_l4acados,
    label="l4acados",
    color=h_l4acados[0].get_color(),
    alpha=0.3,
)
plt.axvline(
    x=warmup_iter, color="k", linestyle="--", linewidth=1, label="warmup", alpha=0.5
)
# axes scaling log
# plt.xscale("log")
plt.yscale("log")
# axes title y
plt.ylabel("Time [s]")
plt.xlabel("Iteration")
plt.legend()
# plt.ylim([1e-3, 1e-1])
plt.grid()

opt_times_l4casadi_avg[-1], opt_times_l4acados_avg[-1], opt_times_l4casadi_avg[
    -1
] / opt_times_l4acados_avg[-1]

plt.plot(x_l4casadi, linewidth=3)
plt.plot(x_l4acados, linewidth=1)

np.linalg.norm(np.array(x_l4casadi) - np.array(x_l4acados))

plt.plot(np.array(x_l4acados) - np.array(x_l4casadi))
