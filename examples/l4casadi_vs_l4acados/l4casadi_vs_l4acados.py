# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: l4acados_dev
#     language: python
#     name: python3
# ---

# %%
import acados
import importlib

importlib.reload(acados)

# %%
from acados import (
    run,
    NaiveMultiLayerPerceptron,
    MultiLayerPerceptron,
    DoubleIntegratorWithLearnedDynamics,
    MPC,
)
import l4casadi as l4c
import numpy as np
import time
import l4acados as l4a
from typing import Optional, Union
import torch
import casadi as cs

# %%
from l4acados.controllers.residual_learning_mpc import ResidualLearningMPC
from l4acados.models import ResidualModel, PyTorchFeatureSelector
from l4acados.controllers.zoro_acados_utils import setup_sim_from_ocp

# %%
import copy

# %%
from run_single_experiment import *

# %%
N = 1
batch_dim = 1
# hidden_layers = 2
# hidden_size = 2048
# hidden_layers = 64
# hidden_size = 256
hidden_layers = 1
hidden_size = 256
# hidden_layers = 1
# hidden_size = 1024
# hidden_layers = 16
# hidden_size = 256
warmup_iter = 10
solve_steps = 10000
num_threads = 1
device = "cpu"
# device = "cuda"
num_threads_acados_openmp = 0
run_methods = ["l4casadi", "l4casadi_naive", "l4acados"]
# run_methods = ["l4acados"]

# %%
hidden_layers

# %%
n_inputs = 2
n_outputs = 1

# check number of parameters in NN
mlp = MultiLayerPerceptron(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    hidden_layers=hidden_layers,
    hidden_size=hidden_size,
)
num_params = sum(p.numel() for p in mlp.parameters())

mlp_naive = NaiveMultiLayerPerceptron(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    hidden_layers=hidden_layers,
    hidden_size=hidden_size,
)
num_params_naive = sum(p.numel() for p in mlp_naive.parameters())

count_params = lambda n_inputs, n_outputs, hidden_layers, hidden_size: (
    (n_inputs + n_outputs + hidden_layers + 1) * hidden_size
    + (hidden_layers) * (hidden_size**2)
    + 1
)  # + 1 for bias in output layer

num_params_fun = count_params(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    hidden_layers=hidden_layers,
    hidden_size=hidden_size,
)

num_params, num_params_naive, num_params_fun

# %%
hidden_layers_2 = 4**2
hidden_size_2 = np.sqrt(hidden_size**2 / hidden_layers_2)

params_2 = count_params(
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    hidden_layers=hidden_layers_2,
    hidden_size=hidden_size_2,
)

params_2, hidden_size_2, hidden_layers_2

# %%
# hidden_layers = int(hidden_layers_2)
# hidden_size = int(hidden_size_2)

results_dict = run(
    N,
    hidden_layers,
    hidden_size,
    solve_steps,
    device=device,
    num_threads=num_threads,
    num_threads_acados_openmp=num_threads_acados_openmp,
    save_data=True,
    run_methods=run_methods,
)

# %%
results_dict.keys()

# %%
# load data
save_file = os.path.join(
    "data",
    f"l4casadi_vs_l4acados_N{N}_layers{hidden_layers}_size{hidden_size}_steps{solve_steps}_{device}_threads{num_threads}_acados{num_threads_acados_openmp}.npz",
)
results_dict = dict(np.load(save_file))

# %%
results_dict.keys()

# %%
import matplotlib.pyplot as plt

# %%
keys_time = [key for key in results_dict.keys() if "time" in key]
for key_values in keys_time:
    results_dict[f"{key_values}_avg"] = np.cumsum(
        results_dict[key_values][warmup_iter:]
    ) / np.arange(1, len(results_dict[key_values][warmup_iter:]) + 1)

# %%
results_dict.keys()

# %%
results_dict["l4acados_time_preparation_avg"]

# %%
keys_methods = ["l4casadi", "l4casadi_naive", "l4acados"]
# key_plot = "time_total"
key_plot = "time_preparation"
# key_plot = "time_feedback"
for key in keys_methods:
    key_avg = f"{key}_{key_plot}_avg"
    key_avg_extra = f"{key}_{key_plot}_extra_avg"
    if key_avg in results_dict.keys():
        h_plot = plt.plot(
            np.arange(warmup_iter, len(results_dict[key_avg]) + warmup_iter),
            results_dict[key_avg],
            label=key,
        )
    if key_avg_extra in results_dict.keys():
        h_plot = plt.plot(
            np.arange(warmup_iter, len(results_dict[key_avg]) + warmup_iter),
            results_dict[key_avg],
            label=key_avg_extra,
            color=h_plot[0].get_color(),
            linestyle="--",
            linewidth=3.0,
        )
    key_time = f"{key}_{key_plot}"
    if key_time in results_dict.keys():
        plt.plot(
            results_dict[key_time],
            color=h_plot[0].get_color(),
            alpha=0.3,
        )

# plot l4acados to_tensor_time
if "l4acados_time_to_tensor_avg" in results_dict.keys():
    plt.plot(
        np.arange(
            warmup_iter, len(results_dict["l4acados_time_to_tensor_avg"]) + warmup_iter
        ),
        results_dict["l4acados_time_to_tensor_avg"],
        label="l4acados to_tensor",
        linestyle="--",
    )

if "l4acados_time_residual_model_avg" in results_dict.keys():
    plt.plot(
        np.arange(
            warmup_iter,
            len(results_dict["l4acados_time_residual_model_avg"]) + warmup_iter,
        ),
        results_dict["l4acados_time_residual_model_avg"],
        label="l4acados residual model",
        linestyle="--",
    )

if "l4acados_time_nominal_model_avg" in results_dict.keys():
    plt.plot(
        np.arange(
            warmup_iter,
            len(results_dict["l4acados_time_nominal_model_avg"]) + warmup_iter,
        ),
        results_dict["l4acados_time_nominal_model_avg"],
        label="l4acados nominal model",
        linestyle="--",
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

# %%
results_dict.keys()

# %%
fig, ax = plt.subplots(1, 2)

key_plot = "state_trajectory"
for key in keys_methods:
    key_state = f"{key}_{key_plot}"
    if key_state in results_dict.keys():
        for i in range(2):
            ax[i].plot(
                results_dict[key_state][:, i],
                label=key,
            )

plt.legend()

# %%
diff_matrix = np.nan * np.ones((len(keys_methods), len(keys_methods)))
for i, key in enumerate(keys_methods):
    for j, key_2 in enumerate(keys_methods[i + 1 :]):
        j_2 = j + i + 1
        key_state_1 = f"{key}_{key_plot}"
        key_state_2 = f"{key_2}_{key_plot}"
        if key_state_1 in results_dict.keys() and key_state_2 in results_dict.keys():
            diff_matrix[i, j_2] = np.linalg.norm(
                np.array(results_dict[key_state_1])
                - np.array(results_dict[key_state_2]),
                ord=np.inf,
            )

diff_matrix, keys_methods

# %%
