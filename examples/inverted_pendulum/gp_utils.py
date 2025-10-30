import sys, os
import numpy as np
import gpytorch
import torch
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List


@dataclass
class GPlotData:
    x_path: np.ndarray = (None,)
    mean_on_path: np.ndarray = (None,)
    stddev_on_path: np.ndarray = (None,)
    conf_lower_on_path: np.ndarray = (None,)
    conf_upper_on_path: np.ndarray = (None,)
    sample_on_path: np.ndarray = (None,)
    train_x: np.ndarray = (None,)
    train_y: np.ndarray = None


def gp_data_from_model_and_path(
    gp_model, likelihood, x_path, num_samples=0, use_likelihood=False
):
    # GP data
    if gp_model.train_inputs[0].device.type == "cuda":
        # train_x = gp_model.train_inputs[0].cpu().numpy()
        # train_y = gp_model.train_targets.cpu().numpy()
        x_path_tensor = torch.Tensor(x_path).cuda()
        to_numpy = lambda T: T.cpu().numpy()
    else:
        # train_x = gp_model.train_inputs[0].numpy()
        # train_y = gp_model.train_targets.numpy()
        x_path_tensor = torch.Tensor(x_path)
        to_numpy = lambda T: T.numpy()

    train_x = to_numpy(gp_model.train_inputs[0])
    train_y = to_numpy(gp_model.train_targets)

    # dimensions
    num_points = x_path.shape[0]
    nout = train_y.shape[1]

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if use_likelihood:
            predictions = likelihood(gp_model(x_path_tensor))  # predictions with noise
        else:
            predictions = gp_model(
                x_path_tensor
            )  # only model (we want to find true function)

        mean_on_path = predictions.mean
        stddev_on_path = predictions.stddev
        conf_lower_on_path, conf_upper_on_path = predictions.confidence_region()

        # reshape
        mean_on_path = to_numpy(mean_on_path).reshape((num_points, nout))
        stddev_on_path = to_numpy(stddev_on_path).reshape((num_points, nout))
        conf_lower_on_path = to_numpy(conf_lower_on_path).reshape((num_points, nout))
        conf_upper_on_path = to_numpy(conf_upper_on_path).reshape((num_points, nout))

    # draw function samples
    sample_on_path = np.zeros((num_points, nout, num_samples))
    for j in range(0, num_samples):
        sample_on_path[:, :, j] = to_numpy(predictions.sample()).reshape(
            (num_points, nout)
        )

    return GPlotData(
        x_path,
        mean_on_path,
        stddev_on_path,
        conf_lower_on_path,
        conf_upper_on_path,
        sample_on_path,
        train_x,
        train_y,
    )


def gp_derivative_data_from_model_and_path(gp_model, likelihood, x_path, num_samples=0):
    # GP data
    train_x = gp_model.train_inputs[0].numpy()
    train_y = gp_model.train_targets.numpy()

    # dimensions
    num_points = x_path.shape[0]
    nout = train_y.shape[1]

    x_path_tensor = torch.Tensor(x_path)
    # Make predictions
    with gpytorch.settings.fast_pred_var():
        # predictions = likelihood(gp_model(test_x))
        # predictions = gp_model(x_path_tensor) # only model (we want to find true function)
        # mean_on_path = predictions.mean.detach().numpy()
        # stddev_on_path = predictions.stddev.detach().numpy()
        # conf_lower_on_path, conf_upper_on_path = predictions.confidence_region().detach().numpy()

        # DERIVATIVE
        mean_dx = torch.autograd.functional.jacobian(
            lambda x: gp_model(x).mean.sum(dim=0), x_path_tensor
        )

        # project derivative along path
        x_path_diff = x_path[1:, :] - x_path[:-1, :]
        x_path_norm = x_path_diff / np.tile(
            np.sqrt(np.sum(x_path_diff**2, axis=1)).reshape((num_points - 1, 1)),
            (1, x_path.shape[1]),
        )
        mean_dx_on_path = np.array(
            [
                np.inner(mean_dx[:, i, :], x_path_norm[i, :])
                for i in range(num_points - 1)
            ]
        )

        # kernel derivative
        # k = gp_model.covar_module
        # kernel_dx_left_at_train = torch.autograd.functional.jacobian(
        #     lambda x: k(x,train_x).sum(dim=0).unsqueeze(0),
        #     x_path_tensor
        # )
        # kernel_dx_right_at_train = torch.autograd.functional.jacobian(
        #     lambda x: k(train_x,x).sum(dim=1).unsqueeze(1),
        #     x_path_tensor
        # )
        # kernel_dx_left_at_eval = torch.autograd.functional.jacobian(
        #     lambda x: k(x,train_x).sum(dim=0).unsqueeze(0),
        #     train_x
        # )
        # kernel_ddx_at_eval = torch.autograd.functional.jacobian(
        #     lambda x: kernel_dx_left_at_eval(x,train_x).sum(dim=0).unsqueeze(0),
        #     train_x
        # )

    # draw function samples
    # sample_on_path = np.zeros((num_points,nout,num_samples))
    # for j in range(0,num_samples):
    #     sample_on_path[:,:,j] = predictions.sample()

    return GPlotData(
        x_path[1:, :], mean_dx_on_path, None, None, None, None, train_x, train_y
    )


def plot_gp_data(
    gp_data_list: List[GPlotData],
    marker_size_lim=[5, 100],
    marker_size_reldiff_zero=1e-3,
    marker_style="x",
    plot_train_data=True,
    x_path_mode="shared",
):
    # color_list
    cmap = plt.cm.tab10  # define the colormap
    # extract all colors from the .jet map
    color_list = [cmap(i) for i in range(cmap.N)]

    # Initialize plots
    nout = gp_data_list[0].train_y.shape[1]
    fig, ax = plt.subplots(nout, 1, figsize=(8, 3 * nout))
    if nout == 1:
        ax = [ax]

    if x_path_mode == "sequential":
        x_plot_segments_step = 1.0 / len(gp_data_list)
        x_plot_segments = np.linspace(0, 1, len(gp_data_list) + 1)
        # x_plot_all = np.linspace(0, 1, num_points * len(gp_data_list))

    for j, gp_data in enumerate(gp_data_list):
        num_points = gp_data.x_path.shape[0]
        if x_path_mode == "sequential":
            # x_plot_step = x_plot_segments_step / num_points
            # x_plot = np.arange(x_plot_segments[j], x_plot_segments[j+1], x_plot_step)
            x_plot = np.linspace(x_plot_segments[j], x_plot_segments[j + 1], num_points)
        else:
            x_plot = np.linspace(0, 1, num_points)

        for i in range(0, nout):
            add_gp_plot(
                gp_data,
                ax[i],
                i,
                x_plot=x_plot,
                color=color_list[j],
                marker_style=marker_style,
                marker_size_lim=marker_size_lim,
                marker_size_reldiff_zero=marker_size_reldiff_zero,
                plot_train_data=plot_train_data,
            )

    # general settings
    fig.set_facecolor("white")

    return fig, ax


def add_gp_plot(
    gp_data: GPlotData,
    ax,
    idx_out,
    x_plot=None,
    color="b",
    marker_size_lim=[5, 100],
    marker_size_reldiff_zero=1e-3,
    marker_style="x",
    plot_train_data=True,
):
    x_path = gp_data.x_path

    if plot_train_data:
        train_x = gp_data.train_x
        train_y_plot = gp_data.train_y[:, idx_out]

        # project on path
        train_x_on_path, train_x_dist_to_path, train_x_on_path_index = (
            project_data_on_path(train_x, x_path)
        )

        # square distance again for marker scaling (also quadratic)
        train_x_dist_to_path = train_x_dist_to_path**2

        # rescale to marker size limits
        train_x_dist_scale_to_zero = (
            train_x_dist_to_path / np.sum(train_x_on_path**2, axis=1)
            >= marker_size_reldiff_zero
        ) * 1.0
        marker_size_reldiff = (train_x_dist_to_path - min(train_x_dist_to_path)) / (
            max(train_x_dist_to_path) - min(train_x_dist_to_path)
        )
        train_x_dist_scale = marker_size_lim[1] + (
            marker_size_lim[0] - marker_size_lim[1]
        ) * (marker_size_reldiff * train_x_dist_scale_to_zero)

        num_points = x_path.shape[0]
        if x_plot is None:
            x_plot = np.linspace(0, 1, num_points)

        train_x_on_plot = x_plot[train_x_on_path_index]
        ax.scatter(
            train_x_on_plot,
            train_y_plot,
            s=train_x_dist_scale,
            marker=marker_style,
            color=color,
            alpha=0.5,
        )

    # Predictive mean as blue line
    mean = gp_data.mean_on_path[:, idx_out]
    ax.plot(x_plot, mean, color=color)

    # Shade in confidence
    # stddev = gp_data.stddev_on_path[:, idx_out]
    if (gp_data.conf_upper_on_path is not None) and (
        gp_data.conf_lower_on_path is not None
    ):
        upper = gp_data.conf_upper_on_path[:, idx_out]
        lower = gp_data.conf_lower_on_path[:, idx_out]
        y_plot_fill = ax.fill_between(x_plot, lower, upper, alpha=0.3, color=color)

    # plot samples
    if gp_data.sample_on_path is not None:
        num_samples = gp_data.sample_on_path.shape[2]
        for j in range(0, num_samples):
            sample = gp_data.sample_on_path[:, idx_out, j]
            ax.plot(x_plot, sample, color=color, linestyle=":", lw=1)

    # if gp_out_lims is not None:
    #     ax.set_ylim(gp_out_lims[i,:])

    ax.legend(["Observed Data", "Mean", "Confidence"])
    ax.set_title(f"Observed Values (Likelihood), output {idx_out}")
    return ax


def plot_gp_model(
    gp_model,
    likelihood,
    x_path,
    num_samples=0,
    marker_style="x",
    marker_size_lim=[1, 5],
):
    # gp_model, likelihood and x_path can be lists
    # Initialize plots
    nout = gp_model.train_targets[0].shape[0]
    fig, ax = plt.subplots(1, nout, figsize=(8, 3))
    if nout == 1:
        ax = [ax]

    # along which dim to plot (in gp_dim_lims[dim], other values fixed to gp_dim_slice[other_dims])
    # nout = gp_model.train_targets[0].shape[0]
    num_points = x_path.shape[0]

    # Set into eval mode
    gp_model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # predictions = likelihood(gp_model(test_x))
        predictions = gp_model(
            torch.Tensor(x_path)
        )  # only model (we want to find true function)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        # draw function samples
        predictions_samples = np.zeros((num_points, nout, num_samples))
        for j in range(0, num_samples):
            predictions_samples[:, :, j] = predictions.sample()

    # loop through all outputs and plot
    for i in range(0, nout):
        # TODO: Why do I need train_inputs[0]? Some batching possibilities?
        # plot data projected on path
        train_x = gp_model.train_inputs[0].numpy()
        train_y_plot = gp_model.train_targets[:, i].numpy()
        train_x_on_path, train_x_dist_to_path, train_x_on_path_index = (
            project_data_on_path(train_x, x_path)
        )

        # square distance again for marker scaling (also quadratic)
        train_x_dist_to_path = train_x_dist_to_path**2

        # rescale to marker size limits
        train_x_dist_scale = marker_size_lim[0] + (
            marker_size_lim[1] - marker_size_lim[0]
        ) * (train_x_dist_to_path - min(train_x_dist_to_path)) / (
            max(train_x_dist_to_path) - min(train_x_dist_to_path)
        )
        test_x_plot = np.linspace(0, 1, num_points)
        train_x_on_plot = test_x_plot[train_x_on_path_index]
        ax[i].scatter(
            train_x_on_plot,
            train_y_plot,
            s=train_x_dist_scale,
            marker=marker_style,
            color="k",
            alpha=0.5,
        )

        # Predictive mean as blue line
        ax[i].plot(test_x_plot, mean[:, i].numpy(), "b")

        # Shade in confidence
        y_plot_fill = ax[i].fill_between(
            test_x_plot, lower[:, i].numpy(), upper[:, i].numpy(), alpha=0.5
        )

        # plot samples
        for j in range(0, num_samples):
            ax[i].plot(test_x_plot, predictions_samples[:, i, j], "b:", lw=1)

        # if gp_out_lims is not None:
        #     ax[i].set_ylim(gp_out_lims[i,:])

        ax[i].legend(["Observed Data", "Mean", "Confidence"])
        ax[i].set_title(f"Observed Values (Likelihood), output {i}")

        # gp_data = gp_data_from_model_and_path(gp_model, likelihood, x_path,
        #     num_samples = num_samples
        # )

    # general settings
    fig.set_facecolor("white")


def project_data_on_path(x_data: np.array, x_path: np.array):
    # x_path: n_plot x dim
    # x_data: n_data x dim
    n_data = x_data.shape[0]
    n_path = x_path.shape[0]
    i_path, i_data = np.meshgrid(np.arange(0, n_path), np.arange(0, n_data))
    dist = np.sqrt(np.sum((x_data[i_data] - x_path[i_path]) ** 2, axis=2))
    dist_min_data_index = np.argmin(dist, axis=1)
    dist_min_data = dist[np.arange(0, n_data), dist_min_data_index]
    x_data_on_path = x_path[dist_min_data_index, :]
    return x_data_on_path, dist_min_data, dist_min_data_index


def generate_grid_points(x_dim_lims, x_dim_slice, x_dim_plot, num_points=200):
    x_dim_fix = np.arange(len(x_dim_slice))
    for i in x_dim_fix:
        if i == x_dim_plot:
            x_add = np.linspace(x_dim_lims[i, 0], x_dim_lims[i, 1], num_points)
        else:
            x_add = x_dim_slice[i] * np.ones((num_points,))

        # vstack
        if i == 0:
            x_grid = x_add
        else:
            x_grid = np.vstack((x_grid, x_add))
    return x_grid.transpose()
