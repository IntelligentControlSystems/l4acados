import numpy as np
from numpy import linalg as npla
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class EllipsoidTubeData2D:
    center_data: np.ndarray = (None,)
    ellipsoid_data: np.ndarray = (None,)
    ellipsoid_colors: np.ndarray = None


def base_plot(lb_theta=None):
    fig, ax = plt.subplots()

    if lb_theta is not None:
        ax.axvline(lb_theta)

    return fig, ax


def add_plot_ellipse(ax, E, e0, n=50):
    # sample angle uniformly from [0,2pi] and length from [0,1]
    radius = 1.0
    theta_arr = np.linspace(0, 2 * np.pi, n)
    w_rad_arr = [[radius, theta] for theta in theta_arr]
    w_one_arr = np.array(
        [
            [w_rad[0] * np.cos(w_rad[1]), w_rad[0] * np.sin(w_rad[1])]
            for w_rad in w_rad_arr
        ]
    )
    w_ell = np.array([e0 + E @ w_one for w_one in w_one_arr])
    h = ax.plot(w_ell[:, 0], w_ell[:, 1], linewidth=1)
    return h


def add_plot_trajectory(
    ax, tube_data: EllipsoidTubeData2D, color_fun=plt.cm.Blues, prob_tighten=1
):
    n_data = tube_data.center_data.shape[0]
    evenly_spaced_interval = np.linspace(0.6, 1, n_data)
    colors = [color_fun(x) for x in evenly_spaced_interval]

    h_plot = ax.plot(tube_data.center_data[:, 0], tube_data.center_data[:, 1])
    # set color
    h_plot[0].set_color(colors[-1])

    for i, color in enumerate(colors):
        center_i = tube_data.center_data[i, :]
        if tube_data.ellipsoid_data is not None:
            ellipsoid_i = tube_data.ellipsoid_data[i, :, :]
            # get eigenvalues
            eig_val, eig_vec = npla.eig(ellipsoid_i)
            ellipsoid_i_sqrt = (
                prob_tighten
                * eig_vec
                @ np.diag(np.sqrt(np.maximum(eig_val, 0)))
                @ np.transpose(eig_vec)
            )
            # print(i, eig_val, ellipsoid_i)
            h_ell = add_plot_ellipse(ax, ellipsoid_i_sqrt, center_i)
            h_ell[0].set_color(color)
