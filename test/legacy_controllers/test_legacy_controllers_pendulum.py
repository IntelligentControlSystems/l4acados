import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../examples/inverted_pendulum")
    ),
)

from run_example import solve_pendulum
from utils import base_plot, EllipsoidTubeData2D, add_plot_trajectory
import matplotlib.pyplot as plt


def test_legacy_controllers_pendulum():
    X = {}
    U = {}
    P = {}

    k_v1 = "zoro_acados"
    k_cupdate = "zoro_acados_custom_update"
    k_package = "zero_order_gpmpc"
    k_list = [k_v1, k_cupdate, k_package]

    X[k_v1], U[k_v1], P[k_v1] = solve_pendulum(k_v1)
    X[k_cupdate], U[k_cupdate], P[k_cupdate] = solve_pendulum(k_cupdate)
    X[k_package], U[k_package], P[k_package] = solve_pendulum(k_package)

    # ## Plot results
    # lb_theta = 0.0
    # fig, ax = base_plot(lb_theta=lb_theta)

    # plot_data_zoro_acados = EllipsoidTubeData2D(center_data=X[k_v1], ellipsoid_data=P[k_v1])
    # plot_data_zoro_cupdate = EllipsoidTubeData2D(
    #     center_data=X[k_cupdate], ellipsoid_data=P[k_cupdate]
    # )
    # plot_data_zero_order_gpmpc = EllipsoidTubeData2D(
    #     center_data=X[k_package], ellipsoid_data=P[k_package]
    # )
    # add_plot_trajectory(ax, plot_data_zoro_acados, color_fun=plt.cm.Purples, linewidth=5)
    # add_plot_trajectory(ax, plot_data_zoro_cupdate, color_fun=plt.cm.Oranges, linewidth=3)
    # add_plot_trajectory(ax, plot_data_zero_order_gpmpc, color_fun=plt.cm.Blues, linewidth=1)

    # plt.title("All controllers should give the same result")

    # plt.show()

    all_comparisons = [
        (i, j) for i in range(len(k_list)) for j in range(i + 1, len(k_list))
    ]
    all_comparisons_atol = [1e-2, 1e-2, 1e-8]

    for dic in [X, U, P]:
        for ij_index, ij_tup in enumerate(all_comparisons):
            i, j = ij_tup
            key = k_list[i]
            key_comp = k_list[j]
            # print(f"Comparing {key} and {key_comp}")
            print(
                f"Error in {key} and {key_comp}: {np.max(abs(dic[key]-dic[key_comp]))}"
            )
            assert np.allclose(
                dic[key], dic[key_comp], atol=all_comparisons_atol[ij_index]
            )


if __name__ == "__main__":
    test_legacy_controllers_pendulum()
