from run_single_experiment import *
import subprocess

N_arr = [int(i) for i in np.ceil(np.logspace(0, 3, 10))]
batch_dim = 1
hidden_layers_arr = [1, 10, 20]
solve_steps = 1000
# device_arr = ["cpu", "cpu", "cuda", "cpu", "cuda"]
# num_threads_arr = [1, 10, 1, 1, 1]
# num_threads_acados_openmp_arr = [0, 0, 0, 10, 10]

# first device_arr again
device_arr = ["cpu"]
num_threads_arr = [1]
num_threads_acados_openmp_arr = [0]

# just N=1000 for first device_arr
# N_arr = [N_arr[-1]]
# device_arr = ["cpu"]
# num_threads_arr = [1]
# num_threads_acados_openmp_arr = [0]
# TODO: missing data here: with 10,20 hidden layers

# all other device_arr
# device_arr = ["cpu", "cuda", "cpu", "cuda"]
# num_threads_arr = [10, 1, 1, 1]
# num_threads_acados_openmp_arr = [0, 0, 10, 10]


assert len(num_threads_arr) == len(device_arr) == len(num_threads_acados_openmp_arr)
device_threads_arr = list(
    zip(device_arr, num_threads_arr, num_threads_acados_openmp_arr)
)

save_data = True


print(
    f"Running experiments with\nN={N_arr}\nhidden_layers={hidden_layers_arr}\ndevices={device_arr}\nnum_threads_torch={num_threads_arr}\nnum_threads_acados={num_threads_acados_openmp_arr}"
)
print(
    f"Total number of experiments: {len(N_arr)*len(hidden_layers_arr)*len(device_threads_arr)}"
)

num_threads_acados_openmp_previous = -1
for device, num_threads, num_threads_acados_openmp in device_threads_arr:

    build_acados = False
    if not num_threads_acados_openmp == num_threads_acados_openmp_previous:
        build_acados = True

    num_threads_acados_openmp_previous = num_threads_acados_openmp

    for i, N in enumerate(N_arr):
        for hidden_layers in hidden_layers_arr:
            print(
                f"Calling subprocess with N={N}, hidden_layers={hidden_layers}, device={device}, num_threads={num_threads}, num_threads_acados_openmp={num_threads_acados_openmp}"
            )

            if build_acados:
                build_acados_arg = "--build_acados"
            else:
                build_acados_arg = "--no-build_acados"

            build_acados = False

            subprocess_call_list = [
                "python",
                "run_single_experiment.py",
                "--N",
                str(N),
                "--hidden_layers",
                str(hidden_layers),
                "--device",
                str(device),
                "--num_threads",
                str(num_threads),
                "--num_threads_acados_openmp",
                str(num_threads_acados_openmp),
                "--solve_steps",
                str(solve_steps),
                build_acados_arg,
            ]
            subprocess_call_str = " ".join(subprocess_call_list)

            print(f"Start subprocess: {subprocess_call_str}")
            subprocess.check_call(subprocess_call_list)
            print(f"Finished subprocess: {subprocess_call_str}")
