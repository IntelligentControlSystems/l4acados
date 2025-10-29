from run_single_experiment import *
import subprocess

debug = False

if debug:
    N_arr = [1, 10, 100]
    hidden_layers_arr = [1, 16, 1]
    hidden_size_arr = [256, 256, 1024]
    solve_steps = 100
    device_arr = ["cpu", "cpu", "cuda", "cpu", "cuda"]
    num_threads_arr = [1, 10, 1, 1, 1]
    num_threads_acados_openmp_arr = [0, 0, 0, 10, 10]
    # device_arr = ["cuda"]
    # num_threads_arr = [1]
    # num_threads_acados_openmp_arr = [10]
    save_data = True
else:
    # N_arr = [int(i) for i in np.ceil(np.logspace(0, 3, 10))]
    N_arr = [int(i) for i in np.ceil(np.logspace(0, 2, 5))]
    hidden_layers_arr = [1, 16, 1]
    hidden_size_arr = [256, 256, 1024]
    solve_steps = 1000
    device_arr = ["cpu", "cpu", "cuda", "cpu", "cuda"]
    num_threads_arr = [1, 10, 1, 1, 1]
    num_threads_acados_openmp_arr = [0, 0, 0, 10, 10]
    save_data = True

assert len(num_threads_arr) == len(device_arr)
assert len(num_threads_arr) == len(num_threads_acados_openmp_arr)

device_threads_arr = list(
    zip(device_arr, num_threads_arr, num_threads_acados_openmp_arr)
)
nn_size_arr = list(zip(hidden_layers_arr, hidden_size_arr))

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
        for hidden_layers, hidden_size in nn_size_arr:
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
                "--hidden_size",
                str(hidden_size),
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
