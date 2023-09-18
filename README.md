# Zero-Order GP-MPC

> WORK-IN-PROGRESS: The repository is currently being set up and might not be functional!

A high-performance implementation of a tailored SQP method for learning-based model predictive control with ellipsoidal uncertainties,
- using the `acados` Python interface to solve the optimal control problems, and
- employing `PyTorch` to evaluate learning-based dynamics models (neural networks, Gaussian processes, ...) with support for GPU acceleration.

## Background

This software is built upon the results of the article "Zero-Order Optimization for Gaussian Process-based Model Predictive Control", published in the ECC 2023 Special Issue of the European Journal of Control (EJC), available at https://www.sciencedirect.com/science/article/pii/S0947358023000912. The code for the numerical experiments in the publication can be found [here](https://gitlab.ethz.ch/ics/zero-order-gp-mpc).

## Installation instructions

### 

1. Clone this repository.
    ```bash
        git clone https://github.com/lahramon/zero-order-gpmpc.git
    ```

2. Initialize submodules.
    ```bash
        git submodule update --recursive --init
    ```

3. Build the submodule `acados` according to the [installation instructions](https://docs.acados.org/installation/index.html).
    ```bash
        mkdir -p acados/build
        cd acados/build
        cmake -DACADOS_PYTHON=ON .. # do not forget the ".."
        make install -j4
    ```

4. Set up Python environment (Python version 3.9.13). In the following we use `pipenv`.    
    1. Change into main directory, then run
        ```bash
            pipenv install
        ```
    2. Export the variables `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` point towards the right locations in your `acados` installation:
        ```bash
            export ACADOS_SOURCE_DIR=<yourpath>/zero-order-gpmpc/acados
            export LD_LIBRARY_PATH=<yourpath>/zero-order-gpmpc/acados/lib
        ```
        With `pipenv`, this can be done by defining a file called `.env` in the root directory, e.g., `<yourpath>/zero-order-gpmpc/.env`, which contains
        ```bash
            ACADOS_SOURCE_DIR=<yourpath>/zero-order-gpmpc/acados
            LD_LIBRARY_PATH=<yourpath>/zero-order-gpmpc/acados/lib
        ```
3. Run example:
    ```bash
        cd exp/hanging_chain/
        python main.py
    ```
    > At the first execution, you might be asked by `acados` to install `tera_renderer`.

## Citing us

If you find this software useful, please consider citing our corresponding article as written below.

```
@article{lahr_zero-order_2023,
  title = {Zero-Order optimization for {{Gaussian}} process-based model predictive control},
  author = {Lahr, Amon and Zanelli, Andrea and Carron, Andrea and Zeilinger, Melanie N.},
  year = {2023},
  journal = {European Journal of Control},
  pages = {100862},
  issn = {0947-3580},
  doi = {10.1016/j.ejcon.2023.100862}
}
```