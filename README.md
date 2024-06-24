# Zero-Order GP-MPC

An efficient implementation of a tailored SQP method for learning-based model predictive control with ellipsoidal uncertainties,

- using the `acados` Python interface to solve the optimal control problems, and
- employing `PyTorch` to evaluate Gaussian process dynamics models with support for GPU acceleration,
- efficiently handling the optimization over the posterior covariances of the GP inside the optimizer,

by modifying the Jacobians inside the SQP loop directly.

## Background

This software is built upon the results of the article "Zero-Order Optimization for Gaussian Process-based Model Predictive Control", published in the ECC 2023 Special Issue of the European Journal of Control (EJC), available at https://www.sciencedirect.com/science/article/pii/S0947358023000912. The code for the numerical experiments in the publication can be found [here](https://gitlab.ethz.ch/ics/zero-order-gp-mpc).

## Installation instructions

1. `clone` this repository and initialize submodules with
    ```bash
        git submodule update --recursive --init
    ```

2. Build the submodule `acados` according to the [installation instructions](https://docs.acados.org/installation/index.html):
    ```bash
        mkdir -p external/acados/build
        cd external/acados/build
        cmake -DACADOS_PYTHON=ON .. # do not forget the ".."
        make install -j4
    ```

3. Set environment variables (where `$PROJECT_DIR` is the directory you cloned the repository into in Step 1):
    ```bash
        export ACADOS_SOURCE_DIR=$PROJECT_DIR/external/acados
        export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
    ```

4. Create virtual environment and install Python dependencies (Python version 3.9.13):
    ```bash
        pip install -r requirements.txt
    ```

5. Test your installation by executing the example script
    ```bash
        cd examples/inverted_pendulum
        python inverted_pendulum_residual_learning_zoro.py
    ```

## Examples

You can find an example notebook of the zero-order GP-MPC method at `examples/inverted_pendulum/inverted_pendulum_zoro_acados.ipynb`.

## Citing us

If you use this software, please cite our corresponding article as written below.

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
