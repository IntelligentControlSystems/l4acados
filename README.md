# L4acados

Integrate learning-based Python models into acados for real-time model predictive control.

## Installation

### Install `acados`

1. Clone
    ```bash
    git submodule update --recursive --init external/acados
    ```
1. Build the submodule `acados` according to the [installation instructions](https://docs.acados.org/installation/index.html):
    ```bash
    mkdir -p external/acados/build
    cd external/acados/build
    cmake -DACADOS_PYTHON=ON .. # do not forget the ".."
    make install -j4
    ```

2. Install acados Python interface
    ```bash
    pip install -e external/acados/interfaces/acados_template
    ```

3. Set environment variables (where `$PROJECT_DIR` is the directory you cloned the repository into in Step 1):
    ```bash
    export ACADOS_SOURCE_DIR=$PROJECT_DIR/external/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
    ```

### Install `L4acados` with optional dependencies of your choice

Install `L4acados` with optional dependencies of your choice

```bash
pip install -e .[<optional-dependencies>]
```

Available options:
- (without optional dependencies): Basic ResidualLearningMPC for custom implementations (e.g. Jacobian approximations, finite-difference approximations for e.g. models without sensitivity information)
- `[pytorch]`: PyTorch models.
- `[gpytorch]`: GPyTorch models.
- `[online_gp]`: GpyTorch models with online-learning improvements.
- not supported yet: JAX, TensorFlow


## Contributing

If you would like to contribute features to `L4acados`, please follow the development installation instructions below to set up your working environment.

### Development installation

1. Install `L4acados` with development dependencies (in addition to the used learning framework, see above):

    ```bash
    pip install -e .[dev]
    ```

2. Sync notebooks with jupytext:
    ```bash
    jupytext --set-formats ipynb,py examples/*/*.ipynb
    ```

3. Add pre-commit hooks
    ```bash
    pre-commit install
    ```

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
