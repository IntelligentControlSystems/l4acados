# L4acados

Integrate learning-based Python models into acados for real-time model predictive control.

## Usage

1. Define your [`AcadosOcp`](https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp), including the nominal dynamics model
    ```python
    from acados_template import AcadosOcp
    ocp = AcadosOcp()
    # ...
    ```

2. Define the residual model using the `L4acados` `ResidualModel` (here as a [PyTorchResidualModel](https://github.com/IntelligentControlSystems/l4acados/blob/main/src/l4acados/models/pytorch_models/pytorch_residual_model.py) example):
    ```python
    import l4acados as l4a
    residual_model = l4a.PyTorchResidualModel(your_pytorch_model)
    ```

    > [Other models](https://github.com/IntelligentControlSystems/l4acados/tree/main/src/l4acados/models) can be straightforwardly implemented using as a [`ResidualModel`](https://github.com/IntelligentControlSystems/l4acados/blob/main/src/l4acados/models/residual_model.py) instance; see [here](https://github.com/IntelligentControlSystems/l4acados?tab=readme-ov-file#install-l4acados-with-optional-dependencies-of-your-choice) for already available residual models.

3. Generate the `L4acados` solver object
    ```python
    l4acados_solver = l4a.ResidualLearningMPC(
        ocp=ocp,
        residual_model=residual_model,
        use_cython=True, # accelerate get/set operations by using the acados Cython interface
    )
    ```

4. Done! The `ResidualLearningMPC` object can be interfaced like the `AcadosOcpSolver`:
    - `l4acados_solver.set(...)`
    - `l4acados_solver.solve()`
    - `l4acados_solver.get(...)`
    > Not all solver interface functions are mapped by the `ResidualLearningMPC`. You can still access the underlying `AcadosOcpSolver` object through the `l4acados_solver.ocp_solver` property. Besides the dynamics model and parameter definition, the `l4acados_solver.ocp_solver` is equivalent to the `acados_solver.ocp_solver` generated from the original `ocp`.


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

    > You can also install `acados` separately; in this case, refer to the acados version of the submodule in case you encounter errors with a newer version of `acados`.

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
- `[gpytorch-exo]`: GpyTorch models with online-learning improvements.
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

If you use this software, please cite our corresponding articles as written below.

### General software

```
@online{lahr_l4acados_2024,
  title = {L4acados: {{Learning-based}} Models for Acados, Applied to {{Gaussian}} Process-Based Predictive Control},
  shorttitle = {L4acados},
  author = {Lahr, Amon and Näf, Joshua and Wabersich, Kim P. and Frey, Jonathan and Siehl, Pascal and Carron, Andrea and Diehl, Moritz and Zeilinger, Melanie N.},
  date = {2024-11-28},
  eprint = {2411.19258},
  eprinttype = {arXiv},
  doi = {10.48550/arXiv.2411.19258},
  pubstate = {prepublished}
}
```

### Zero-Order GP-MPC algorithm

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
