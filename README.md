# L4acados

Integrate learning-based Python models into acados for real-time model predictive control.

## Installation

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

3. Install acados
    ```bash
        pip install -e external/acados/interfaces/acados_template
    ```

3. Set environment variables (where `$PROJECT_DIR` is the directory you cloned the repository into in Step 1):
    ```bash
        export ACADOS_SOURCE_DIR=$PROJECT_DIR/external/acados
        export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
    ```

4. Test your installation by executing the example script
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
