try:
    import gpytorch

    _gpytorch_available = True
except ImportError:
    _gpytorch_available = False

if _gpytorch_available:

    from .gpytorch_residual_model import GPyTorchResidualModel

else:

    err_msg = "GPyTorch is not available. 'l4acados.gpytorch_models' requires the [gpytorch] optional dependencies (see pyproject.toml)."

    class GPyTorchResidualModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(err_msg)
