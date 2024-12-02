try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:

    from .pytorch_feature_selector import PyTorchFeatureSelector
    from .pytorch_residual_model import PyTorchResidualModel

else:

    from l4acados.models import ResidualModel

    err_msg = "Torch is not available. 'l4acados.pytorch_models' requires the [pytorch] optional dependencies (see pyproject.toml)."

    class PyTorchFeatureSelector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(err_msg)

    class PyTorchResidualModel(ResidualModel):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(err_msg)
