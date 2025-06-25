import torch
from torch.func import vmap, jacrev, jacfwd
from typing import Optional

from l4acados.models import ResidualModel
from l4acados.models.pytorch_models.pytorch_feature_selector import (
    PyTorchFeatureSelector,
)
from l4acados.models.pytorch_models.pytorch_utils import to_numpy, to_tensor


class PyTorchResidualModel(ResidualModel):
    """Basic PyTorch residual model class.

    Args:
        - model: A torch.nn.Module model.
        - feature_selector: Optional feature selector mapping controller states and inputs (x,u) to model inputs.
          If set to None, then no selection is performed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_selector: Optional[PyTorchFeatureSelector] = None,
        use_jacfwd=True,
    ):
        self.model = model
        self._feature_selector = (
            feature_selector
            if feature_selector is not None
            else PyTorchFeatureSelector()
        )
        self.device = next(model.parameters()).device
        self.to_numpy = lambda T: to_numpy(T, self.device.type)
        self.to_tensor = lambda X: to_tensor(X, self.device.type)
        self.eval_fun = lambda y: self.model(self._feature_selector(y))
        if use_jacfwd:
            jacfun = jacfwd
        else:
            jacfun = jacrev
        self.jacfun_fun_vmap = vmap(jacfun(lambda y: self.eval_fun(y)))

    def evaluate(self, y, require_grad=False):
        y_tensor = self.to_tensor(y)
        if require_grad:
            self.predictions = self.eval_fun(y_tensor)
        else:
            with torch.no_grad():
                self.predictions = self.eval_fun(y_tensor)

        self.current_prediction = self.to_numpy(self.predictions)
        return self.current_prediction

    def jacobian(self, y):
        y_tensor = self.to_tensor(y)
        self.current_prediction_dy = self.to_numpy(
            # TODO: remove transpose
            self.jacfun_fun_vmap(y_tensor).transpose(0, 1)
        )
        return self.current_prediction_dy

    def value_and_jacobian(self, y):
        """Computes the necessary values for GPMPC

        Args:
            - x_input: (N, state_dim) tensor

        Returns:
            - mean:  (N, residual_dim) tensor
            - mean_dy:  (residual_dim, N, state_dim) tensor
            - covariance:  (N, residual_dim) tensor
        """
        return self.evaluate(y), self.jacobian(y)
