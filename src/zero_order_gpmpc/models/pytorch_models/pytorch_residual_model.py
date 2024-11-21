import torch
from typing import Optional

from zero_order_gpmpc.models import ResidualModel
from zero_order_gpmpc.models.pytorch_models.pytorch_feature_selector import (
    FeatureSelector,
)
from zero_order_gpmpc.models.pytorch_models.pytorch_utils import to_numpy, to_tensor


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
        feature_selector: Optional[FeatureSelector] = None,
    ):
        self.model = model
        self._feature_selector = (
            feature_selector if feature_selector is not None else FeatureSelector()
        )
        self.device = next(model.parameters()).device
        self.to_numpy = lambda T: to_numpy(T, self.device.type)
        self.to_tensor = lambda X: to_tensor(X, self.device.type)

    def _predictions_fun_sum(self, y):
        """Helper function for jacobian computation

        sums up the mean predictions along the first dimension
        (i.e. along the horizon).
        """
        self.evaluate(y, require_grad=True)
        return self.predictions.sum(dim=0)

    def evaluate(self, y, require_grad=False):
        y_tensor = self.to_tensor(y)
        if require_grad:
            self.predictions = self.model(self._feature_selector(y_tensor))
        else:
            with torch.no_grad():
                self.predictions = self.model(self._feature_selector(y_tensor))

        self.current_prediction = self.to_numpy(self.predictions)
        return self.current_prediction

    def jacobian(self, y):
        y_tensor = self.to_tensor(y)
        mean_dy = torch.autograd.functional.jacobian(
            self._predictions_fun_sum, y_tensor
        )
        return self.to_numpy(mean_dy)

    def value_and_jacobian(self, y):
        """Computes the necessary values for GPMPC

        Args:
            - x_input: (N, state_dim) tensor

        Returns:
            - mean:  (N, residual_dim) tensor
            - mean_dy:  (residual_dim, N, state_dim) tensor
            - covariance:  (N, residual_dim) tensor
        """
        self.current_prediction_dy = self.jacobian(y)
        return self.current_prediction, self.current_prediction_dy
