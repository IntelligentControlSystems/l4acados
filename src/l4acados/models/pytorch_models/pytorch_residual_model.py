import torch
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
        feature_selector: Optional[PyTorchFeatureSelector] = PyTorchFeatureSelector(),
        use_jacfwd: bool = True,
        measure_to_tensor_time: bool = False,
    ):
        self.model = model
        self._feature_selector = feature_selector
        self.device = next(model.parameters()).device

        self.to_numpy = lambda T: to_numpy(
            T, self.device.type, synchronize=measure_to_tensor_time
        )
        self.to_tensor = lambda X: to_tensor(
            X, self.device.type, synchronize=measure_to_tensor_time
        )
        self.to_tensor_time = 0.0

    def _predictions_fun_sum(self, y):
        """Helper function for jacobian computation

        sums up the mean predictions along the first dimension
        (i.e. along the horizon).
        """
        self.evaluate(y, require_grad=True)
        return self.predictions.sum(dim=0)

    def evaluate(self, y, require_grad=False):
        y_tensor, time_to_tensor = self.to_tensor(y)
        if require_grad:
            self.predictions = self.model(self._feature_selector(y_tensor))
        else:
            with torch.no_grad():
                self.predictions = self.model(self._feature_selector(y_tensor))

        self.current_prediction, time_to_numpy = self.to_numpy(self.predictions)
        self.to_tensor_time += time_to_tensor + time_to_numpy
        return self.current_prediction

    def jacobian(self, y):
        y_tensor, time_to_tensor = self.to_tensor(y)
        self.predictions_dy = torch.autograd.functional.jacobian(
            self._predictions_fun_sum, y_tensor
        )
        self.current_prediction_dy, time_to_numpy = self.to_numpy(self.predictions_dy)
        self.to_tensor_time += time_to_tensor + time_to_numpy
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
