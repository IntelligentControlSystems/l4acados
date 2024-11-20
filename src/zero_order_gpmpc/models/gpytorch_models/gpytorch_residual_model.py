from typing import Optional, Union
import torch
import gpytorch

from ..residual_model import ResidualModel

import numpy as np
import torch


class FeatureSelector:
    """Creates the Feature selector which is used for GP input dimensionality reduction.

    I.e. generates  :math:`F \in \mathbb{R}^{gp_input_dim \times state_dim}` for :math:`y \sim gp(Fx)`,
    where :math:`gp_input_dim \leq state_dim`. The `input_selection_matrix` is either the full matrix
    or an array of size :math:`n_x` with either zeros or ones where the nonzero elements indicate
    the input features and the sum of nonzero elements corresponds to :math:`gp_input_dim`.

    Args:
        - input_selection: Either a vector or an (gp_input_dim, state_dim) matrix providing the mapping from
          full feature space to GP input feature space. If None, then this represents an identity map.
        - external_inputs: Optional constant feature vector which are concatenated to the extracted features.
    """

    def __init__(
        self,
        input_selection: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        external_inputs: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        device="cpu",
    ) -> torch.Tensor:
        if input_selection is not None:
            if isinstance(input_selection, (list, np.ndarray)):
                input_selection = torch.Tensor(input_selection)

            if input_selection.ndim == 1:
                input_selection_matrix = get_input_selection_matrix(input_selection)
            elif input_selection.ndim == 2:
                input_selection_matrix = input_selection
            else:
                raise ValueError(
                    f"input_selection must be 1D or 2D but has shape {input_selection.size()}"
                )

            self._input_selection_matrix = input_selection_matrix.to(device=device)

        else:
            self._input_selection_matrix = None

        if external_inputs is not None:
            if isinstance(external_inputs, (list, np.ndarray)):
                self.external_inputs = torch.Tensor(external_inputs, device=device)
            elif isinstance(external_inputs, torch.Tensor):
                self.external_inputs = external_inputs.to(device=device)
            else:
                raise RuntimeError(
                    f"Unexpected type {type(external_inputs)} for constant_features."
                    f"Supported types are list, np.array, torch.tensor."
                )

            if self.external_inputs.ndim == 1:
                self.external_inputs = self.external_inputs.unsqueeze(-1)
            elif self.external_inputs.ndim != 2:
                raise ValueError("constant_features has invalid number of dimensions")

        else:
            self.external_inputs = None

    def __str__(self) -> str:
        return str(self._input_selection_matrix)

    def __call__(self, x_input: torch.Tensor) -> torch.Tensor:
        """Extracts GP features from the full state vector and adds constant features if available.

        Args:
            - x_input in (N, state_dim)

        Returns:
            - extracted features in (N, gp_input_dim)
        """

        if self._input_selection_matrix is not None:
            features = torch.matmul(x_input, self._input_selection_matrix.T)
        else:
            features = x_input

        if self.external_inputs is not None:
            features = torch.cat(
                (features, self.external_inputs[: x_input.size(0), :]), dim=-1
            )

        return features


def get_input_selection_matrix(
    input_selection: Union[list, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Generate an input selection matrix from a 1D vector.

    Args:
        - input_selection: 1D selection vector of type list, np.ndarray or torch.Tensor.
          The nonzero elements indicate the selected input features.

    Returns:
        - input_selection_matrix: (gp_input_dim, state_dim) 2D matrix of type torch.Tensor.
          Provides a mapping from the full feature space to the selected GP input feature space.
    """
    if isinstance(input_selection, (list, np.ndarray)):
        input_selection = torch.Tensor(input_selection)

    if input_selection.ndim != 1:
        raise ValueError("input_selection must be a 1D vector")

    input_selection_matrix = torch.diag(input_selection)[~(input_selection == 0), :]

    return input_selection_matrix


class GPyTorchResidualModel(ResidualModel):
    """Basic gpytorch based GP residual model class.

    Args:
        - gp_model: A conditioned and trained instance of a gpytorch.models.ExactGP.
        - feature_selector: Optional feature selector if certain state dimensions are
          are known to be irrelevant for the GP inference. If set to None, then no selection
          is performed.
    """

    def __init__(
        self,
        gp_model: gpytorch.models.ExactGP,
        feature_selector: Optional[FeatureSelector] = None,
    ):
        self.gp_model = gp_model

        self._gp_feature_selector = (
            feature_selector if feature_selector is not None else FeatureSelector()
        )

        # TODO(@naefjo): fix uncond GP
        if (
            gp_model.train_inputs is not None
            and self.gp_model.train_inputs[0].device.type == "cuda"
        ):
            self.to_tensor = lambda X: torch.Tensor(X).cuda()
            self.to_numpy = lambda T: T.cpu().detach().numpy()
        else:
            self.to_tensor = lambda X: torch.Tensor(X)
            self.to_numpy = lambda T: T.detach().numpy()

    def _mean_fun_sum(self, y):
        """Helper function for jacobian computation

        sums up the mean predictions along the first dimension
        (i.e. along the horizon).
        """
        self.evaluate(y, require_grad=True)
        return self.predictions.mean.sum(dim=0)

    def evaluate(self, y, require_grad=False):
        # NOTE(@naefjo): covar_root_decomposition=False forces linear_operator to use cholesky.
        # This is needed as otherwise our approach falls apart.
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(
            covar_root_decomposition=False
        ):
            y_tensor = self.to_tensor(y)
            if require_grad:
                self.predictions = self.gp_model(self._gp_feature_selector(y_tensor))
            else:
                with torch.no_grad():
                    self.predictions = self.gp_model(
                        self._gp_feature_selector(y_tensor)
                    )

        self.current_mean = self.to_numpy(self.predictions.mean)

        # NOTE(@naefjo): If we skipped posterior covariances, we keep the old covars in cache for hewing method.
        if gpytorch.settings.skip_posterior_variances.off():
            self.current_variance = self.to_numpy(self.predictions.variance)

        return self.current_mean

    def jacobian(self, y):
        y_tensor = self.to_tensor(y)
        mean_dy = torch.autograd.functional.jacobian(self._mean_fun_sum, y_tensor)
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
        self.current_mean_dy = self.jacobian(y)

        return self.current_mean, self.current_mean_dy
