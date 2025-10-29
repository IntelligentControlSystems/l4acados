import torch
import numpy as np
from typing import Optional, Union


class PyTorchFeatureSelector(torch.nn.Module):
    """Creates the Feature selector which is used for GP input dimensionality reduction.

    I.e. generates  :math:`F \in \mathbb{R}^{gp_input_dim \times state_dim}` for :math:`y \sim gp(Fx)`,
    where :math:`gp_input_dim \leq state_dim`. The `input_selection_matrix` is either the full matrix
    or an array of size :math:`n_x` with either zeros or ones where the nonzero elements indicate
    the input features and the sum of nonzero elements corresponds to :math:`gp_input_dim`.

    Args:
        - input_selection: Either a vector or an (gp_input_dim, state_dim) matrix providing the mapping from
        full feature space to GP input feature space. If None, then this represents an identity map.
        - external_inputs: Optional constant feature vector which are concatenated to the extracted features.
        - clock: Optional function which provides the current time. If None, then no time feature is added by default.
        - time_delta: Optional constant specifying the time delta between input points (default: 0.0).
        - device: Device on which the feature selector is stored.
    """

    def __init__(
        self,
        input_selection: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        external_inputs: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        clock: Optional[callable] = None,
        time_delta: Optional[float] = 0.0,
        device="cpu",
    ) -> torch.Tensor:
        super().__init__()

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

        self.clock = clock
        self.time_delta = time_delta

    def __str__(self) -> str:
        return str(self._input_selection_matrix)

    def forward(
        self,
        x_input: torch.Tensor,
        timestamp: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extracts GP features from the full state vector and adds constant features if available.

        Args:
            - x_input in (N, state_dim)
            - timestamp: Optional time feature.

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

        if timestamp is None and self.clock is not None:
            timestamp = self.clock()
        if timestamp is not None:
            if not torch.is_tensor(timestamp):
                timestamp = (
                    timestamp
                    + torch.arange(x_input.size(0), device=features.device).view(-1, 1)
                    * self.time_delta
                )
            features = torch.cat((features, timestamp), dim=-1)

        return features


def get_input_selection_matrix(
    input_selection: Union[list, np.ndarray, torch.Tensor],
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
