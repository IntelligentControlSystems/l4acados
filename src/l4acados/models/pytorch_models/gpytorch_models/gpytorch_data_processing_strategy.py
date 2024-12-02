from abc import ABC
import threading
from typing import Optional

import numpy as np
import torch
import gpytorch
import linear_operator

from ..pytorch_feature_selector import PyTorchFeatureSelector
from ..pytorch_utils import to_numpy, to_tensor


class DataProcessingStrategy(ABC):
    """Abstract base class for the data processing strategies.

    Depending on the experiment/operating mode of the controller, the data processing
    strategy might be different. To enable as much flexibility as possible, the data
    processing strategy is impelented using a strategy pattern, allowing the user to
    easily define their own strategies and swap them out when necessary
    """

    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:
        """Function which is processed in the `record_datapoint` method of `ResidualGaussianProcess`

        Args:
            - residual_gp_instance: Instance of the `ResidualGaussianProcess` class so we can access
              relevant attributes
            - x_input: data which should be saved. Should have dimension (state_dimension,) or equivalent
            - y_target: the residual which was measured at x_input. Should have dimension
              (residual_dimension,) or equivalent
        """
        raise NotImplementedError


class VoidDataStrategy(DataProcessingStrategy):
    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:
        pass


class RecordDataStrategy(DataProcessingStrategy):
    """Implements a processing strategy which saves the data continuously to a file.

    The strategy keeps a buffer of recent datapoints and asynchronously saves the buffer to
    a file.
    """

    def __init__(self, x_data_path: str, y_data_path: str, buffer_size: int = 50):
        """Construct the data recorder.

        Args:
            - x_data_path: file path where x data should be saved.
            - y_data_path: file path where residual data should be saved
        """
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.buffer_size = buffer_size
        self._gp_training_data = {"x_training_data": [], "y_training_data": []}

    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:
        self._gp_training_data["x_training_data"].append(x_input)
        self._gp_training_data["y_training_data"].append(y_target)

        if len(self._gp_training_data["x_training_data"]) == self.buffer_size:
            # Do we need a local copy?
            save_data_x = np.array(self._gp_training_data["x_training_data"])
            save_data_y = np.array(self._gp_training_data["y_training_data"])

            self._gp_training_data["x_training_data"].clear()
            self._gp_training_data["y_training_data"].clear()

            threading.Thread(
                target=lambda: (
                    RecordDataStrategy._save_to_file(save_data_x, self.x_data_path),
                    RecordDataStrategy._save_to_file(save_data_y, self.y_data_path),
                    print("saved gp training data"),
                )
            ).start()

    @staticmethod
    def _save_to_file(data: np.ndarray, filename: str) -> None:
        """Appends data to a file"""
        with open(filename, "ab") as f:
            # f.write(b"\n")
            np.savetxt(
                f,
                data,
                delimiter=",",
            )


class OnlineLearningStrategy(DataProcessingStrategy):
    """Implements an online learning strategy.

    The received data is incorporated in the GP and used for further predictions.
    This data processing strategy depends on the [online_gp] optional dependencies (see pyproject.toml).
    """

    def __init__(self, max_num_points: int = 200, device: str = "cpu") -> None:
        self.max_num_points = max_num_points
        self.device = device

    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:

        # Convert to tensor
        if not torch.is_tensor(x_input):
            x_input = to_tensor(arr=x_input, device=self.device)

        if not torch.is_tensor(y_target):
            y_target = to_tensor(arr=y_target, device=self.device)

        # Extend to 2D for further computation
        x_input = torch.atleast_2d(x_input)
        y_target = torch.atleast_2d(y_target)

        if (
            gp_model.prediction_strategy is None
            or gp_model.train_inputs is None
            or gp_model.train_targets is None
        ):
            if gp_model.train_inputs is not None:
                raise RuntimeError(
                    "train_inputs in GP is not None. Something went wrong."
                )

            # Set the training data and return (in-place modification)
            gp_model.set_train_data(
                gp_feature_selector(x_input),
                y_target,
                strict=False,
            )
            return

        # Check if GP is already full
        if gp_model.train_inputs[0].shape[-2] >= self.max_num_points:
            with torch.no_grad():
                selector = torch.ones(self.max_num_points, requires_grad=False)
                # TODO(@naefjo): Add super cool logic to determine which points to kick out. Ideally with O(-n^3)
                drop_idx = torch.randint(
                    0, self.max_num_points, torch.Size(), requires_grad=False
                ).item()
                selector[drop_idx] = 0
                # Calculate fantasy model with data selector
                try:
                    fantasy_model = gp_model.get_fantasy_model(
                        gp_feature_selector(x_input),
                        y_target,
                        data_selector=selector,
                    )
                except TypeError as err:
                    # check if error message contains data_selector
                    if "data_selector" in str(err):
                        raise ImportError(
                            "OnlineLearningStrategy requires the [online_gp] optional dependencies (see pyproject.toml)."
                        )
                    raise err

                return fantasy_model

        with torch.no_grad():
            # Add observation and return updated model
            fantasy_model = gp_model.get_fantasy_model(
                gp_feature_selector(x_input), y_target
            )

            return fantasy_model
