from abc import ABC, abstractmethod
import numpy as np


class ResidualModel(ABC):
    @abstractmethod
    def value_and_jacobian(y: np.array):
        raise NotImplementedError
