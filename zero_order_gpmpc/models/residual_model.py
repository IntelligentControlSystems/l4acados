from abc import ABC, abstractmethod


class ResidualModel(ABC):
    @abstractmethod
    def value_and_jacobian(y):
        raise NotImplementedError

    # @abstractmethod
    # def evaluate(y):
    #     raise NotImplementedError

    # @abstractmethod
    # def jacobian(y):
    #     raise NotImplementedError
