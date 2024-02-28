import torch
import gpytorch
from .base import ResidualModel


class GPyTorchModel(ResidualModel):
    def __init__(self, gp_model):
        self.gp_model = gp_model
        if self.gp_model.train_inputs[0].device.type == "cuda":
            self.to_tensor = lambda X: torch.Tensor(X).cuda()
            self.to_numpy = lambda T: T.cpu().numpy()
        else:
            self.to_tensor = lambda X: torch.Tensor(X)
            self.to_numpy = lambda T: T.numpy()

        def mean_fun_sum(y):
            with gpytorch.settings.fast_pred_var():
                return self.gp_model(y).mean.sum(dim=0)

        self._mean_fun_sum = mean_fun_sum

    def evaluate(self, y):
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=False)
            with torch.no_grad():
                self.predictions = self.gp_model(y_tensor)
        return self.to_numpy(self.predictions.mean)

    def jacobian(self, y):
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=True)
            mean_dy = torch.autograd.functional.jacobian(self._mean_fun_sum, y_tensor)
        return self.to_numpy(mean_dy)

    def value_and_jacobian(self, y):
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=True)
            with torch.no_grad():
                self.predictions = self.gp_model(y_tensor)
            mean_dy = torch.autograd.functional.jacobian(self._mean_fun_sum, y_tensor)

        self.current_mean = self.to_numpy(self.predictions.mean)
        self.current_variance = self.to_numpy(self.predictions.variance)
        self.current_mean_dy = self.to_numpy(mean_dy)

        return self.current_mean, self.current_mean_dy
