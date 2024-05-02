from typing import Optional

import gpytorch
import torch


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(
        self,
        train_x: Optional[torch.tensor],
        train_y: Optional[torch.tensor],
        likelihood: gpytorch.likelihoods.Likelihood,
        use_ard: bool = False,
        residual_dimension: Optional[int] = None,
        input_dimension: Optional[int] = None,
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
    ):
        super().__init__(train_x, train_y, likelihood)

        if train_y is None and residual_dimension is None:
            raise RuntimeError(
                "train_y and residual_dimension are both None. Please specify one."
            )

        if (
            train_y is not None
            and residual_dimension is not None
            and train_y.size(-1) != residual_dimension
        ):
            raise RuntimeError(
                f"train_y shape {train_y.shape()} and residual_dimension {residual_dimension}"
                " do not correspond."
            )

        if train_x is None and input_dimension is None:
            raise RuntimeError(
                "train_x and input_dimension are both None. Please specify one."
            )

        if use_ard:
            ard_input_shape = (
                train_x.size(-1) if train_x is not None else input_dimension
            )
        else:
            ard_input_shape = None

        residual_dimension = (
            train_y.size(-1) if train_y is not None else residual_dimension
        )

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([residual_dimension])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_input_shape,
                batch_shape=torch.Size([residual_dimension]),
                lengthscale_prior=lengthscale_prior,
            ),
            batch_shape=torch.Size([residual_dimension]),
            outputscale_prior=outputscale_prior,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
