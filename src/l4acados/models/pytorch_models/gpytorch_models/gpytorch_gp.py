from typing import Optional, Union, Tuple

import gpytorch
import torch

from linear_operator.operators import (
    KroneckerProductLinearOperator,
    CholLinearOperator,
    TriangularLinearOperator,
    IdentityLinearOperator,
    RootLinearOperator,
    DiagLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky


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


class BatchIndependentInducingPointGpModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: Optional[torch.tensor],
        train_y: Optional[torch.tensor],
        likelihood: gpytorch.likelihoods.Likelihood,
        inducing_points: Union[int, torch.tensor] = 10,
        use_ard: bool = False,
        residual_dimension: Optional[int] = None,
        input_dimension: Optional[int] = None,
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
    ):
        """
        TODO(@naefjo): write doc
        """
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
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_input_shape,
                batch_shape=torch.Size([residual_dimension]),
                lengthscale_prior=lengthscale_prior,
            ),
            batch_shape=torch.Size([residual_dimension]),
            outputscale_prior=outputscale_prior,
        )

        if isinstance(inducing_points, int):
            permuted_indices = torch.randperm(train_x.size(0))
            inducing_point_indices = permuted_indices[:inducing_points]
            inducing_points = train_x[inducing_point_indices, :]

        self.num_inducing_points = inducing_points.size(0)

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_covar_module, inducing_points=inducing_points, likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class BatchIndependentApproximateSpatioTemporalGPModel(gpytorch.models.GP):
    """Approximate spatio-temporal GP model for fast online learning.

    Uses inducing points in the spatial domain whose distribution can be updated online
    using a Kalman filter. Therefore, the temporal component of the covariance function is
    (approximately) represented in state-space form. Updating only works forward in time and
    the current estimate is defined at time t=0. For making predictions, a time delta must be
    appended to the spatial input features.

    Args:
        - x_train: Spatial input data
        - y_train: Residual data
        - t_train: Temporal input data. Alternatively, a constant time step size dt can be provided
        - inducing_points: Spatial inducing points
            If an integer is provided, random points are selected from x_train
            If None, x_train is used as inducing points
        - likelihood: Multitask Gaussian likelihood. Number of tasks must match the residual dimension
        - spatial_covariance: Spatial covariance function. Must only take the spatial input dimensions
        - temporal_covariance: Temporal covariance function. Must only take the temporal input dimension
        - use_ard: Whether to use separate lengthscales for each input dimension
        - spatial_input_dimension: Dimension of the spatial input data. Must be specified if x_train is None
        - residual_dimension: Dimension of the residual data. Must be specified if y_train is None
        - dt: Fixed time step size. If None and t_train is None, a constant time step of size 0 is assumed
        - dtype: Data type of the model parameters
        - inducing_point_optimization_iterations: Number of optimization iterations for the inducing points

    Attributes:
        - dt: Time step size (only if fixed time step is used)
        - likelihood: Multitask Gaussian likelihood
        - spatial_kernel: Spatial component of the covariance function of the spatio-temporal GP
        - temporal_kernel: Temporal component of the covariance function of the spatio-temporal GP
        - inducing_points: Spatial inducing points of the approximate GP
        - K_ZZ: Evaluated prior covariance matrix of the pseudo points
        - pseudo_mean_proj: Current posterior mean of the pseudo points estimate
            Projected such that they are ready for predictions to save computation time
        - pseudo_covar_proj: Current posterior covariance matrix of the pseudo points estimate
            Projected such that they are ready for predictions to save computation time
        - state_space_model: Linear state-space representation of the temporal covariance function
            Introduces a Markov process in the latent space of the inducing points
        - latent_inducing_mean: Current mean of the inducing points in latent space
        - latent_inducing_covar: Current covariance matrix of the inducing points in latent space
        - A_bar: Discrete-time state transition matrix of the inducing points Markov process in latent space
            (only if fixed time step is used)
        - Q_bar: Discrete-time process noise covariance matrix of the inducing points Markov process in latent space
            (only if fixed time step is used)
        - H_bar: Observation matrix mapping the inducing points from the latent space to the residual space
        - t_update: Last time stamp of the training data (only if no fixed time step is used)
    """

    def __init__(
        self,
        train_x: Optional[torch.Tensor] = None,
        train_y: Optional[torch.Tensor] = None,
        train_t: Optional[torch.Tensor] = None,
        inducing_points: Optional[Union[torch.Tensor, int]] = None,
        likelihood: Optional[
            Union[
                gpytorch.likelihoods.MultitaskGaussianLikelihood,
                gpytorch.likelihoods.GaussianLikelihood,
            ]
        ] = None,
        mean: Optional[gpytorch.means.Mean] = None,
        spatial_covariance: Optional[gpytorch.kernels.Kernel] = None,
        temporal_covariance: Optional[gpytorch.kernels.MaternKernel] = None,
        use_ard: bool = False,
        spatial_input_dimension: Optional[int] = None,
        residual_dimension: Optional[int] = None,
        dt: Optional[float] = None,
        dtype: Optional[torch.dtype] = torch.float64,
        inducing_point_optimization_iterations: Optional[int] = 0,
    ):
        super().__init__()

        with torch.no_grad():
            if train_x is not None:
                self.train_inputs = [
                    torch.atleast_2d(train_x).detach().requires_grad_(False)
                ]
                self.spatial_input_dimension = self.train_inputs[0].size(1)
                if (
                    spatial_input_dimension is not None
                    and spatial_input_dimension != self.spatial_input_dimension
                ):
                    raise RuntimeError(
                        f"x_train shape {self.train_inputs[0].size()} and input_dim {spatial_input_dimension}"
                        " do not correspond"
                    )
            elif spatial_input_dimension is not None:
                self.spatial_input_dimension = spatial_input_dimension
                self.train_inputs = None
            elif isinstance(inducing_points, torch.Tensor):
                self.spatial_input_dimension = inducing_points.size(1)
                self.train_inputs = None
            else:
                raise RuntimeError(
                    "x_train and input_dim are both None. Please specify one."
                )
            if train_y is not None:
                if train_y.ndim == 1:
                    self.train_targets = (
                        train_y.unsqueeze(-1).detach().requires_grad_(False)
                    )
                else:
                    self.train_targets = train_y.detach().requires_grad_(False)
                if self.train_targets.size(0) != self.train_inputs[0].size(0):
                    raise RuntimeError(
                        f"x_train shape {self.train_inputs[0].size()} and y_train shape"
                        f" {self.train_targets.shape} do not correspond"
                    )
                self.residual_dimension = self.train_targets.size(1)
                if (
                    residual_dimension is not None
                    and residual_dimension != self.residual_dimension
                ):
                    raise RuntimeError(
                        f"y_train shape {self.train_targets.shape} and residual_dim"
                        f" {residual_dimension} do not correspond"
                    )
            elif residual_dimension is not None:
                self.residual_dimension = residual_dimension
                self.train_targets = None
            else:
                raise RuntimeError(
                    "y_train and residual_dim are both None. Please specify one."
                )

        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.residual_dimension,
                has_global_noise=False,
            )
        elif isinstance(likelihood, gpytorch.likelihoods.MultitaskGaussianLikelihood):
            self.likelihood = likelihood
            if self.likelihood.num_tasks != self.residual_dimension:
                raise RuntimeError(
                    f"num_tasks of likelihood {self.likelihood.num_tasks} and residual_dim"
                    f" {self.residual_dimension} do not correspond"
                )
        elif isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.residual_dimension,
                has_global_noise=False,
            )
            self.likelihood.task_noises = torch.full(
                (self.residual_dimension,), likelihood.noise.item()
            )
        else:
            raise RuntimeError(
                "likelihood must be of type MultitaskGaussianLikelihood or GaussianLikelihood"
            )
        if mean is None:
            mean = gpytorch.means.ZeroMean(
                batch_shape=torch.Size([self.residual_dimension])
            )
        if spatial_covariance is None:
            if use_ard:
                ard_spatial_input_shape = self.spatial_input_dimension
            else:
                ard_spatial_input_shape = None
            spatial_covariance = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([self.residual_dimension]),
                    active_dims=range(self.spatial_input_dimension),
                    ard_num_dims=ard_spatial_input_shape,
                ),
                batch_shape=torch.Size([self.residual_dimension]),
            )
        if temporal_covariance is None:
            temporal_covariance = gpytorch.kernels.MaternKernel(
                nu=0.5,
                batch_shape=torch.Size([self.residual_dimension]),
                active_dims=[self.spatial_input_dimension],
            )
        self.is_initialized = False

        self.mean_module = mean
        self.covar_module = gpytorch.kernels.ProductKernel(
            spatial_covariance, temporal_covariance
        )

        if inducing_points is not None:
            if isinstance(inducing_points, int):
                if self.train_inputs is not None:
                    inducing_points_idx = torch.randperm(self.train_inputs[0].size(0))[
                        :inducing_points
                    ]
                    self.register_parameter(
                        name="inducing_points",
                        parameter=torch.nn.Parameter(
                            self.train_inputs[0][inducing_points_idx, :].type(dtype)
                        ),
                    )
                else:
                    raise RuntimeError(
                        "x_train is None. Cannot select inducing points."
                    )
            elif isinstance(inducing_points, torch.Tensor):
                self.register_parameter(
                    name="inducing_points",
                    parameter=torch.nn.Parameter(
                        torch.atleast_2d(inducing_points).type(dtype)
                    ),
                )
                if self.inducing_points.size(1) != self.spatial_input_dimension:
                    raise RuntimeError(
                        f"inducing_points shape {self.inducing_points.shape}"
                        f" and input_dim {self.spatial_input_dimension} do not correspond"
                    )
            else:
                raise RuntimeError(
                    "inducing_points must be of type int or torch.Tensor"
                )
        elif self.train_inputs is not None:
            print(
                "inducing_points is None. Using full data set x_train as inducing points."
            )
            self.register_parameter(
                name="inducing_points",
                parameter=torch.nn.Parameter(self.train_inputs[0].type(dtype)),
            )
        else:
            raise RuntimeError(
                "inducing_points and x_train are both None. Please specify one."
            )
        if inducing_point_optimization_iterations > 0:
            if self.train_inputs is None:
                raise RuntimeError("x_train is None. Cannot optimize inducing points.")
            self.optimize_inducing_points(
                max_iter=inducing_point_optimization_iterations
            )
        if dt is not None:
            self.dt = dt
            print("dt is provided. Any other temporal input will be ignored.")
            self.use_constant_timestep = True
            self.time_inputs = None
        elif train_t is not None:
            self.use_constant_timestep = False
            self.time_inputs = train_t.reshape(-1, 1)
            if self.time_inputs.size(0) != self.train_inputs[0].size(0):
                raise RuntimeError(
                    f"x_train shape {self.train_inputs[0].size()} and t_train shape {self.time_inputs.shape}"
                    " do not correspond"
                )
        else:
            print(
                "dt and t_train are both None. Assuming constant time step of size 0."
            )
            self.dt = 0.0
            self.use_constant_timestep = True
            self.time_inputs = None

    def optimize_inducing_points(self, max_iter: int = 100, lr: float = 0.01):
        """Optimize the spatial inducing point locations.

        Minimize the determinant of the prior covariance in the training data that cannot be explained
        by the inducing points.

        Args:
            - max_iter: Maximum number of optimization iterations
            - lr: Learning rate of the optimizer
        """

        optimizer = torch.optim.Adam([self.inducing_points], lr=lr)

        print("Optimizing inducing points...")
        K_XX = (
            self.covar_module.kernels[0](self.train_inputs[0])
            .add_jitter(1e-6)
            .to_dense()
            .detach()
        )
        for i in range(max_iter):
            optimizer.zero_grad()
            K_ZZ = (
                self.covar_module.kernels[0](self.inducing_points)
                .add_jitter(1e-6)
                .to_dense()
            )
            K_XZ = self.covar_module.kernels[0](
                self.train_inputs[0], self.inducing_points
            ).to_dense()
            Q = K_XZ @ torch.linalg.solve(K_ZZ, K_XZ.mT)
            loss = torch.logdet(K_XX - Q).sum()
            loss.backward()
            optimizer.step()
            print(f"Iteration {i+1}/{max_iter}: Loss {loss.item()}")

    def initialize(
        self,
        train_x: torch.Tensor = None,
        train_y: torch.Tensor = None,
        train_t: torch.Tensor = None,
        **kwargs,
    ):
        super().initialize(**kwargs)
        for param in self.parameters():
            param.type(self.dtype)

        with torch.no_grad():

            # precomputed (cached) covariance matrices needed
            if not torch.all(
                self.covar_module.kernels[0].active_dims
                == torch.tensor(range(self.spatial_input_dimension))
            ):
                raise RuntimeError(
                    "Only the spatial input dimensions of the spatial kernel can be active. (spatial_kernel.active_dims = range(spatial_input_dimension))"
                )
            self.K_ZZ_chol = (
                self.covar_module.kernels[0](self.inducing_points)
                .add_jitter(1e-9)  # add jitter for numerical stability
                .type(self.dtype)
                .cholesky()
            )
            self.K_ZZ_inv_chol = self.K_ZZ_chol.inverse().mT

            # matrices needed for Kalman filtering
            if not torch.all(
                self.covar_module.kernels[1].active_dims
                == torch.tensor([self.spatial_input_dimension])
            ):
                raise RuntimeError(
                    "Only the last dimension of the temporal kernel can be active. (temporal_kernel.active_dims = [spatial_input_dimension])"
                )
            self.state_space_model = StateSpaceGPModel(self.covar_module.kernels[1])
            if self.use_constant_timestep:
                A, Q = self.state_space_model.get_discrete_time_matrices(self.dt)
                self.A_bar = KroneckerProductLinearOperator(
                    IdentityLinearOperator(self.num_inducing_points), A
                ).type(self.dtype)
                self.Q_bar_chol = (
                    KroneckerProductLinearOperator(self.K_ZZ_chol, psd_safe_cholesky(Q))
                    .type(self.dtype)
                    .to_dense()
                )
            self.H_bar = KroneckerProductLinearOperator(
                IdentityLinearOperator(self.num_inducing_points),
                self.state_space_model.H,
            ).type(self.dtype)

            # initial distribution of inducing points in latent space
            self.latent_inducing_mean = torch.zeros(
                (self.num_inducing_points * self.state_space_model.latent_dimension, 1),
                dtype=self.dtype,
            )
            self.latent_inducing_covar_chol = (
                KroneckerProductLinearOperator(
                    self.K_ZZ_chol, psd_safe_cholesky(self.state_space_model.P_inf)
                )
                .type(self.dtype)
                .to_dense()
            )

            self.latent_output_map = self.K_ZZ_inv_chol.mT @ self.H_bar
            self.t_update = 0.0

        if train_x is not None and train_y is not None:
            self.train_inputs = [
                torch.atleast_2d(train_x).detach().requires_grad_(False)
            ]
            self.train_targets = train_y.detach().requires_grad_(False)
            if not self.use_constant_timestep:
                self.time_inputs = train_t

        self.unconditioned_model = True
        if self.train_inputs is not None and self.train_targets is not None:
            self.update(self.train_inputs[0], self.train_targets, self.time_inputs)

        if self.use_constant_timestep:
            self.dt_pred = torch.tensor(self.dt, dtype=self.dtype)
            self.A_bar_pred = self.A_bar.unsqueeze(0)
            self.Q_bar_pred_chol = self.Q_bar_chol.unsqueeze(0)
        else:
            self.dt_pred = torch.tensor(-1, dtype=self.dtype)

        self.is_initialized = True
        return self

    @property
    def num_inducing_points(self):
        return self.inducing_points.size(0)

    @property
    def dtype(self):
        return self.inducing_points.dtype

    def update(
        self, x: torch.Tensor, y: torch.Tensor, t: Optional[torch.Tensor] = None
    ):
        """Use Kalman filter equations to update the current estimate of the latent inducing points.

        Args:
            - x: Spatial input data
            - y: Residual data
            - t: Temporal input data (optional)
        """

        if x.size(0) != y.size(0):
            raise RuntimeError(
                f"Size of training data x {x.size()} and y {y.size()} do not correspond."
            )
        if x.size(1) != self.spatial_input_dimension:
            if x.size(1) == self.spatial_input_dimension + 1 and t is None:
                t = x[..., -1]
                x = x[..., :-1]
            else:
                raise RuntimeError(
                    f"Size of training data x {x.size()} does not correspond to spatial_input_dim {self.spatial_input_dimension}."
                )
        if y.size(1) != self.residual_dimension:
            raise RuntimeError(
                f"Size of training data y {y.size()} does not correspond to residual_dim {self.residual_dimension}."
            )
        y = y.T.unsqueeze(-1)
        if not self.use_constant_timestep:
            if t is None:
                raise RuntimeError("t is None. Please provide temporal input data.")
            if t.size(0) != x.size(0):
                raise RuntimeError(
                    f"Size of training data x {x.size()} and t {t.size()} do not correspond."
                )

        with torch.no_grad():
            N = x.size(0)
            m_X = self.mean_module(x).unsqueeze(-1)
            K_XX_diag = self.covar_module.kernels[0](x, diag=True)
            K_XZ = (
                self.covar_module.kernels[0](x, self.inducing_points)
                .type(self.dtype)
                .to_dense()
            )

            interp_term = K_XZ @ self.K_ZZ_inv_chol
            R_diag = (
                K_XX_diag
                - interp_term.square().sum(dim=-1)
                + self.likelihood.task_noises.view(self.residual_dimension, 1)
            )
            C_bar = interp_term @ self.latent_output_map
            if t is None:
                A_bar = self.A_bar
                Q_bar_chol = self.Q_bar_chol
                self.t_update += N * self.dt

            for i in range(N):
                if t is not None:
                    dt = t[i] - self.t_update
                    assert dt >= 0.0, "time stamps must be non-decreasing"
                    self.t_update = t[i]
                    if (
                        self.use_constant_timestep
                        and torch.round(dt / self.dt).item() == 1
                    ):
                        A_bar = self.A_bar
                        Q_bar_chol = self.Q_bar_chol
                    else:
                        A, Q = self.state_space_model.get_discrete_time_matrices(dt)
                        A_bar = KroneckerProductLinearOperator(
                            IdentityLinearOperator(self.num_inducing_points), A
                        ).type(self.dtype)
                        Q_bar_chol = (
                            KroneckerProductLinearOperator(
                                self.K_ZZ_chol, psd_safe_cholesky(Q)
                            )
                            .type(self.dtype)
                            .to_dense()
                        )

                latent_inducing_mean_pred = A_bar @ self.latent_inducing_mean
                latent_inducing_covar_pred_root = torch.cat(
                    (A_bar @ self.latent_inducing_covar_chol, Q_bar_chol), dim=-1
                )

                z = (
                    y[..., i : i + 1, :]
                    - m_X[..., i : i + 1, :]
                    - C_bar[..., i : i + 1, :] @ latent_inducing_mean_pred
                )
                L = C_bar[..., i : i + 1, :] @ latent_inducing_covar_pred_root
                R = R_diag[..., i : i + 1].unsqueeze(-1)
                K = torch.linalg.solve(
                    L @ L.mT + R, L @ latent_inducing_covar_pred_root.mT
                ).mT

                self.latent_inducing_mean = latent_inducing_mean_pred + K @ z
                latent_inducing_covar_root = torch.cat(
                    (latent_inducing_covar_pred_root - K @ L, K @ R.sqrt()), dim=-1
                )
                self.latent_inducing_covar_chol = psd_safe_cholesky(
                    latent_inducing_covar_root @ latent_inducing_covar_root.mT
                )

            self.unconditioned_model = False

    def predict(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the posterior mean and covariance of the residual at the augmented input points.

        Args:
            - x: (Spatial) GP input data points
            - t: Temporal data of the GP input data points (optional)

        Returns:
            - mean: Posterior mean of the residual
            - covar: Posterior covariance of the residual (temporal covariance is approximated by diagonal)
        """

        if not self.is_initialized:
            print("Model is not initialized yet. Initializing now.")
            self.initialize()

        if x.size(1) != self.spatial_input_dimension:
            raise RuntimeError(
                f"Size of x {x.size()} does not correspond to spatial_input_dim {self.spatial_input_dimension}."
            )
        if t is not None:
            if x.size(0) != t.size(0):
                raise RuntimeError(
                    f"Size of x {x.size()} and t {t.size()} do not correspond."
                )
            if self.unconditioned_model and self.use_constant_timestep:
                self.t_update = t[0].item()
                # doesn't matter for estimates and makes inference more efficient
            dt = t.view(-1).type(self.dtype) - self.t_update
            if self.use_constant_timestep:
                dt = torch.round(dt / self.dt) * self.dt
        else:
            dt = torch.zeros(x.size(0), dtype=self.dtype)
        assert torch.all(dt >= 0.0), "time stamps must be non-decreasing"

        if not (self.dt_pred.size() == dt.size() and torch.all(self.dt_pred == dt)):
            self.dt_pred = dt
            A, Q = self.state_space_model.get_discrete_time_matrices(
                self.dt_pred.view(-1, 1, 1, 1)
            )
            self.A_bar_pred = KroneckerProductLinearOperator(
                IdentityLinearOperator(self.num_inducing_points), A
            ).type(self.dtype)
            self.Q_bar_pred_chol = (
                KroneckerProductLinearOperator(self.K_ZZ_chol, psd_safe_cholesky(Q))
                .type(self.dtype)
                .to_dense()
            )

        # TODO: smart way to reuse previous predictions
        latent_inducing_mean_pred = self.A_bar_pred @ self.latent_inducing_mean
        latent_inducing_covar_pred_root = torch.cat(
            (self.A_bar_pred @ self.latent_inducing_covar_chol, self.Q_bar_pred_chol),
            dim=-1,
        )

        m_X = self.mean_module(x)
        K_XX = self.covar_module.kernels[0](x).type(self.dtype)
        K_XZ = (
            self.covar_module.kernels[0](x, self.inducing_points)
            .type(self.dtype)
            .to_dense()
        )

        interp_term = K_XZ @ self.K_ZZ_inv_chol
        output_map = (
            (interp_term @ self.latent_output_map).unsqueeze(0).transpose(0, -2)
        )
        Sigma = DiagLinearOperator(
            RootLinearOperator(output_map @ latent_inducing_covar_pred_root)
            .to_dense()
            .reshape(x.size(0), self.residual_dimension)
            .T
        )  # Diagonal approximation of the temporal covariance

        mean = (
            m_X
            + (output_map @ latent_inducing_mean_pred)
            .reshape(x.size(0), self.residual_dimension)
            .T
        )
        covar = K_XX - RootLinearOperator(interp_term) + Sigma

        return mean, covar

    def eval(self):
        if not self.is_initialized:
            self.initialize()

        super().eval()

    def __call__(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Make predictions with the GP model.

        Args:
            - x: Input data points. If temporal data is provided, it must be appended as the last column.
                Otherwise, the time stamp of data is assumed to be at the latest GP update.

        Returns:
            - MultitaskMultivariateNormal distribution
        """

        if x.size(1) == self.spatial_input_dimension + 1:
            t = x[..., -1]
            x = x[..., :-1]
        elif x.size(1) == self.spatial_input_dimension:
            t = None
        else:
            raise RuntimeError(
                f"Size of x {x.size()} does not correspond to spatial_input_dim {self.spatial_input_dimension}."
            )

        mean, covar = self.predict(x, t)

        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean, covar)
        )


class StateSpaceGPModel:
    """Representation of a temporal GP model defined by its covariance function as a
    continuous-time stochastic state-space model.

    Args:
        - covariance: Temporal covariance function. Currently only half-integer
        Matern kernels are supported (nu = 0.5, 1.5, 2.5).

    Attributes:
        - F: State transition matrix
        - L: Process noise scaling matrix
        - H: Observation matrix
        - P_inf: Steady-state covariance matrix
    """

    def __init__(
        self, covariance: gpytorch.kernels.MaternKernel
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(
            covariance, gpytorch.kernels.MaternKernel
        ) or covariance.nu not in [0.5, 1.5, 2.5]:
            raise RuntimeError(
                "Only Matern kernels with nu = [0.5, 1.5, 2.5] are currently supported."
            )

        timescale = torch.atleast_3d(covariance.lengthscale)
        nu = torch.tensor(covariance.nu)

        self.latent_dimension = int((nu + 0.5).item())
        self.residual_dimension = timescale.size(0)

        gamma = torch.sqrt(2.0 * nu) / timescale

        self.F = torch.diag_embed(
            torch.ones(
                self.residual_dimension,
                self.latent_dimension - 1,
                dtype=gamma.dtype,
            ),
            offset=1,
        )
        self.L = torch.zeros(
            self.residual_dimension, self.latent_dimension, 1, dtype=gamma.dtype
        )
        self.L[..., -1, 0] = torch.ones_like(timescale).view(self.residual_dimension)
        self.H = torch.zeros(
            self.residual_dimension, 1, self.latent_dimension, dtype=gamma.dtype
        )

        if nu == 0.5:
            self.F[..., -1, :] = -gamma.view(
                self.residual_dimension, self.latent_dimension
            )
            self.H[..., 0, 0] = torch.sqrt(2.0 * gamma).view(self.residual_dimension)

        elif nu == 1.5:
            self.F[..., -1, :] = torch.concatenate(
                (-(gamma**2), -2.0 * gamma), axis=-1
            ).view(self.residual_dimension, self.latent_dimension)
            self.H[..., 0, 0] = torch.sqrt(4.0 * gamma**3).view(self.residual_dimension)

        elif nu == 2.5:
            self.F[..., -1, :] = torch.cat(
                (-(gamma**3), -3.0 * gamma**2, -3.0 * gamma), axis=-1
            ).view(self.residual_dimension, self.latent_dimension)
            self.H[..., 0, 0] = torch.sqrt(16.0 / 3.0 * gamma**5).view(
                self.residual_dimension
            )

        P_inf_vec = torch.linalg.solve(
            torch.kron(torch.eye(self.latent_dimension, dtype=self.F.dtype), self.F)
            + torch.kron(self.F, torch.eye(self.latent_dimension, dtype=self.F.dtype)),
            -(self.L @ self.L.mT).reshape(self.residual_dimension, -1),
        )
        self.P_inf = P_inf_vec.reshape(
            self.residual_dimension, self.latent_dimension, self.latent_dimension
        )

    def get_discrete_time_matrices(
        self, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the discrete-time representation of the state-space model.

        Args:
            - dt: Time step size

        Returns:
            - A: Discrete-time state transition matrix
            - Q: Discrete-time process noise covariance matrix
        """

        A = torch.matrix_exp(self.F * dt).type(self.F.dtype)
        Q = self.P_inf.to_dense() - A @ self.P_inf.to_dense() @ A.mT

        return A, Q
