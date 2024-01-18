from typing import Literal

import torch
from torch import Tensor

from pydentification.data import decay
from pydentification.models.nonparametric import functional as nonparametric_functional
from pydentification.models.nonparametric.estimators import noise_variance as noise_variance_estimator
from pydentification.models.nonparametric.kernels import KernelCallable


class KernelRegression:
    def __init__(
        self,
        memory_manager,
        bandwidth: float,
        kernel: KernelCallable,
        lipschitz_constant: float,
        delta: float = 0.1,
        noise_variance: float | Literal["estimate"] = "estimate",
        k: int = 10,
        memory_epsilon: float = 0.1,
        decay: float = 0.0,
        p: int = 2,
        noise_var_kernel_size: int = 5,  # only used when noise_var is "estimate"
    ):
        """
        :param memory_manager: memory manager class, which will be build in `adapt` and used to access training data
        :param bandwidth: bandwidth of the kernel of the kernel regression
        :param kernel: kernel function used for kernel regression, see `pydentification.models.nonparametric.kernels`
        :param lipschitz_constant: Lipschitz constant of the function to be estimated, needs to be known
        :param delta: confidence level, defaults to 0.1
        :param noise_variance: variance of the noise in the function to be estimated, defaults to "estimate"
        :param k: number of nearest neighbors to use for kernel regression, defaults to 10
        :param memory_epsilon: epsilon parameter for memory manager, defaults to 0.1
        :param decay: decay parameter for exponential decay of inputs, defaults to 0.0
        :param p: exponent for point-wise distance, defaults to 2
        :param noise_var_kernel_size: kernel size for noise variance estimator, defaults to 5
        """
        self.memory_manager = memory_manager
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lipschitz_constant = lipschitz_constant
        self.delta = delta
        self.noise_variance = noise_variance
        self.k = k
        self.memory_epsilon = memory_epsilon
        self.decay = decay
        self.p = p
        self.noise_var_kernel_size = noise_var_kernel_size

        self.adapted: bool = False  # flag to check if model has been adapted to training data

    @torch.no_grad()
    def adapt(self, x: Tensor, y: Tensor) -> None:
        """
        This method prepares the non-parametric function for usage. Kernel regression can only be used for SISO systems
        with one-step ahead prediction, which forces the inputs to have required shapes (given below).

        :note: in adapt BATCH can be size of entire dataset

        :param x: input data with shape (BATCH, TIME_STEPS, SYSTEM_DIM), where SYSTEM_DIM needs to be 1
        :param y: target data with shape (BATCH, 1, 1)
        """
        if x.size(-1) != 1 or y.size(-1) != 1 or y.size(1) != 1:
            raise RuntimeError("Kernel regression can only be used for SISO systems with one-step ahead prediction!")

        x = x.squeeze(dim=-1)  # (BATCH, TIME_STEPS, SYSTEM_DIM) -> (BATCH, TIME_STEPS) for SISO systems
        y = y.squeeze(dim=-1)  # (BATCH, TIME_STEPS, 1) -> (BATCH, ) for SISO systems

        if self.decay > 0.0:  # exponential decay can be applied to the inputs to prioritize recent data in time-series
            x = decay(x, gamma=self.decay)
        if self.noise_variance == "estimate":  # estimate noise variance if its value is not given
            # only 1D signal is supported for noise variance estimation, so y is squeezed to (BATCH,)
            self.noise_variance = noise_variance_estimator(y.squeeze(dim=-1), kernel_size=self.noise_var_kernel_size)

        # create memory manager to access training data and prevent high memory usage
        # and build index for nearest neighbors search during adapt to save time later
        self.memory_manager = self.memory_manager(x, y)  # type: ignore
        self.memory_manager.prepare()  # type: ignore
        self.adapted = True

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """
        Predicts the function value at given input points using kernel regression with fixed settings given
        during object creation. Shape interface is the same as for models used in `pydentification.models` package.
        """
        if not self.adapted:
            raise RuntimeError("Model needs to be adapted to training data before it can be used for prediction!")

        if x.size(-1) != 1:
            raise RuntimeError("Kernel regression can only be used for SISO systems with one-step ahead prediction!")

        x = self.exponential_decay(x.squeeze(dim=-1))
        x_from_memory, y_from_memory = self.memory_manager.query_nearest(x, k=self.k, epsilon=self.memory_epsilon)

        predictions, kernels = nonparametric_functional.kernel_regression(
            memory=x_from_memory,
            targets=y_from_memory.squeeze(dim=-1),  # (BATCH, TIME_STEPS) -> (BATCH, )
            inputs=x,
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            p=self.p,
            return_kernel_density=True,  # always return kernel density for bounds, hybrid trainer requires it
        )

        bounds = nonparametric_functional.kernel_regression_bounds(
            kernels=kernels,
            bandwidth=self.bandwidth,
            delta=self.delta,
            lipschitz_constant=self.lipschitz_constant,
            noise_variance=self.noise_variance,
            dim=1,  # always 1 for SISO dynamical systems
        )

        return predictions, bounds
