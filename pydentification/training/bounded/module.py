from typing import Any, Literal
from warnings import warn

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from pydentification.data.process import unbatch
from pydentification.models.modules.activations import bounded_linear_unit
from pydentification.models.modules.losses import BoundedMSELoss
from pydentification.models.nonparametric import functional as nonparametric_functional
from pydentification.models.nonparametric.estimators import noise_variance as noise_variance_estimator
from pydentification.models.nonparametric.kernels import KernelCallable


class HybridBoundedSimulationTrainingModule(pl.LightningModule):
    """
    This class contains training module for neural network to identify nonlinear dynamical systems or static nonlinear
    functions with guarantees by using bounded activation incorporating theoretical bounds from the kernel regression
    estimator. The approach is limited to finite memory single-input single-output dynamical systems,
    which can be converted to static multiple-input single-output systems by using delay-line.

    Bounds are computed using kernel regression working with the same data, but we are able to guarantees of the
    estimation, which are used to activate a network during and after training, in order to ensure that the predictions
    are never outside of those theoretical bounds.

    Bounds can be also used as penalty during training, which is implemented in this class using `BoundedMSELoss`.

    This module contains non-trivial torch device settings, since memory manger and network are independent.
    `memory_device` contains device used for memory manager, which is used to access training data and for
    non-parametric estimator and `predict_device` contains device used for inference, which is used for network and
    memory, which can be CPU or GPU (if memory manager supports it).
    """

    def __init__(
        self,
        network: Module,
        optimizer: torch.optim.Optimizer,
        memory_manager,
        kernel: KernelCallable,
        bandwidth: float,
        lipschitz_constant: float,
        delta: float,
        noise_variance: float | Literal["estimate"] = "estimate",
        k: int | None = None,
        r: float | None = None,
        memory_epsilon: float = 0.1,
        p: int = 2,
        noise_var_kernel_size: int = 5,  # only used when noise_var is "estimate"
        bound_during_training: bool = False,
        bound_crossing_penalty: float = 0.0,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        memory_device: Literal["cpu", "cuda"] = "cpu",
        predict_device: Literal["cpu", "cuda"] = "cpu",
    ):
        """
        :param network: initialized neural network to be wrapped by HybridBoundedSimulationTrainingModule
        :param optimizer: initialized optimizer to be used for training
        :param memory_manager: memory manager class, which will be build in `adapt` and used to access training data
        :param bandwidth: bandwidth of the kernel of the kernel regression
        :param kernel: kernel function used for kernel regression, see `pydentification.models.nonparametric.kernels`
        :param lipschitz_constant: Lipschitz constant of the function to be estimated, needs to be known
        :param delta: confidence level, defaults to 0.1
        :param noise_variance: variance of the noise in the function to be estimated, defaults to "estimate"
        :param k: number of nearest neighbors to use for kernel regression, either k or r needs to be defined
        :param r: radius of the neighborhood to use for kernel regression, either k or r needs to be defined
        :param memory_epsilon: epsilon parameter for memory manager, defaults to 0.1
        :param p: exponent for point-wise distance, defaults to 2
        :param noise_var_kernel_size: kernel size for noise variance estimator, defaults to 5
        :param bound_during_training: flag to enable bounding during training, defaults to False
        :param bound_crossing_penalty: penalty factor for crossing bounds, see: BoundedMSELoss, defaults to 0.0
        :param lr_scheduler: initialized learning rate scheduler to be used for training, defaults to None
        :param memory_device: device to use for memory manager, defaults to "cpu"
                              currently only single device strategy is supported, due to memory manager limitations
        :param predict_device: device to use for inference, needs to be supported by memory manager, defaults to "cpu"
        """
        super().__init__()
        # neural network training properties
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bound_during_training = bound_during_training
        self.loss = BoundedMSELoss(gamma=bound_crossing_penalty)
        # non-parametric estimator properties
        self.memory_manager = memory_manager
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lipschitz_constant = lipschitz_constant
        self.delta = delta
        self.noise_variance = noise_variance
        self.k = k
        self.r = r
        self.memory_epsilon = memory_epsilon
        self.p = p
        self.noise_var_kernel_size = noise_var_kernel_size

        # dtype and device properties
        self.memory_device = memory_device
        self.predict_device = predict_device
        self.prepared: bool = False

        # validations
        if k is None and r is None:
            raise ValueError("Either k or radius needs to be defined!")

        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, trained_network: Module, **kwargs):
        """
        Shortcut for using module with pretrained network. Calling this method is equivalent to passing the trained
        network directly to `__init__`, but the classmethod can be useful for stating the user intention.
        """
        return cls(network=trained_network, **kwargs)

    def setup(self, stage: Literal["fit", "predict"]) -> None:
        """
        This method is called by lightning to set up the model before training or prediction.
        It is used to prepare memory manager for nonparametric estimator with training data.
        """
        if not self.prepared:
            # prepare is called to store training data in memory manager for nonparametric estimator
            # it always needs to see only training data, even if the model will be called on validation
            dataloader = self.trainer.datamodule.train_dataloader()
            memory, targets = unbatch(dataloader)
            self.prepare(memory, targets)
            self.prepared = True

        if stage == "predict":
            # by default network is on CPU after training
            # it can be moved to CUDA if needed
            self.network.to(torch.device(self.predict_device))
            # move memory manager to device used for inference
            # only if it supports it and used it during training
            if self.predict_device == self.memory_device:
                self.memory_manager.to(torch.device(self.predict_device))

    @torch.no_grad()
    def prepare(self, x: Tensor, y: Tensor):
        """
        Prepare memory manager for nonparametric estimator with training data. This method is called by `setup` method
        automatically, but it can be also called manually to prepare memory manager with custom data.
        """
        if x.size(-1) != 1 or y.size(-1) != 1 or y.size(1) != 1:
            raise RuntimeError("Kernel regression can only be used for SISO systems with one-step ahead prediction!")

        x = x.squeeze(dim=-1)  # (BATCH, TIME_STEPS, SYSTEM_DIM) -> (BATCH, TIME_STEPS) for SISO systems
        y = y.squeeze(dim=-1)  # (BATCH, TIME_STEPS, 1) -> (BATCH, ) for SISO systems

        if self.noise_variance == "estimate":  # estimate noise variance if its value is not given
            # only 1D signal is supported for noise variance estimation, so y is squeezed to (BATCH,)
            self.noise_variance = noise_variance_estimator(y.squeeze(dim=-1), kernel_size=self.noise_var_kernel_size)

        if type(self.trainer.strategy).__name__ != "SingleDeviceStrategy":
            raise RuntimeError(f"{self.__class__.__name__} can only be used with single device strategy!")

        if self.memory_device == "cuda":
            # by default setup (which calls prepare) is running on CPU before tensors are moved to devices
            # some memory managers can handle GPU operations, but this needs moving entire dataset to device
            x = x.cuda()
            y = y.cuda()

        # create memory manager to access training data and prevent high memory usage
        # and build index for nearest neighbors search during adapt to save time later
        self.memory_manager.prepare(x, y)  # type: ignore

    @torch.no_grad()
    def nonparametric_forward(self, x: Tensor) -> Tensor:
        """
        Part of forward function to predict value at given input points using kernel regression with fixed settings.
        Shape interface is the same as for models used in `pydentification.models` package.
        """
        if x.size(-1) != 1:
            raise RuntimeError("Kernel regression can only be used for SISO systems with one-step ahead prediction!")

        x = x.squeeze(dim=-1)  # (BATCH, TIME_STEPS, SYSTEM_DIM) -> (BATCH, TIME_STEPS) for SISO systems
        x_from_memory, y_from_memory = self.memory_manager(x, k=self.k, r=self.r, epsilon=self.memory_epsilon)

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

        # run bounds extrapolation to ensure that bounds are always positive
        # bounds are diverging linearly with lipschitz constant from known points
        bounds = nonparametric_functional.extrapolate_bounds(x, bounds, self.lipschitz_constant, p=self.p)
        return predictions, bounds

    def forward(self, x: Tensor, return_nonparametric: bool = False) -> Tensor:
        nonparametric_predictions, bounds = self.nonparametric_forward(x)
        # bounds are returned as distance from nonparametric predictions
        upper_bound = nonparametric_predictions + bounds
        lower_bound = nonparametric_predictions - bounds

        if torch.isnan(nonparametric_predictions).all():
            warn("Nonparametric predictions contain only NaN values! Increase the bandwidth parameter or batch size!")
            raise RuntimeError("Nonparametric predictions contain only NaN values!")

        predictions = self.network(x)

        if self.bound_during_training:
            predictions = bounded_linear_unit(predictions, lower=lower_bound, upper=upper_bound)

        if return_nonparametric:
            return predictions, nonparametric_predictions, lower_bound, upper_bound

        return predictions, lower_bound, upper_bound

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat, lower_bound, upper_bound = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y, lower_bound, upper_bound)  # type: ignore
        self.log("training/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        # bounds are not used for validation
        y_hat, _, _ = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y)  # type: ignore
        self.log("training/validation_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def configure_optimizers(self) -> dict[str, Any]:
        config = {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "training/validation_loss"}
        return {key: value for key, value in config.items() if value is not None}  # remove None values

    @torch.no_grad()
    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Tensor]:
        """
        Returns network and nonparametric estimator predictions and bounds for given batch.
        Outputs are returned as dictionary, so that they can be easily logged to W&B.

        :note: batch needs to be on `predict_device`
        """
        x, _ = batch  # type: ignore
        predictions, nonparametric_predictions, lower_bound, upper_bound = self.forward(x, return_nonparametric=True)

        return {
            "network_predictions": predictions,
            "nonparametric_predictions": nonparametric_predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def predict_dataloader(self, dataloader: DataLoader) -> dict[str, Tensor]:
        """
        Returns network and nonparametric estimator predictions and bounds for given dataloader.
        Outputs are returned as dictionary, so that they can be easily logged to W&B.
        """
        outputs = []

        for batch_idx, batch in enumerate(dataloader):
            if self.predict_device == "cuda":
                batch = tuple(tensor.cuda() for tensor in batch)

            outputs.append(self.predict_step(batch, batch_idx=batch_idx))

        return {
            "network_predictions": torch.cat([output["network_predictions"] for output in outputs]),
            "nonparametric_predictions": torch.cat([output["nonparametric_predictions"] for output in outputs]),
            "lower_bound": torch.cat([output["lower_bound"] for output in outputs]),
            "upper_bound": torch.cat([output["upper_bound"] for output in outputs]),
        }

    def predict_datamodule(self, dm: pl.LightningDataModule, with_targets: bool = False) -> dict[str, Tensor]:
        """
        Runs predict dataloader on test_dataloader or given datamodule. Makes sure data module is set up properly.

        :param dm: lightning data module to run predict on, uses only `test_dataloader`
        :param with_targets: if True targets are appended as concatenated Tensor to predictions
        """
        dm.setup(stage="predict")
        predictions = self.predict_dataloader(dm.test_dataloader())

        if with_targets:
            predictions["targets"] = torch.cat([y for x, y in dm.test_dataloader()])

        return predictions
