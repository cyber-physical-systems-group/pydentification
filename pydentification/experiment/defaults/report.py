import lightning.pytorch as pl
import torch

from pydentification.experiment.reporters import report_metrics, report_prediction_plot, report_trainable_parameters
from pydentification.metrics import regression_metrics


def report_fn(model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule):
    """Logs the experiment results to W&B"""
    y_hat = trainer.predict(model, datamodule=dm)
    y_pred = torch.cat(y_hat).numpy()
    y_true = torch.cat([y for _, y in dm.test_dataloader()]).numpy()

    metrics = regression_metrics(y_pred=y_pred.flatten(), y_true=y_true.flatten())  # type: ignore

    report_metrics(metrics, prefix="test")  # type: ignore
    report_trainable_parameters(model, prefix="config")
    report_prediction_plot(predictions=y_pred, targets=y_true, prefix="test")
