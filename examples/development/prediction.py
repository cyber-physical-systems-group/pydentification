import lightning.pytorch as pl
import pandas as pd
import torch

from pydentification.data.datamodules.prediction import PredictionDataModule
from pydentification.metrics import regression_metrics
from pydentification.models.networks.transformer import DelayLineFeedforward, DynamicalSelfAttention, LinearProjection
from pydentification.training.lightning.callbacks import StepAutoRegressionLengthScheduler
from pydentification.training.lightning.prediction import LightningPredictionTrainingModule


def input_fn():
    """Example uses the autonomous Lorenz system dataset"""
    data = pd.read_csv(r"data/lorenz.csv")  # example dataset with 3 variables and ~100k samples

    return PredictionDataModule(
        data[["x", "y", "z"]].values,
        test_size=0.5,
        batch_size=32,  # will be overwritten by dev setting of lightning
        validation_size=0.1,
        n_backward_time_steps=64,
        n_forward_time_steps=16,  # initial value
        n_workers=2,
    )


def model_fn():
    """Defines the model and optimizer, for the example model is not parameterized"""
    embedding = LinearProjection(
        n_input_time_steps=64,
        n_input_state_variables=3,
        n_output_time_steps=16,
        n_output_state_variables=8,
    )

    readout = LinearProjection(
        n_input_time_steps=16,  # must be the same as n_output_time_steps of last module
        n_input_state_variables=8,  # must be the same as n_output_state_variables of last module
        n_output_time_steps=1,  # must be 1 for autoregressive prediction
        n_output_state_variables=3,  # must be the same as system dimension
    )

    transformer = torch.nn.Sequential(
        embedding,
        # n_time_steps and n_state_variables must equal to n_output_time_steps of embedding
        DynamicalSelfAttention(n_time_steps=16, n_state_variables=8, n_heads=4, skip_connection=True),
        # shape must be the same as previous module
        DelayLineFeedforward(n_time_steps=16, n_state_variables=8, skip_connection=True),
        torch.nn.GELU(),
        DynamicalSelfAttention(n_time_steps=16, n_state_variables=8, n_heads=4, skip_connection=True),
        DelayLineFeedforward(n_time_steps=16, n_state_variables=8, skip_connection=True),
        torch.nn.GELU(),
        readout,
    )

    optimizer = torch.optim.Adam(transformer.parameters())

    # this class wraps a model and adds training step and predict step functionality for autoregressive prediction
    training_module = LightningPredictionTrainingModule(
        module=transformer, optimizer=optimizer, lr_scheduler=None, teacher_forcing=True, full_residual_connection=True
    )

    dev_trainer = pl.Trainer(
        max_epochs=10,
        precision=64,  # use double precision (same as numpy dataset)
        # example callback for changing n_forward_time_steps
        callbacks=[StepAutoRegressionLengthScheduler(step_size=5, gamma=2, verbose=True)],
        reload_dataloaders_every_n_epochs=1,  # reload dataloaders to get new n_forward_time_steps
        limit_train_batches=5,  # limit number of batches for fast dev
        limit_val_batches=5,
    )

    return training_module, dev_trainer


def report_fn(trained_model, trainer, dm):
    horizons = (16, 128, 1024)  # define a few autoregression horizon for evaluation
    # using dm.test_dataloader can return number of loaders with different n_forward_time_steps
    # LightningPredictionTrainingModule can handle them and return sorted by dataloader_idx
    y_hat = trainer.predict(trained_model, dataloaders=dm.test_dataloader(n_forward_time_steps=horizons))
    # cast to numpy, but preserve the order of horizons
    y_pred = [torch.cat(y).numpy() for y in y_hat]
    # iter over dataloader again to get true values for each horizon with aligned shape
    y_true = [torch.cat([y for _, y in loader]).numpy() for loader in dm.test_dataloader(n_forward_time_steps=horizons)]

    for n, y_hat_n, y_true_n in zip(horizons, y_pred, y_true):
        # compute metrics for each horizon
        metrics = regression_metrics(y_pred=y_hat_n.flatten(), y_true=y_true_n.flatten())  # type: ignore
        print(f"n_forward_time_steps={n}")
        print(metrics)


if __name__ == "__main__":
    datamodule = input_fn()
    model, trainer = model_fn()
    trainer.fit(model, datamodule=datamodule)  # for W&B this can be extracted to function
    report_fn(model, trainer, datamodule)
