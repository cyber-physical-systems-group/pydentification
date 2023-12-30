import lightning.pytorch as pl
import torch

from pydentification.data.datamodules.simulation import SimulationDataModule
from pydentification.metrics import regression_metrics
from pydentification.models.modules.recurrect.gru import TimeSeriesGRU
from pydentification.training.lightning.simulation import LightningSimulationTrainingModule


def input_fn():
    return SimulationDataModule.from_csv(
        dataset_path="data/WienerHammerBenchmark.csv",
        input_columns=["uBenchMark"],
        output_columns=["yBenchMark"],
        test_size=88_000,  # benchmark requirement
        validation_size=0.1,
        shift=1,
        n_workers=4,
        forward_input_window_size=128,
        forward_output_window_size=128,
        forward_output_mask=127,  # forward_input_window_size - 1
    )


def model_fn():
    """Defines the model and optimizer, for the example model is not parameterized"""
    model = TimeSeriesGRU(
        n_input_state_variables=1,
        n_output_state_variables=1,
        n_hidden_state_variables=8,
        n_hidden_layers=2,
        predict_from_hidden_state=True,
    )

    optimizer = torch.optim.Adam(model.parameters())

    # this class wraps a model and adds training step and predict step functionality for autoregressive prediction
    training_module = LightningSimulationTrainingModule(module=model, optimizer=optimizer, lr_scheduler=None)

    dev_trainer = pl.Trainer(
        max_epochs=10,
        precision=64,  # use double precision (same as numpy dataset)
        # example callback for changing n_forward_time_steps
        reload_dataloaders_every_n_epochs=1,  # reload dataloaders to get new n_forward_time_steps
        limit_train_batches=5,  # limit number of batches for fast dev
        limit_val_batches=5,
    )

    return training_module, dev_trainer


def report_fn(trained_model, trainer, dm):
    y_hat = trainer.predict(trained_model, datamodule=dm)
    y_pred = torch.cat(y_hat).numpy()
    y_true = torch.cat([y for _, y in dm.test_dataloader()]).numpy()

    metrics = regression_metrics(y_pred=y_pred.flatten(), y_true=y_true.flatten())
    print(metrics)


if __name__ == "__main__":
    datamodule = input_fn()
    model, trainer = model_fn()
    trainer.fit(model, datamodule=datamodule)  # for W&B this can be extracted to function
    report_fn(model, trainer, datamodule)
