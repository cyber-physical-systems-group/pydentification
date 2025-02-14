# isort: skip_file
import os

import lightning.pytorch as pl
import torch
import wandb

from pydentification.data.datamodules.simulation import SimulationDataModule
from pydentification.experiment.reporters import report_metrics, report_prediction_plot, report_trainable_parameters
from pydentification.metrics import regression_metrics
from pydentification.models.networks.transformer import DelayLineFeedforward, DynamicalSelfAttention, LinearProjection
from pydentification.training.lightning.simulation import LightningSimulationTrainingModule


def input_fn():
    return SimulationDataModule.from_csv(
        # dataset available at https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein
        dataset_path="data/WienerHammerBenchmark.csv",
        input_columns=["uBenchMark"],
        output_columns=["yBenchMark"],
        test_size=88_000,  # benchmark requirement
        validation_size=0.1,
        shift=1,
        n_workers=4,
        # inputs and outputs are controlled by model parameters
        forward_input_window_size=128,
        forward_output_window_size=128,
        forward_output_mask=127,  # forward_input_window_size - 1
    )


def model_fn():
    embedding = LinearProjection(
        n_input_time_steps=128,
        n_input_state_variables=1,
        n_output_time_steps=16,
        n_output_state_variables=8,
    )

    readout = LinearProjection(
        n_input_time_steps=16,  # must be the same as n_output_time_steps of last module
        n_input_state_variables=8,  # must be the same as n_output_state_variables of last module
        n_output_time_steps=1,  # must match the dataset
        n_output_state_variables=1,  # must be the same as system dimension
    )

    transformer = torch.nn.Sequential(
        embedding,
        # n_time_steps and n_state_variables must equal to n_output_time_steps of embedding
        DynamicalSelfAttention(n_time_steps=16, n_state_variables=8, n_heads=2, skip_connection=True),
        # shape must be the same as previous module
        DelayLineFeedforward(n_time_steps=16, n_state_variables=8, skip_connection=True),
        torch.nn.GELU(),
        DynamicalSelfAttention(n_time_steps=16, n_state_variables=8, n_heads=2, skip_connection=True),
        DelayLineFeedforward(n_time_steps=16, n_state_variables=8, skip_connection=True),
        torch.nn.GELU(),
        readout,
    )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True)

    timer = pl.callbacks.Timer(duration="00:04:00:00", interval="epoch")  # 4 hours
    stopping = pl.callbacks.EarlyStopping(monitor="trainer/validation_loss", patience=50, mode="min", verbose=True)

    path = f"models/{wandb.run.id}"
    epoch_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=path, monitor="trainer/validation_loss", every_n_epochs=10)

    model = LightningSimulationTrainingModule(transformer, optimizer, lr_scheduler, loss=torch.nn.MSELoss())

    trainer = pl.Trainer(  # CPU trainer
        max_epochs=10_000,
        precision=64,
        default_root_dir=path,
        callbacks=[timer, stopping, epoch_checkpoint],
        logger=pl.loggers.WandbLogger(project="test"),  # project name should be changed!
    )

    return model, trainer


def train_fn(model, trainer, dm):
    trainer.fit(model, datamodule=dm)
    return model, trainer


def report_fn(model, trainer, dm):
    y_hat = trainer.predict(model, datamodule=dm)
    y_pred = torch.cat(y_hat).numpy()
    y_true = torch.cat([y for _, y in dm.test_dataloader()]).numpy()

    metrics = regression_metrics(y_pred=y_pred.flatten(), y_true=y_true.flatten())

    report_metrics(metrics, prefix="test")
    report_trainable_parameters(model, prefix="config")
    report_prediction_plot(predictions=y_pred, targets=y_true, prefix="test")


def run_single_experiment():
    try:
        dm = input_fn()
        model, trainer = model_fn()
        model, trainer = train_fn(model, trainer, dm)
        report_fn(model, trainer, dm)
        # store trained model and send it to W&B
        path = f"models/{wandb.run.id}/trained-model.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
        wandb.save(path)
    except Exception as e:
        print(e)  # print traceback, since W&B uses multiprocessing, which can lose information about exception
        raise ValueError("Experiment failed.") from e


if __name__ == "__main__":
    # run just single model training and evaluation
    with wandb.init(project="test") as run:
        run_single_experiment()
