# isort: skip_file
import os
from datetime import timedelta

import lightning.pytorch as pl
import pandas as pd
import torch
import wandb

from pydentification.data.datamodules.prediction import PredictionDataModule
from pydentification.experiment.reporters import report_metrics, report_prediction_plot, report_trainable_parameters
from pydentification.metrics import regression_metrics
from pydentification.models.transformer import CausalDelayLineFeedforward, DynamicalSelfAttention, LinearProjection
from pydentification.training.lightning.prediction import LightningPredictionTrainingModule


def input_fn(parameters: dict):
    df = pd.read_csv("dataset.csv")  # assume dataset exists and has ~100 000 samples with 3 columns: x, y, z

    return PredictionDataModule(
        df[["x", "y", "z"]],
        test_size=30_000,  # 30% assuming 100 000 sample
        batch_size=32,
        validation_size=0.1,  # 10% of the training set, which is 70% of the whole dataset
        n_backward_time_steps=parameters["n_input_time_steps"],  # sweep parameter, which can be changed between runs
        n_forward_time_steps=parameters["n_output_time_steps"],
        n_workers=4,
    )


def model_fn(parameters: dict):
    """Define the model based on sweep parameters"""
    layers = []

    embedding = LinearProjection(
        n_input_time_steps=parameters["n_input_time_steps"],
        n_input_state_variables=parameters["n_input_state_variables"],
        n_output_time_steps=parameters["n_hidden_time_steps"],
        n_output_state_variables=parameters["n_hidden_state_variables"],
    )

    layers.append(embedding)

    for _ in range(parameters["n_layers"]):  # loop and create transformer blocks
        layers.append(torch.nn.LayerNorm([parameters["n_hidden_time_steps"], parameters["n_hidden_state_variables"]]))
        layers.append(
            DynamicalSelfAttention(
                n_time_steps=parameters["n_hidden_time_steps"],
                n_state_variables=parameters["n_hidden_state_variables"],
                n_heads=parameters["nheads"],
                skip_connection=parameters["sa_skip_connection"],
                is_causal=True,
            )
        )

        layers.append(
            CausalDelayLineFeedforward(
                n_time_steps=parameters["n_hidden_time_steps"],
                n_state_variables=parameters["n_hidden_state_variables"],
                skip_connection=parameters["ff_skip_connection"],
            )
        )

        layers.append(torch.nn.GELU())

    readout = LinearProjection(
        n_input_time_steps=parameters["n_hidden_time_steps"],
        n_input_state_variables=parameters["n_hidden_state_variables"],
        n_output_time_steps=parameters["n_output_time_steps"],
        n_output_state_variables=parameters["n_output_state_variables"],
    )
    layers.append(readout)

    return torch.nn.Sequential(*layers)


def trainer_fn(model, parameters: dict):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True)
    # callbacks for stopping the training early, with 4 hour timeout and patience of 50 epochs (with 20 for reducing LR)
    timer = pl.callbacks.Timer(duration="00:04:00:00", interval="epoch")
    stopping = pl.callbacks.EarlyStopping(monitor="training/validation_loss", patience=50, mode="min", verbose=True)
    # checkpointing the model every 100 epochs and every hour to single directory
    path = f"models/{wandb.run.id}"
    epoch_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=path, monitor="validation/loss", every_n_epochs=100)
    time_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=path, train_time_interval=timedelta(hours=1))

    # wrap model in training class with auto-regression training defined
    model = LightningPredictionTrainingModule(
        module=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        teacher_forcing=parameters["teacher_forcing"],  # sweep can toggle between forcing on or off
        full_residual_connection=parameters["full_residual_connection"],
    )

    trainer = pl.Trainer(
        max_epochs=10000,  # max limit, will most likely be stopped early by timeout or early stopping
        precision=64,
        accelerator="gpu",
        devices=1,
        default_root_dir=path,  # for logging and checkpointing
        callbacks=[timer, stopping, epoch_checkpoint, time_checkpoint],
    )

    return model, trainer


def train_fn(model, trainer, dm):
    """Runs training using lightning trainer and given datamodule"""
    trainer.fit(model, datamodule=dm)
    return model, trainer


def report_fn(model, dm, auto_regression_scales):
    report_trainable_parameters(model, prefix="config")  # save number of trainable parameters to W&B

    # create CPU trainer for testing to avoid distributed computing
    trainer_for_testing = pl.Trainer(precision=64)
    # create multiple dataloaders for different auto-regression lengths
    dataloaders = dm.test_dataloader(n_forward_time_steps=auto_regression_scales)

    # returns list of predictions for each dataloader as list
    y_hat = trainer_for_testing.predict(model, dataloaders=dataloaders)
    y_pred = [torch.cat(y).numpy() for y in y_hat]  # concatenate predictions into single array for each scale
    y_true = [torch.cat([y for _, y in loader]).numpy() for loader in dataloaders]

    # report metrics and plots for each auto-regression scale separately
    for n, y_hat_n, y_true_n in zip(auto_regression_scales, y_pred, y_true):
        metrics = regression_metrics(y_pred=y_hat_n.flatten(), y_true=y_true_n.flatten())
        report_metrics(metrics, prefix=f"test/{n}_time_steps")  # prefix with number of steps for each metric
        report_prediction_plot(predictions=y_hat_n, targets=y_true_n, prefix=f"test/{n}_time_steps")


def run_single_experiment():
    with wandb.init(reinit=True):
        parameters = wandb.config
        try:
            dm = input_fn(parameters)
            model = model_fn(parameters)
            model, trainer = trainer_fn(model, parameters)
            model, trainer = train_fn(model, trainer, dm)
            report_fn(model, dm, auto_regression_scales=[16, 32, 128])  # sample of regression scales
            # store trained model and send it to W&B
            os.makedirs(f"models/{wandb.run.id}", exist_ok=True)
            path = f"models/{wandb.run.id}/trained-model.pt"
            torch.save(model, path)
            wandb.save(path)
        except Exception as e:
            print(e)  # print traceback, since W&B uses multiprocessing, which can loose information about exception
            raise ValueError("Experiment failed.") from e


# this dict defined sweep configuration, which is used to create model and datamodule
SWEEP_CONFIG = {
    "name": "transformer-sweep-for-prediction",
    "method": "grid",
    "metric": {
        "name": "NRMSE",
        "goal": "minimize",
    },
    "parameters": {
        "batch_size": {
            "values": [32],
        },
        "shift": {
            "values": [1],
        },
        "model_name": {"values": ["Transformer"]},
        # number of input and output states is controlled by dataset, but the number of input time-steps can be changed
        # (longer history gives more information, but results in slower training)
        "n_input_time_steps": {"values": [32, 64, 128, 256, 512, 1024]},
        "n_output_time_steps": {"values": [1]},  # step-ahead prediction, auto-regression is defined in trainer class
        "n_input_state_variables": {"values": [3]},
        "n_output_state_variables": {"values": [3]},
        # model
        "nheads": {"values": [1, 2, 4, 8]},
        "n_hidden_time_steps": {"values": [8, 16, 32, 64]},
        "n_hidden_state_variables": {
            "values": [4, 8, 16, 32],
        },
        "n_layers": {
            "values": [1, 2, 3, 4],
        },
        "sa_skip_connection": {
            # only single value is allowed, but parameters is defined in sweep
            # so it is logged properly and can be tweaked in future
            "values": [True],
        },
        "ff_skip_connection": {
            "values": [True],
        },
        "teacher_forcing": {"values": [True, False]},
        "full_residual_connection": {"values": [True, False]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="test")  # change project name
    wandb.agent(sweep_id, function=run_single_experiment, count=10, project="test")
