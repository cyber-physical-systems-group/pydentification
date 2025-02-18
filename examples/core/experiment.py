# isort: skip_file
import os

import lightning.pytorch as pl
import pandas as pd
import torch

import wandb

from pydentification.experiment.storage.wrapper import dump
from pydentification.experiment.storage.models import save_lightning
from pydentification.experiment.storage.code import save_code_snapshot
from pydentification.experiment.storage.sync import save_to_wandb
from pydentification.training.lightning.prediction import LightningPredictionTrainingModule
from pydentification.data.datamodules.prediction import PredictionDataModule


def input_fn(parameters: dict):
    data = pd.read_csv("data/lorenz.csv")  # assume dataset exists and has ~100 000 samples with 3 columns: x, y, z
    return PredictionDataModule(
        data[["x", "y", "z"]].values,
        test_size=30_000,  # 30% assuming 100 000 sample
        batch_size=32,
        validation_size=0.1,  # 10% of the training set, which is 70% of the whole dataset
        n_backward_time_steps=parameters["n_input_time_steps"],  # sweep parameter, which can be changed between runs
        n_forward_time_steps=parameters["n_output_time_steps"],
        n_workers=4,
    )


# pass parameterless lambda function dynamically returning path to the decorator, after W&B run is initialized
@dump(path=lambda: f"outputs/{wandb.run.id}", param_store="both")  # noqa
def model_fn(hidden_dim: int = 64):
    return torch.nn.Sequential(
        torch.nn.Linear(3, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 3),
    )


def trainer_fn(model, parameters: dict):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    timer = pl.callbacks.Timer(duration="00:04:00:00", interval="epoch")
    model = LightningPredictionTrainingModule(module=model, optimizer=optimizer)

    trainer = pl.Trainer(
        max_epochs=3,  # just an example
        precision=64,
        accelerator="cpu",
        devices=1,
        callbacks=[timer],
    )

    return model, trainer


def train_fn(model, trainer, dm):
    """Runs training using lightning trainer and given datamodule"""
    trainer.fit(model, datamodule=dm)
    return model, trainer


def run_single_experiment():
    with wandb.init(reinit=True):
        # prepare directories and library code snapshot
        os.makedirs(f"outputs/{wandb.run.id}/models", exist_ok=True)
        os.makedirs(f"outputs/{wandb.run.id}/code", exist_ok=True)
        save_code_snapshot(name="code", source_dir="pydentification", target_dir=f"outputs/{wandb.run.id}/code")

        parameters = dict(wandb.config)  # cast to dict is needed to serialize the parameters
        try:
            print(f"Starting experiment with {wandb.run.id}")
            dm = input_fn(parameters)
            model = model_fn()
            model, trainer = trainer_fn(model, parameters)
            model, trainer = train_fn(model, trainer, dm)

            # store trained model and send it to W&B
            save_lightning(f"outputs/{wandb.run.id}/models", model=model, method="safetensors", save_hparams=True)
            save_to_wandb(f"outputs/{wandb.run.id}")  # save all files in the directory to W&B
        except Exception as e:
            print(e)  # print traceback, since W&B uses multiprocessing, which can lose information about exception
            raise ValueError("Experiment failed.") from e


if __name__ == "__main__":
    sweep_id = wandb.sweep({"hidden_di": [32, 64, 128]}, project="test")
    wandb.agent(sweep_id, function=run_single_experiment, count=3, project="test")
