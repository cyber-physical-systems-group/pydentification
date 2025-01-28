import os

import lightning.pytorch as pl
import torch
import wandb


def save_torch_module(name: str, model: pl.LightningModule):
    """Saves the torch model from given LightningModule to W&B"""
    path = f"models/{name}/trained-model.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(model.module.state_dict(), path)  # saves only torch
    wandb.save(path)
