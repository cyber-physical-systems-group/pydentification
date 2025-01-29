import os
from typing import Literal

import lightning.pytorch as pl
import torch
import wandb
from safetensors.torch import save_model


def save_torch_module(name: str, model: pl.LightningModule, method: Literal["pt", "safetensors"] = "safetensors"):
    """Saves the torch model from given LightningModule to W&B"""
    path = f"models/{name}/trained-model.{method}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if method == "safetensors":
        save_model(model.module, path)
    elif method == "pt":
        torch.save(model.module.state_dict(), path)  # saves only torch
    else:
        raise ValueError(f"Unknown method: {method}!")

    wandb.save(path)
