from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
import wandb
from safetensors.torch import save_model


def save_torch(path: Path, model: torch.nn.Module, method: Literal["pt", "safetensors"] = "safetensors"):
    if method == "safetensors":
        save_model(model, path)
    elif method == "pt":
        torch.save(model.state_dict(), path)  # saves only torch
    else:
        raise ValueError(f"Unknown method: {method}!")


def save_fn(name: str, model: pl.LightningModule, method: Literal["pt", "safetensors"] = "safetensors"):
    """Saves the torch model from given LightningModule to W&B"""
    path = Path(f"models/{name}/trained-model.{method}")
    path.parent.mkdir(parents=True, exist_ok=True)

    save_torch(path, model=model.module, method=method)  # save only the model
    wandb.save(path)
