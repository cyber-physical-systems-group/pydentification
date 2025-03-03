import json
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from safetensors.torch import save_model

from pydentification.stubs import cast_to_path


def save_torch(path: Path, model: torch.nn.Module, method: Literal["pt", "safetensors"] = "safetensors"):
    if method == "safetensors":
        save_model(model, path)
    elif method == "pt":
        torch.save(model.state_dict(), path)  # saves only torch
    else:
        raise ValueError(f"Unknown method: {method}!")


def save_json(path: Path, data: dict):
    with path.open("w") as f:
        json.dump(data, f)  # type: ignore


@cast_to_path
def save_lightning(
    path: str | Path,
    model: pl.LightningModule,
    method: Literal["pt", "safetensors"] = "safetensors",
    save_hparams: bool = False,
):
    """
    :param path: name of the parent directory with the model and settings
    :param model: PyTorch model
    :param method: method of saving the model, either "pt" or "safetensors"
    :param save_hparams: whether to save hyperparameters in a JSON file
    """
    path.mkdir(parents=True, exist_ok=True)
    save_torch(path / f"trained-model.{method}", model=model.module, method=method)  # save only the model

    if save_hparams:
        save_json((path / "hparams.json"), model.hparams or {})
