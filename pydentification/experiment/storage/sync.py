from pathlib import Path

import wandb

from pydentification.stubs import cast_to_path


@cast_to_path
def save_to_wandb(path: str | Path):
    """Save all files from directory to W&B"""
    for file in path.rglob("*"):
        wandb.save(str(file))  # save file one by one
