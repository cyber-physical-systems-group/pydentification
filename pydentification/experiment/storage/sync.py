from pathlib import Path

import wandb


def save_to_wandb(path: str | Path):
    """Save all files from directory to W&B"""
    if isinstance(path, str):
        path = Path(path)

    for file in path.rglob("*"):
        wandb.save(str(file))  # save file one by one
