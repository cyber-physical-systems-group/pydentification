import os
import shutil
from pathlib import Path

import wandb

from pydentification.stubs import cast_to_path


@cast_to_path
def save_to_wandb(path: str | Path, symlinks: bool = False):
    """
    Save all files from directory to W&B

    :param path: Path to the directory to save
    :param symlinks: If True, use symlinks instead of copying files (for Windows symlinks  can cause PermissionError)
    """
    if not symlinks:  # not needed, when using symlinks
        os.makedirs(Path(wandb.run.dir) / "outputs", exist_ok=True)

    for item in path.rglob("*"):
        if item.is_file():
            if symlinks:
                wandb.save(str(item))
            else:
                destination = Path(wandb.run.dir) / "outputs" / item.parts[-1]
                shutil.copy(item, destination)
                wandb.save(destination)
