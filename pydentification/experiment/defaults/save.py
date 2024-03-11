import os

import torch
import wandb


def save_fn(name: str, model: torch.nn.Module):
    path = f"models/{name}/trained-model.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)
    wandb.save(path)
