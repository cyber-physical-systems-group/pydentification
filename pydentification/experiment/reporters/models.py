# isort:skip_file
# skip file from sorting due to use of try/except import
from torch.nn import Module

try:
    import wandb
except ImportError as ex:
    message = (
        "Missing optional dependency wandb, to install all optionals from experiment module run:\n"
        "`pip install -r pydentification/experiment/extra-requirements.txt`"
    )

    raise ImportError(message) from ex


def count_trainable_parameters(module: Module) -> int:
    """Return the number of trainable parameters in neural model"""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def report_trainable_parameters(model: Module, prefix: str = "") -> None:
    """Report number of trainable parameters of any model to WANDB"""
    key = f"{prefix}/n_trainable_parameters" if bool(prefix) else "n_trainable_parameters"
    return wandb.log({key: count_trainable_parameters(model)})
