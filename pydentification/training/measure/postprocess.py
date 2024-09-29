import warnings

import numpy as np
import pandas as pd
from torch import Tensor

try:
    from lovely_tensors import lovely
except ImportError:
    warnings.warn(
        "Missing optional dependency 'lovely-tensors'."
        "Run `pip install lovely-tensors` or `pip install pydentification[experiment] to use `LoggingMeasureStorage`"
    )


def make_tensor_lovely(tensor: Tensor) -> str:
    """Function for making tensor lovely."""
    return str(lovely(tensor))


def tensor_describe(tensor: Tensor, precision: int = 2) -> dict[str, float]:
    """Function for describing tensor."""
    return pd.Series(tensor.numpy()).describe().apply(lambda value: round(value, precision)).to_dict()


def complex_tensor_describe(tensor: Tensor) -> dict[str, float]:
    """Similar to pd.Series.describe() but for complex-valued tensors."""
    values = tensor.numpy()
    return {
        "count": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "median": np.median(values),
        "min": np.min(values),
        "max": np.max(values),
    }
