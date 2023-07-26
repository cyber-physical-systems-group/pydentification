import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics


def _array_to_scalar(a: np.array) -> NDArray | float:
    """Convert array to scalar, but only if it contains singe item"""
    return a.item() if a.size == 1 else a


def _infer_axis(multioutput: Literal["raw_values", "uniform_average"], ndim: int) -> int | None:
    """Infers which axis to use for STD computation"""
    if multioutput == "uniform_average" and ndim == 1:
        warnings.warn("Attempting to use `uniform_average` with 1D array. This will have no effect.")

    return 0 if multioutput == "raw_values" and ndim == 2 else None


def nmse(y_true: NDArray, y_pred: NDArray, multioutput: Literal["raw_values", "uniform_average"] = "raw_values"):
    """Runs computation for NMSE"""
    axis = _infer_axis(multioutput, ndim=y_true.ndim)
    errors = metrics.mean_squared_error(y_true, y_pred, multioutput=multioutput)

    return _array_to_scalar(errors / np.std(y_true, axis=axis))


def nrmse(y_true: NDArray, y_pred: NDArray, multioutput: Literal["raw_values", "uniform_average"] = "raw_values"):
    """Runs computation for NRMSE"""
    axis = _infer_axis(multioutput, ndim=y_true.ndim)
    errors = metrics.mean_squared_error(y_true, y_pred, multioutput=multioutput, squared=False)

    return _array_to_scalar(errors / np.std(y_true, axis=axis))


def nmae(y_true: NDArray, y_pred: NDArray, multioutput: Literal["raw_values", "uniform_average"] = "raw_values"):
    """Runs computation for NRMSE"""
    axis = _infer_axis(multioutput, ndim=y_true.ndim)
    errors = metrics.mean_absolute_error(y_true, y_pred, multioutput=multioutput)

    return _array_to_scalar(errors / np.std(y_true, axis=axis))
