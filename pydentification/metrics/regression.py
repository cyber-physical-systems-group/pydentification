import warnings
from numbers import Number
from typing import Literal, Sequence

import numpy as np
from sklearn import metrics
from sklearn.utils import check_array, check_consistent_length

from . import _compute

NumericSequence = Sequence[Number]


def validate_params(func: callable) -> callable:
    """
    Wrapper validating input parameters by checking array type type and length consistency
    Shorthand for sklearn `validate_param` with default values used in this module
    """

    def wrapper(y_true: NumericSequence, y_pred: NumericSequence, *args, **kwargs) -> Number | NumericSequence:
        y_true = check_array(y_true, ensure_2d=False)
        y_pred = check_array(y_pred, ensure_2d=False)
        check_consistent_length(y_true, y_pred)
        return func(y_true, y_pred, *args, **kwargs)

    return wrapper


def non_zero_std(func: callable) -> callable:
    """
    Wrapper checking if y_true does not have zero standard deviation.
    Returns NaN if yes and runs metric computation normally otherwise
    """

    def wrapper(y_true: NumericSequence, y_pred: NumericSequence, *args, **kwargs) -> Number | NumericSequence:
        if np.std(y_true) == 0:
            warnings.warn(
                "The standard deviation of the ground truth is zero."
                "The normalized mean squared error cannot be computed."
                "Returning nan instead."
            )
            return float("nan")
        return func(y_true, y_pred, *args, **kwargs)

    return wrapper


@validate_params
@non_zero_std
def normalized_mean_squared_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> Number:
    """
    Computes MSE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multioutput input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.

    :return: normalized mean squared error as single numeric value or array of values for multivariate case
    """
    return _compute.nmse(y_true=y_true, y_pred=y_pred, multioutput=multioutput)  # type: ignore


@validate_params
@non_zero_std
def normalized_root_mean_square_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> Number:
    """
    Computes RMSE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multioutput input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.

    :return: normalized root mean squared error as single numeric value or array of values for multivariate case
    """
    return _compute.nrmse(y_true=y_true, y_pred=y_pred, multioutput=multioutput)  # type: ignore


@validate_params
@non_zero_std
def normalized_mean_absolute_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
) -> Number:
    """
    Computes MAE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multioutput input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.

    :return: normalized mean absolute error as single numeric value or array of values for multivariate case
    """
    return _compute.nmae(y_true=y_true, y_pred=y_pred, multioutput=multioutput)  # type: ignore


@validate_params
@non_zero_std
def regression_metrics(y_true: NumericSequence, y_pred: NumericSequence) -> dict[str, float]:
    """
    Computes multiple regression scores and returns a dictionary with results.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values

    :warning: assumes `uniform_average` for multioutput
    """
    multioutput = "uniform_average"

    return {
        "MSE": metrics.mean_squared_error(y_true, y_pred, multioutput=multioutput),
        "RMSE": metrics.mean_squared_error(y_true, y_pred, squared=False, multioutput=multioutput),
        "MAE": metrics.mean_absolute_error(y_true, y_pred, multioutput=multioutput),
        "MAX": metrics.max_error(y_true, y_pred, multioutput=multioutput),
        # sequences are cast to arrays in `validate_params` wrapper
        "NMSE": _compute.nmse(y_true, y_pred, multioutput=multioutput),  # type: ignore
        "NRMSE": _compute.nrmse(y_true, y_pred, multioutput=multioutput),  # type: ignore
        "NMAE": _compute.nmae(y_true, y_pred, multioutput=multioutput),  # type: ignore
        # max error is dimension independent, so normalized max error it is just computed simply as max / std
        "NMAX": metrics.max_error(y_true.flatten(), y_pred.flatten()) / np.std(y_true, axis=0),  # type: ignore
        "R2": metrics.r2_score(y_true, y_pred),  # type: ignore
        "TRUE_MEAN": np.mean(y_true),
        "PRED_MEAN": np.mean(y_pred),
        "TRUE_STD": np.std(y_true),
        "PRED_STD": np.std(y_pred),
    }


@validate_params
@non_zero_std
def regression_report(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    *,
    precision: int = 4,
    width: int = 32,
    use_percentage: bool = False,
) -> str:
    """
    Returns detailed regression report as string similar to classification report from sklearn.

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model
    :param precision: keyword only, precision to which metrics are expressed, defaults to 4
    :param width: keyword only, spacing between columns, defaults to 32,
                  too small values might results in unreadable table
    :param use_percentage: if True print normalized values in %

    :return: regression metrics and data statistics as single string
    """
    computed_metrics = regression_metrics(y_true=y_true, y_pred=y_pred)  # type: ignore
    precision_marker = "%" if use_percentage else "f"

    report = ""
    # add report header
    report += f"{'':<{width}}{'Absolute':<{width}}{'Normalized':<{width}}\n"
    # add metrics
    report += (
        f"{'Mean Squared Error:':<{width}}"
        f"{computed_metrics['MSE']:<{width}.{precision}f}"
        f"{computed_metrics['NMSE']:<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Root Mean Squared Error:':<{width}}"
        f"{computed_metrics['RMSE']:<{width}.{precision}f}"
        f"{computed_metrics['NRMSE']:<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Mean Absolute Error:':<{width}}"
        f"{computed_metrics['MAE']:<{width}.{precision}f}"
        f"{computed_metrics['NMAE']:<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Max Error:':<{width}}"
        f"{computed_metrics['MAX']:<{width}.{precision}f}"
        f"{computed_metrics['NMAX']:<{width}.{precision}{precision_marker}}\n"
    )
    # add single column for R2 score
    report += f"{'R2':<{width}}{'':<{width}}{computed_metrics['R2']:<{width}.{precision}f}\n"
    report += "\n\n"
    # add empty row for some space
    report += f"{'':<{width}}{'True':<{width}}{'Predicted':<{width}}\n"
    # add true and predicted distribution statistics
    report += f"{'Mean:':<{width}}{computed_metrics['TRUE_MEAN']:<{width}.{precision}f}"
    report += f"{computed_metrics['PRED_MEAN']:<{width}.{precision}f}\n"
    report += f"{'std:':<{width}}{computed_metrics['TRUE_STD']:<{width}.{precision}f}"
    report += f"{computed_metrics['PRED_STD']:<{width}.{precision}f}\n"

    return report
