import math
from numbers import Number

import numpy as np
import pytest

from pydentification import metrics as pyidentification_metrics  # do not shadow sklearn metrics
from pydentification.metrics.regression import NumericSequence


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # exact 1D arrays return zero-error
        ([1, 2, 3], [1, 2, 3], "raw_values", 0.0),
        # different 1D arrays std = 0.5 and e = 0.5 so e / std = 1
        ([0, 1], [1, 1], "raw_values", 1.0),
        # exact float arrays
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "raw_values", 0.0),
        # exact 2D arrays with raw values return 2 values for each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "raw_values",
            np.asarray([0.0, 0.0]),
        ),
        # exact 2D arrays with uniform average return mean value of error in each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "uniform_average",
            0.0,
        ),
        # mixed 2D case
        (
            np.asarray([[1, 5], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1]]).T,
            "raw_values",
            np.asarray([0.0, 1.0]),
        ),
        # 3D case
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "raw_values",
            np.asarray([0.0, 1.0, 2.0]),
        ),
        # 3D case with uniform average
        # average error is 0.5 divided by the std of y_true variable
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "uniform_average",
            0.5 / np.std(np.asarray([1, 5, 0, 1, 0, 1])),
        ),
        # exact 2D arrays with uniform average return mean value of error in each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "dimension_average",
            0.0,
        ),
        # 3D case with uniform average
        # average error is 0.5 divided by the std of y_true variable
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "dimension_average",
            1.0,
        ),
    ],
)
def test_normalized_mean_squared_error(
    y_true: NumericSequence, y_pred: NumericSequence, multioutput: str, expected: Number
):
    error = pyidentification_metrics.normalized_mean_squared_error(y_true, y_pred, multioutput=multioutput)  # type: ignore
    np.testing.assert_array_equal(error, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # exact 1D arrays return zero-error
        ([1, 2, 3], [1, 2, 3], "raw_values", 0.0),
        # different 1D arrays std = 0.5 and e = 1/sqrt(2) so e / std = sqrt(2)
        ([0, 1], [1, 1], "raw_values", math.sqrt(2)),
        # exact float arrays
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "raw_values", 0.0),
        # exact 2D arrays with raw values return 2 values for each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "raw_values",
            np.asarray([0.0, 0.0]),
        ),
        # exact 2D arrays with uniform average return mean value of error in each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "uniform_average",
            0.0,
        ),
        # mixed 2D case
        (
            np.asarray([[1, 5], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1]]).T,
            "raw_values",
            np.asarray([0.0, math.sqrt(2)]),
        ),
        # 3D case
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "raw_values",
            np.asarray([0.0, math.sqrt(2), 2.0]),
        ),
        # 3D case with uniform average
        (
            np.asarray([[0, 1], [0, 1], [0, 1]]).T,
            np.asarray([[1, 1], [1, 1], [1, 1]]).T,
            "uniform_average",
            np.sqrt(2) / 2 / np.std(np.asarray([0, 1, 0, 1, 0, 1])),
        ),
        # exact 2D arrays with uniform average return mean value of error in each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "dimension_average",
            0.0,
        ),

        # 3D case with uniform average
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "dimension_average",
            # errors are 0, 2 and sqrt(2) for each dimension
            (2 + math.sqrt(2)) / 3
        ),
    ],
)
def test_normalized_root_mean_squared_error(
    y_true: NumericSequence, y_pred: NumericSequence, multioutput: str, expected: Number
):
    error = pyidentification_metrics.normalized_root_mean_squared_error(y_true, y_pred, multioutput=multioutput)  # type: ignore
    np.testing.assert_array_equal(error, expected)


@pytest.mark.parametrize(
    "y_true, y_pred, multioutput, expected",
    [
        # exact 1D arrays return zero-error
        ([1, 2, 3], [1, 2, 3], "raw_values", 0.0),
        # different 1D arrays std = 0.5 and e = 1/sqrt(2) so e / std = sqrt(2)
        ([0, 1], [1, 1], "raw_values", 1),
        # exact float arrays
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "raw_values", 0.0),
        # exact 2D arrays with raw values return 2 values for each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "raw_values",
            np.asarray([0.0, 0.0]),
        ),
        # exact 2D arrays with uniform average return mean value of error in each dimension
        (
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            np.asarray([[1, 2, 3], [1, 2, 3]]).T,
            "uniform_average",
            0.0,
        ),
        # mixed 2D case
        (
            np.asarray([[1, 5], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1]]).T,
            "raw_values",
            np.asarray([0.0, 1.0]),
        ),
        # 3D case
        (
            np.asarray([[1, 5], [0, 1], [0, 1]]).T,
            np.asarray([[1, 5], [1, 1], [1, 2]]).T,
            "raw_values",
            np.asarray([0.0, 1.0, 2.0]),
        ),
        # 3D case with uniform average
        (
            np.asarray([[0, 1], [0, 1], [0, 1]]).T,
            np.asarray([[1, 1], [1, 1], [1, 1]]).T,
            "uniform_average",
            0.5 / np.std(np.asarray([0, 1, 0, 1, 0, 1])),
        ),
    ],
)
def test_normalized_mean_absolute_error(
    y_true: NumericSequence, y_pred: NumericSequence, multioutput: str, expected: Number
):
    error = pyidentification_metrics.normalized_mean_absolute_error(y_true, y_pred, multioutput=multioutput)  # type: ignore
    np.testing.assert_array_equal(error, expected)


@pytest.mark.parametrize(
    ["metric_func"],
    (
        [pyidentification_metrics.normalized_mean_squared_error],
        [pyidentification_metrics.normalized_root_mean_squared_error],
        [pyidentification_metrics.normalized_mean_absolute_error],
        [pyidentification_metrics.regression_report],
        [pyidentification_metrics.regression_metrics],
    ),
)
def test_zero_std_is_nan(metric_func: callable):
    assert math.isnan(metric_func(y_true=[1, 1, 1], y_pred=[1, 1, 1]))
