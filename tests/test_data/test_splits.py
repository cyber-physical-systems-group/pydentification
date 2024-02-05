from typing import Sequence, Union

import numpy as np
import pytest

from pydentification.data import splits


@pytest.mark.parametrize(
    ["validation_size", "n_samples", "expected_validation_samples"],
    (
        (0.2, 100, 20),
        (20, 100, 20),
        (0.5, 100, 50),
        (50, 100, 50),
        (0.8, 100, 80),
        (80, 100, 80),
    ),
)
def test_compute_n_validation_samples(validation_size: int | float, n_samples: int, expected_validation_samples: int):
    n_validation_samples = splits.compute_n_validation_samples(validation_size, n_samples)
    assert n_validation_samples == expected_validation_samples


def test_compute_n_validation_samples_raises_value_error():
    with pytest.raises(ValueError):
        splits.compute_n_validation_samples(validation_size=100, n_samples=50)


@pytest.mark.parametrize(
    ["validation_size", "n_samples", "expected_shape"],
    (
        (0.2, 100, (20,)),
        (20, 100, (20,)),
        (0.5, 100, (50,)),
        (50, 100, (50,)),
        (0.8, 100, (80,)),
        (80, 100, (80,)),
    ),
)
def test_draw_validation_samples(validation_size: int | float, n_samples: int, expected_shape: tuple[int]):
    validation_samples = splits.draw_validation_indices(validation_size, n_samples)
    assert validation_samples.shape == expected_shape


@pytest.mark.parametrize(
    "sequence, test_size, expected_train_size, expected_test_size",
    [
        (np.arange(100), 0.5, 50, 50),
        (list(range(100)), 0.5, 50, 50),
        (np.arange(100), 0.3, 70, 30),
        (np.arange(100), 56, 44, 56),
        (np.arange(100), 28, 72, 28),
        (np.arange(300).reshape(100, 3), 0.4, 60, 40),
    ],
)
def test_time_series_train_test_split(
    sequence: Sequence, test_size: Union[int, float], expected_train_size: int, expected_test_size: int
):
    train_sequence, test_sequence = splits.time_series_train_test_split(sequence, test_size)

    assert len(train_sequence) == expected_train_size
    assert len(test_sequence) == expected_test_size
