import pytest

from pydentification.data.datamodules.splits import compute_n_validation_samples, draw_validation_indices


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
    n_validation_samples = compute_n_validation_samples(validation_size, n_samples)
    assert n_validation_samples == expected_validation_samples


def test_compute_n_validation_samples_raises_value_error():
    with pytest.raises(ValueError):
        compute_n_validation_samples(validation_size=100, n_samples=50)


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
    validation_samples = draw_validation_indices(validation_size, n_samples)
    assert validation_samples.shape == expected_shape
