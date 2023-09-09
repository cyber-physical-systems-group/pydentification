import numpy as np
from numpy.typing import NDArray


def compute_n_validation_samples(validation_size: int | float, n_samples: int) -> int:
    """Computes the number of validation samples based on the number of training windows"""
    if validation_size > 1:  # when validation_size is absolute number of samples
        if validation_size > n_samples:
            raise ValueError(
                "Number of validation windows  must be smaller than total number of windows!"
                f"{validation_size} > {n_samples}"
            )
        return int(validation_size)
    # when validation_size is a fraction of training data
    return int(validation_size * n_samples)


def draw_validation_indices(validation_size: int | float, n_samples: int) -> NDArray:
    """Returns randomly chosen validation samples"""
    n_validation_samples = compute_n_validation_samples(validation_size, n_samples)
    return np.random.choice(list(range(n_samples)), size=n_validation_samples)
