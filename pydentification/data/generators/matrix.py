import numpy as np
from numpy.typing import NDArray


def random_sparse_matrix(
    rows: int,
    cols: int,
    value_range: tuple[float, float] = (float(0), float(1)),
    non_zero_prob: float = 0.5,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> NDArray:
    """
    Generate a random sparse matrix with given dimensions and value range.

    :param rows: Number of rows in the matrix.
    :param cols: Number of columns in the matrix.
    :param value_range: Range of values to fill the matrix with, default is (0.0, 1.0).
    :param non_zero_prob: Probability of a non-zero entry in the matrix, default is 0.5.
    :param rng: Optional random number generator. If None, a default generator is created.
    :param seed: Seed for the random number generator, default is 0.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    mask = rng.random((rows, cols)) < non_zero_prob
    random_matrix = rng.uniform(value_range[0], value_range[1], size=(rows, cols))

    return mask * random_matrix
