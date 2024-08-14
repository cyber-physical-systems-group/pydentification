import math

import pytest
import torch

from pydentification.measure.matrix import unitarity
from tests.test_measure import utils


@pytest.mark.parametrize(
    ["matrix"],
    (
        # identity matrix is unitary
        (torch.eye(4),),
        # rotation matrix is unitary
        (utils.compose_rotation_matrix(math.pi / 6),),
        # another rotation matrix
        (utils.compose_rotation_matrix(math.pi / 3),),
        # FFT of identity matrix is the DFT matrix, which performs orthogonal transformation -> it is unitary matrix
        (utils.compose_dft_matrix(4),),
    ),
)
def test_measure_exact_orthogonality(matrix: torch.Tensor):
    assert unitarity(matrix) == float(1)


@pytest.mark.parametrize(
    ["randomness", "expected_low", "expected_high"],
    (
        # see description of measure statistics in pydentification/measure/README.md
        (0.1, 0.82, 0.88),
        (0.2, 0.62, 0.72),
        (0.3, 0.40, 0.56),
        (0.4, 0.22, 0.35),
        (0.5, 0.10, 0.22),
        (0.6, 0.04, 0.12),
        (0.7, 0, 0.5),
        (0.8, 0, 0.02),
        (0.9, 0, 0.01),
        (1.0, 0, 0.005),
    ),
)
def test_measure_approximate_orthogonality(randomness: float, expected_low: float, expected_high: float):
    matrix = torch.eye(32) + randomness * torch.randn(32, 32)
    assert expected_low <= unitarity(matrix) <= expected_high
