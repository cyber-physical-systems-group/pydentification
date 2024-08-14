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
