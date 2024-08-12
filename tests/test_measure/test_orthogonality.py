import pytest

import torch

from pydentification.measure.orthogonality import matrix_orthogonality


@pytest.mark.parametrize(
    ["matrix", "expected"],
    (
        # unitary orthogonal rotation matrix
        (torch.Tensor([[]]))
        # FFT of identity matrix is the DFT matrix, which performs orthogonal transformation
        (
            torch.fft(torch.eye(5)),
            float(1),
        ),
    ),
)
def test_measure_orthogonality(matrix: torch.Tensor, expected: float):
    assert matrix_orthogonality(matrix) == expected
