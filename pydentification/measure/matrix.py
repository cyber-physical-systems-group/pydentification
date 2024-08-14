import math

import torch
from torch import Tensor

from .functional import conjugate_transpose


def unitarity(matrix: Tensor) -> float:
    """
    Function for measuring the unitarity of a matrix in range 0 to 1 using Frobenius norm.
    For formuals and explanation, see pydentification/measure/README.md and tests/test_measure/test_unitarity.py.
    """
    n, m = matrix.size()

    if torch.is_complex(matrix):
        transposed = conjugate_transpose(matrix)
    else:
        transposed = torch.transpose(matrix, 0, 1)

    left = matrix @ transposed
    right = transposed @ matrix
    # equivalent to 1/2 (I - QQ^T) + (I - Q^TQ)
    measure = torch.eye(n) - ((left + right) / 2)
    norm = torch.linalg.norm(measure, ord="fro")

    return torch.exp(-1 * norm / math.sqrt(n * m)).item()
