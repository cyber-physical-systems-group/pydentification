import math

import torch
from torch import Tensor


def compose_rotation_matrix(angle: float) -> Tensor:
    """Generate a rotation matrix for given angle (in radians) in 2D"""

    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )


def compose_dft_matrix(n: int) -> Tensor:
    """Generate Discrete Fourier Transform matrix of size n x n"""
    return torch.fft.fft(torch.eye(n)) / torch.sqrt(torch.tensor(n, dtype=torch.float32))
