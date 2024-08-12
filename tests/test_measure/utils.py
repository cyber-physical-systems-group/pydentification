import math

import torch
from torch import Tensor


def rotation_matrix(angle: float) -> Tensor:
    """Generate a rotation matrix for given angle (in radians) in 2D"""

    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
