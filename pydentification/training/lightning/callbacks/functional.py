from typing import Literal


def is_better(current: float, best: float, threshold: float, threshold_mode: Literal["abs", "rel"]) -> bool:
    if threshold_mode == "rel":
        return current < best * (float(1) - threshold)

    else:  # threshold_mode == "abs":
        return current < best - threshold
