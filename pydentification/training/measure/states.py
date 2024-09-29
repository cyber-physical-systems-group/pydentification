from enum import Enum
from typing import Final


class TrainingStage(Enum):
    on_train_start: Final[str] = "on_train_start"
    on_train_end: Final[str] = "on_train_end"
    on_train_epoch_start: Final[str] = "on_train_epoch_start"
    on_train_epoch_end: Final[str] = "on_train_epoch_end"


class TrainingStates(Enum):
    training: Final[str] = "training"
    initial: Final[str] = "initial"
    final: Final[str] = "final"
