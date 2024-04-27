import numpy as np
import pytest

from pydentification.data.datamodules.prediction import PredictionDataModule


@pytest.fixture(scope="module")
def random_prediction_datamodule():
    """Random datamodule used for callback testing."""

    return PredictionDataModule(
        states=np.random.rand(10000, 1),
        test_size=0.5,
        validation_size=0.1,
        batch_size=1,
    )
