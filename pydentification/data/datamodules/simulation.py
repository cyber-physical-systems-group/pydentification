# typing is ignored due to using dynamic casting between torch, numpy and pandas, which is not properly handled by mypy
# type: ignore
from pathlib import Path
from typing import Iterable, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ..sequences import generate_time_series_windows, time_series_train_test_split
from ..splits import draw_validation_indices


def dataloader_from_numpy(
    arrays: Iterable[NDArray],
    batch_size: int,
    n_workers: int = 0,
    shuffle: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """
    Creates torch DataLoader from any number numpy arrays

    :param arrays: iterable of numpy arrays
    :param batch_size: batch size used to train and test the model by torch DataLoaders
    :param n_workers: number of workers used by torch DataLoaders
    :param shuffle: whether to shuffle the dataset
    :param dtype: torch dtype of the tensors returned by dataloaders, defaults to torch.float32
    """
    tensors = map(lambda sample: torch.from_numpy(sample).to(dtype), arrays)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)


class StaticSimulationDataModule(pl.LightningDataModule):
    """
    Simulation modelling can be applied to static systems, which is equivalent to curve fitting from dataset
    perspective. This datamodule can be used to generate dataset for such problems.
    """

    def __init__(
        self,
        inputs: NDArray,
        outputs: NDArray,
        *,
        test_size: int | float = 0.5,
        test_split: Literal["index", "random"] = "random",
        validation_size: int | float | None = None,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        self.inputs = inputs
        self.outputs = outputs

        self.test_size = test_size
        self.test_split = test_split
        self.validation_size = validation_size
        self.batch_size = batch_size

        # cache placeholders
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        # use only NODE_RANK = 0
        # see: https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data-per-node
        self.prepare_data_per_node = False
        self.dtype = dtype

    def setup(self, stage: Literal["fit", "predict"]) -> None:
        """
        Prepares dataset for training or testing using following steps:
        1. Split into train and test based on the time series
        2. Split training into training subset and validation randomly
        3. Store cached samples for dataloaders
        """
        if self.test_split == "random":
            train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
                self.inputs, self.outputs, test_size=self.test_size
            )

            if self.validation_size is not None:
                train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
                    train_inputs, train_outputs, test_size=self.validation_size
                )

                self.train_samples = [train_inputs, train_outputs]
                self.val_samples = [val_inputs, val_outputs]
            else:
                self.train_samples = [train_inputs, train_outputs]
                self.val_samples = [[], []]

            self.test_samples = [test_inputs, test_outputs]

        elif self.test_split == "index":
            train_inputs, test_inputs = time_series_train_test_split(self.inputs, test_size=self.test_size)
            train_outputs, test_outputs = time_series_train_test_split(self.outputs, test_size=self.test_size)

            if self.validation_size is not None:
                train_inputs, val_inputs = time_series_train_test_split(train_inputs, test_size=self.validation_size)
                train_outputs, val_outputs = time_series_train_test_split(train_outputs, test_size=self.validation_size)
            else:
                val_inputs, val_outputs = [], []

            self.train_samples = [train_inputs, train_outputs]
            self.val_samples = [val_inputs, val_outputs]
            self.test_samples = [test_inputs, test_outputs]

    @classmethod
    def from_pandas(cls, dataset: pd.DataFrame, input_columns: list[str], output_columns: list[str], **kwargs):
        """
        Creates SimulationDataModule from pandas DataFrame containing input and output measurements in columns

        :param dataset: pandas DataFrame containing input and output measurements in columns
        :param input_columns: list of columns with input measurements of the system
        :param output_columns: list of columns with output measurements of the system
        """
        inputs = dataset[input_columns].values
        outputs = dataset[output_columns].values
        return cls(inputs, outputs, **kwargs)

    @classmethod
    def from_csv(cls, dataset_path: str | Path, input_columns: list[str], output_columns: list[str], **kwargs):
        """
        Creates SimulationDataModule from CSV file containing input and output measurements in columns
        Shortcut for using `pd.read_csv` and `SimulationDataModule.from_pandas` together
        """
        dataset = pd.read_csv(dataset_path)
        return cls.from_pandas(dataset, input_columns, output_columns, **kwargs)

    def train_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.train_samples, self.batch_size, shuffle=True, dtype=self.dtype)

    def val_dataloader(self) -> Iterable:
        if self.val_samples is None:
            raise ValueError("Validation size must be specified to use validation data loader!")

        return dataloader_from_numpy(self.val_samples, self.batch_size, shuffle=False, dtype=self.dtype)

    def test_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.test_samples, self.batch_size, shuffle=False, dtype=self.dtype)

    def predict_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.test_samples, self.batch_size, shuffle=False, dtype=self.dtype)


class SimulationDataModule(pl.LightningDataModule):
    """
    DataModule for simulation modelling of dynamical systems.
    Windows are generated based on parameters passed to `generate_time_series_windows` function.

    The dataloaders return batches of windows, where each window is one of the system inputs or outputs with time-shifts
    They are returned in following order (can be more than 2, but if given sequence is not used it will not be present):
    1. backward_inputs
    2. backward_outputs
    3. forward_inputs
    4. forward_outputs

    Model and training needs to be defined to follow this order, for example in typical simulation modelling forward
    inputs are features and forward outputs are labels. For other types of modelling this may be different.
    """

    def __init__(
        self,
        inputs: NDArray,
        outputs: NDArray,
        *,
        test_size: int | float = 0.5,
        validation_size: int | float | None = None,
        batch_size: int = 32,
        n_workers: int = 0,
        forward_input_window_size: int = 0,
        backward_input_window_size: int = 0,
        forward_output_window_size: int = 0,
        backward_output_window_size: int = 0,
        shift: int = 1,
        forward_input_mask: int = 0,
        backward_input_mask: int = 0,
        forward_output_mask: int = 0,
        backward_output_mask: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param inputs: numpy array containing systems inputs
        :param outputs: numpy array containing systems outputs
        :param test_size: test size in samples or ration, for details see `time_series_train_test_split`
        :param validation_size: size of validation dataset as a ratio of training data or absolute number of samples
                                validation samples are drawn from training data
        :param batch_size: batch size used to train and test the model by torch DataLoaders
        :param n_workers: number of workers used by torch DataLoaders
        :param forward_input_window_size: number of input (forcing) measurements to include forward from prediction start  # noqa: E501
        :param backward_input_window_size: number of input (forcing)  measurements to include forward from prediction start  # noqa: E501
        :param forward_output_window_size: number of output (state) measurements to include forward from prediction start  # noqa: E501
        :param backward_output_window_size: number of output (state) measurements to include before the prediction start
        :param shift: number of samples to move the prediction starting point, can generate overlapping samples
        :param forward_input_mask: number of masked samples for forward inputs (forcing)
        :param backward_input_mask: number of masked samples for backward inputs (forcing)
        :param forward_output_mask: number of masked samples for forward outputs (states)
        :param backward_output_mask: number of masked samples for backward outputs (states)
        :param dtype: torch dtype of the tensors returned by dataloaders, defaults to torch.float32
        """
        super().__init__()

        self.inputs = inputs
        self.outputs = outputs
        self.test_size = test_size
        self.validation_size = validation_size

        self.batch_size = batch_size
        self.n_workers = n_workers

        # window generation config
        self.window_generation_kwargs = dict(
            forward_input_window_size=forward_input_window_size,
            backward_input_window_size=backward_input_window_size,
            forward_output_window_size=forward_output_window_size,
            backward_output_window_size=backward_output_window_size,
            shift=shift,
            forward_input_mask=forward_input_mask,
            backward_input_mask=backward_input_mask,
            forward_output_mask=forward_output_mask,
            backward_output_mask=backward_output_mask,
        )

        # cache placeholders
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        # use only NODE_RANK = 0
        # see: https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data-per-node
        self.prepare_data_per_node = False
        self.dtype = dtype

    @classmethod
    def from_pandas(cls, dataset: pd.DataFrame, input_columns: list[str], output_columns: list[str], **kwargs):
        """
        Creates SimulationDataModule from pandas DataFrame containing input and output measurements in columns

        :param dataset: pandas DataFrame containing input and output measurements in columns
        :param input_columns: list of columns with input measurements of the system
        :param output_columns: list of columns with output measurements of the system
        """
        inputs = dataset[input_columns].values
        outputs = dataset[output_columns].values
        return cls(inputs, outputs, **kwargs)

    @classmethod
    def from_csv(cls, dataset_path: str | Path, input_columns: list[str], output_columns: list[str], **kwargs):
        """
        Creates SimulationDataModule from CSV file containing input and output measurements in columns
        Shortcut for using `pd.read_csv` and `SimulationDataModule.from_pandas` together
        """
        dataset = pd.read_csv(dataset_path)
        return cls.from_pandas(dataset, input_columns, output_columns, **kwargs)

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None:
        """
        Prepares dataset for training, validation or testing using following steps:
        1. Split into train and test based on the time series
        2. Generate time series windows for training and testing
        3. Split training into training subset and validation randomly
        4. Convert windows to torch tensors and create dataloaders
        """
        train_inputs, test_inputs = time_series_train_test_split(self.inputs, test_size=self.test_size)
        train_outputs, test_outputs = time_series_train_test_split(self.outputs, test_size=self.test_size)

        if stage == "fit":
            kwargs = self.window_generation_kwargs
            windows = generate_time_series_windows(inputs=train_inputs, outputs=train_outputs, **kwargs)
            n_samples = len(windows["forward_outputs"])  # forward_outputs are always present

            validation_index = draw_validation_indices(self.validation_size, n_samples)
            # window dict is converted to list of numpy arrays, where the input and output of the model is determined
            # by order in window generation dict, which is always the same (see main docstring)
            self.train_samples = [np.delete(s, validation_index, axis=0) for s in windows.values() if s.size > 0]
            self.val_samples = [np.take(s, validation_index, axis=0) for s in windows.values() if s.size > 0]

        if stage == "test" or stage == "predict":
            kwargs = self.window_generation_kwargs
            windows = generate_time_series_windows(inputs=test_inputs, outputs=test_outputs, **kwargs)
            self.test_samples = [s for s in windows.values() if s.size > 0]

    def train_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.train_samples, self.batch_size, shuffle=True, dtype=self.dtype)

    def val_dataloader(self) -> Iterable:
        if self.validation_size is None:
            raise ValueError("Validation size must be specified to use validation data loader!")

        return dataloader_from_numpy(self.val_samples, self.batch_size, shuffle=False, dtype=self.dtype)

    def test_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.test_samples, self.batch_size, shuffle=False, dtype=self.dtype)

    def predict_dataloader(self) -> Iterable:
        return dataloader_from_numpy(self.test_samples, self.batch_size, shuffle=False, dtype=self.dtype)
