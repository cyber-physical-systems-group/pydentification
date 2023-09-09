# typing is ignored due to using dynamic casting between torch, numpy and pandas, which is not properly handled by mypy
# type: ignore
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..sequences import generate_time_series_windows, time_series_train_test_split
from ..splits import draw_validation_indices


class CsvDataModule(pl.LightningDataModule):
    """
    DataModule for simulation and prediction datasets based on CSV files. The CSV file must contain columns with inputs
    and outputs of the system (not the model, in predictive modelling system *outputs* can be model *inputs*, depending
    on the time shift). Windows are generated based on parameters passed to `generate_time_series_windows` function.

    The dataloaders return batches of windows, where each window is one of the system inputs or outputs with time-shifts
    They are returned in following order (can be more than 2, but if given sequence is not used it will not be present):
    1. backward_inputs2
    2. backward_outputs
    3. forward_inputs
    4. forward_outputs

    Model and training needs to be defined to follow this order, for example in typical simulation modelling forward
    inputs are features and forward outputs are labels. For other types of modelling this may be different.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        input_columns: list[str],
        output_columns: list[str],
        *,
        test_size: Union[int, float] = 0.5,
        validation_size: Optional[Union[int, float]] = None,
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
    ):
        """
        :param dataset_path: path to CSV file with dataset
        :param input_columns: names of columns containing systems inputs
        :param output_columns: names of columns containing systems outputs
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
        """
        self.dataset_path = dataset_path
        self.input_columns = input_columns
        self.output_columns = output_columns
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
        self.train_dataset = None
        self.test_dataset = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

    def setup(self, stage: Literal["fit", "test"]):
        """
        Prepares dataset for training, validation or testing using following steps:
        1. Load dataset from CSV file
        2. Split into train and test based on the time series
        3. Generate time series windows for training and testing
        4. Split training into training subset and validation randomly
        5. Convert windows to torch tensors and create dataloaders
        """
        dataset = pd.read_csv(self.dataset_path)
        train_dataset, test_dataset = time_series_train_test_split(dataset, test_size=self.test_size)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        if stage == "fit":
            inputs = self.train_dataset[self.input_columns].values if self.input_columns else None
            outputs = self.train_dataset[self.output_columns].values

            windows = generate_time_series_windows(inputs=inputs, outputs=outputs, **self.window_generation_kwargs)
            n_samples = len(windows["forward_outputs"])  # forward_outputs are always present

            validation_index = draw_validation_indices(self.validation_size, n_samples)
            # window dict is converted to list of numpy arrays, where the input and output of the model is determined
            # by order in window generation dict, which is always the same (see main docstring)
            self.train_samples = [np.delete(s, validation_index, axis=0) for s in windows.values() if s.size > 0]
            self.val_samples = [np.take(s, validation_index, axis=0) for s in windows.valyes() if s.size > 0]

        if stage == "test":
            inputs = self.test_dataset[self.input_columns].values if self.input_columns else None
            outputs = self.test_dataset[self.output_columns].values

            windows = generate_time_series_windows(inputs=inputs, outputs=outputs, **self.window_generation_kwargs)
            self.test_samples = [s for s in windows.values() if s.size > 0]

    def train_dataloader(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        dataset = TensorDataset(*map(torch.from_numpy, self.train_samples))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self) -> Iterable:
        """
        Generates training data and returns torch DataLoader
        with validation data which is removed from training data loader
        """
        if self.validation_size is None:
            raise ValueError("Validation size must be specified to use validation data loader!")

        dataset = TensorDataset(*map(torch.from_numpy, self.val_samples))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

    def test_dataloader(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        dataset = TensorDataset(*map(torch.from_numpy, self.test_samples))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
