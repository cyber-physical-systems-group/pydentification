# typing is ignored due to using dynamic casting between torch, numpy and pandas, which is not properly handled by mypy
# type: ignore
from collections import OrderedDict
from typing import Iterable, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from ..sequences import generate_time_series_windows, time_series_train_test_split
from ..splits import draw_validation_indices


class TensorDatasetCache:
    """
    Almost naive class for caching TensorDatasets. It is used in PredictionDataModule to cache train or validation
    datasets. The cache is limited to given size and uses FIFO policy to remove old datasets so memory is over-consumed.

    The cache is implemented as class, since using @cache or @lru_cache caches only single function and the datamodule
    generates the same windows when called with multiple methods (train and validation or test and prediction datasets).
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def add(self, key: int, dataset: TensorDataset) -> None:
        if key in self.cache:
            return  # already in cache

        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # pop with FIFO order

        self.cache[key] = dataset

    def get(self, item: int) -> TensorDataset | None:
        return self.cache.get(item)


class PredictionDataModule(pl.LightningDataModule):
    """
    Datamodule for prediction models supporting dynamically changed autoregression windows. It intentionally supports
    limited number of parameters passed to "generate_time_series_windows" to avoid destroying causality in tests or
    generating multiple overlapping samples, (overlap is allowed during training).

    Windows are created dynamically from the states, split into train and test, (see "time_series_train_test_split")
    and cached for given autoregression window size. This allows to train and test models with different autoregression
    window sizes without generating new windows for each size every time. This datamodule also supports validation set
    generated dynamically from training set, (see "draw_validation_indices").

    `n_forward_time_steps` can be set during method call (training outside lightning) or using callback setting the
    attribute in the datamodule (when using lightning). Implementation of such callback is given in `training.lightning.callback`.  # noqa: E501
    """

    def __init__(
        self,
        states: NDArray | None = None,
        *,
        test_size: Union[int, float] = 0.5,
        validation_size: Optional[Union[int, float]] = None,
        batch_size: int = 32,
        n_workers: int = 0,
        n_backward_time_steps: int = 1,
        n_forward_time_steps: int = 1,
        train_shift: int = 1,
        cache_size: int = 5,
    ):
        """
        :param states: measurements of states of the (autonomous) system given as numpy
        :param test_size: test size in samples or ration, for details see `time_series_train_test_split`
        :param validation_size: size of validation dataset as a ratio of training data or absolute number of samples
                                validation samples are drawn from training data
        :param batch_size: batch size used to train and test the model by torch DataLoaders
        :param n_workers: number of workers used by torch DataLoaders
        :param n_backward_time_steps: number of output (state) measurements to include before the prediction start
        :param n_forward_time_steps: number of output (state) measurements to predict
        :param train_shift: number of samples to move the prediction starting point, can generate overlapping samples
        """
        super().__init__()

        self.states = states

        self.test_size = test_size
        self.validation_size = validation_size

        self.batch_size = batch_size
        self.n_workers = n_workers

        # model input is fixed and it needs to be equal to the number of backward samples
        self.n_backward_time_steps = n_backward_time_steps
        self.n_forward_time_steps = n_forward_time_steps  # set initial value, which can be changed by Callbacks
        self.train_shift = train_shift

        # cache placeholders
        self.train_states = None
        self.test_states = None

        self.train_cache = TensorDatasetCache(cache_size)
        self.validation_cache = TensorDatasetCache(cache_size)
        self.test_cache = TensorDatasetCache(cache_size)

        # use only NODE_RANK = 0
        # see: https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data-per-node
        self.prepare_data_per_node = False

    @classmethod
    def from_pandas(cls, dataset: pd.DataFrame, columns: list[str] | None = None, **kwargs):
        """
        Creates PredictionDataModule from pandas dataframe

        :param dataset: pandas dataframe containing measurements of states of the (autonomous) system
        :param columns: list of columns with state measurements of the system
        """
        columns = columns or dataset.columns
        return cls(dataset[columns].values, **kwargs)

    @classmethod
    def from_csv(cls, dataset_path: str, columns: list[str] | None = None, **kwargs):
        """
        Creates PredictionDataModule from CSV file containing input and output measurements in columns
        Shortcut for using `pd.read_csv` and `PredictionDataModule.from_pandas` together
        """
        dataset = pd.read_csv(dataset_path)
        return cls.from_pandas(dataset, columns, **kwargs)

    def setup(self, stage: str | None = None) -> None:
        """
        Prepares dataset for training and test by splitting the data into training and test sets
        The windows are generated dynamically using the parameter passed to each dataloader
        """
        train_outputs, test_outputs = time_series_train_test_split(self.states, test_size=self.test_size)
        self.train_states = train_outputs
        self.test_states = test_outputs

    def sample_and_cache(self, n_time_steps: int) -> tuple[TensorDataset, TensorDataset | None]:
        """
        Sample and cache training and validation windows for train states with fixed (given by model hyper-parameters)
        backward window size and forward window size given as changed parameter, possibly different in different epochs.

        The method prepares data in following way:
        1. Generate time-series windows with given backward and forward window sizes
        2. Split the windows into training and validation windows (if validation size given)
        3. Convert the windows into torch TensorDataset
        4. Cache the datasets for given forward window size and return to generate dataloaders
        """
        windows = generate_time_series_windows(
            outputs=self.train_states,
            backward_output_window_size=self.n_backward_time_steps,
            forward_output_window_size=n_time_steps,
            shift=self.train_shift,
        )

        n_samples = len(windows["forward_outputs"])  # forward_outputs are always present

        if not self.validation_size:  # no validation data
            dataset = TensorDataset(*map(torch.from_numpy, [s for s in windows.values() if s.size > 0]))
            self.train_cache.add(key=n_time_steps, dataset=dataset)
            return dataset, None

        validation_index = draw_validation_indices(self.validation_size, n_samples)
        # window dict is converted to list of numpy arrays, where the input and output of the model is determined
        # by order in window generation dict, which is always the same (see main docstring)
        train_samples = [np.delete(s, validation_index, axis=0) for s in windows.values() if s.size > 0]
        validation_samples = [np.take(s, validation_index, axis=0) for s in windows.values() if s.size > 0]

        train_dataset = TensorDataset(*map(torch.from_numpy, train_samples))
        validation_dataset = TensorDataset(*map(torch.from_numpy, validation_samples))

        self.train_cache.add(key=n_time_steps, dataset=train_dataset)
        self.validation_cache.add(key=n_time_steps, dataset=validation_dataset)

        return train_dataset, validation_dataset

    def sample_for_test_or_predict(self, n_forward_time_steps: int) -> Iterable:
        """Generates data for test or prediction and returns torch DataLoader for given amount of forward time steps"""
        if not n_forward_time_steps:  # when using lightning `self.n_forward_time_steps` is set by callback
            n_forward_time_steps = self.n_forward_time_steps

        if test_dataset := self.test_cache.get(n_forward_time_steps):
            return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

        windows = generate_time_series_windows(
            outputs=self.test_states,
            backward_output_window_size=self.n_backward_time_steps,
            forward_output_window_size=n_forward_time_steps,
            shift=n_forward_time_steps,
        )

        dataset = TensorDataset(*map(torch.from_numpy, [s for s in windows.values() if s.size > 0]))
        self.test_cache.add(key=n_forward_time_steps, dataset=dataset)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

    def train_dataloader(self, n_forward_time_steps: int | None = None) -> Iterable:
        """
        Generates training data and returns torch DataLoader for given amount of forward time steps and fixed backward
        time steps. Windows are generated dynamically with shift given during the training.
        """
        if not n_forward_time_steps:  # when using lightning `self.n_forward_time_steps` is set by callback
            n_forward_time_steps = self.n_forward_time_steps

        if train_dataset := self.train_cache.get(n_forward_time_steps):
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

        train_dataset, _ = self.sample_and_cache(n_forward_time_steps)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self, n_forward_time_steps: int | None = None) -> Iterable:
        """
        Generates training data and returns torch DataLoader
        with validation data which is removed from training data loader
        """
        if not n_forward_time_steps:  # when using lightning `self.n_forward_time_steps` is set by callback
            n_forward_time_steps = self.n_forward_time_steps

        if self.validation_size is None:
            raise ValueError("Validation size must be specified to use validation data loader!")

        if validation_dataset := self.validation_cache.get(n_forward_time_steps):
            return DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

        _, validation_dataset = self.sample_and_cache(n_forward_time_steps)
        return DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def test_dataloader(self, n_forward_time_steps: int | None = None) -> Iterable:
        """Generates test data and returns torch DataLoader for given amount of forward time steps"""
        if not n_forward_time_steps:  # when using lightning `self.n_forward_time_steps` is set by callback
            n_forward_time_steps = self.n_forward_time_steps

        if isinstance(n_forward_time_steps, int):  # return single dataloader
            return self.sample_for_test_or_predict(n_forward_time_steps)

        # returns multiple dataloaders with different number of autoregressive steps
        if isinstance(n_forward_time_steps, Iterable):
            return [self.sample_for_test_or_predict(n) for n in n_forward_time_steps]

        raise ValueError(f"n_forward_time_steps must be int or Iterable[int], got {type(n_forward_time_steps)}")

    def predict_dataloader(self, n_forward_time_steps: int | Iterable[int] | None = None) -> Iterable:
        """Generates test data and returns torch DataLoader for given amount of forward time steps"""
        if not n_forward_time_steps:  # when using lightning `self.n_forward_time_steps` is set by callback
            n_forward_time_steps = self.n_forward_time_steps

        if isinstance(n_forward_time_steps, int):  # return single dataloader
            return self.sample_for_test_or_predict(n_forward_time_steps)

        # returns multiple dataloaders with different number of autoregressive steps
        if isinstance(n_forward_time_steps, Iterable):
            return [self.sample_for_test_or_predict(n) for n in n_forward_time_steps]

        raise ValueError(f"n_forward_time_steps must be int or Iterable[int], got {type(n_forward_time_steps)}")
