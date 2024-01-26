# Data

This module contains the data loading utils for system identification. 

### Contents

* `datamodules` - collection of lightning data modules for loading data, described in detail below
* `process` - collection of `torch` utils for processing and transforming data
* `sequences` - collection of function working with `numpy` for generating sequences of data for system identification
* `splits` - collection of functions for splitting data into train and test sets, works with `numpy` arrays

## Core

### Window Generation

The core functionality is window generation for time series data or system measurements. Two core identification
problems are supported: simulation modelling and predictive modelling [1]. 

Generating a windowed dataset for simulation modelling, with input length of 64 samples of excitation and output length
of 16 samples of measured system response, where the alignment of excitation and response is from the last sample.

```python
import numpy as np
from pydentification.data import generate_time_series_windows

# some random paths for the example 
inputs = np.loadtxt("inputs.csv")
outputs = np.loadtxt("outputs.csv")

windows = generate_time_series_windows(
    inputs=inputs,
    outputs=outputs,
    shift=1,
    forward_input_window_size=64,
    forward_output_window_size=16,
    forward_output_mask=64 - 16,
)
```

Loading data for predictive modelling can be done using similar method:

```python
import numpy as np
from pydentification.data import generate_time_series_windows

# just outputs are required
outputs = np.loadtxt("outputs.csv")


windows = generate_time_series_windows(
    outputs=outputs,
    shift=1,
    backward_output_window_size=16,
    forward_output_window_size=16,
)
```

### Splitting

This module also contains utils for splitting data into train and test sets. It is not random sampling, like for most 
datasets, since the goal of system identification is to predict future system behaviour, so splitting is done based
on the time axis of the dataset.

```python
from pydentification.data import time_series_train_test_split


train, test = time_series_train_test_split(dataset, test_size=0.5)  # assume dataset exists and can be indexed
```

## Data Modules

Data modules are utils based on lightning data modules. They are used for loading data and can be extended for
preprocessing. Implemented datamodules are:
* `SimulationDataModule` - for loading data from datasets meant for simulation modelling
* `PredictionDataModule` - for loading data from datasets meant for system prediction with ability to dynamically change autoregression window size

### Simulation DataModule

Simulation and prediction support for CSV based datasets. It uses `generate_time_series_windows` and pandas for loading
data. It is recommended to use this module for small datasets.

Model and training needs to be defined to follow this order, for example in typical simulation modelling forward inputs
are features and forward outputs are labels. For other types of modelling this may be different. 

```python
from pydentification.data.datamodules.simulation import SimulationDataModule


dm = SimulationDataModule(
    dataset_path="dataset.csv",  # CSV file with data 
    input_columns=["x"],
    output_columns=["y"],
    test_size=0.5, 
    batch_size=32,
    validation_size=0.1,
    shift=1,
    forward_input_window_size=32,
    forward_output_window_size=32,
    forward_output_mask=31,
)

trainer.fit(model, datamodule=dm)  # assume trainer and model exist and define required lightning interface
```

Module defined as above will return 2 items in each dataloader, one will be forward inputs to the system, which are
model features and one will be forward system outputs, which are model target. Example code to handle this is in model:

```python
def training_step(self, batch, batch_idx):  # assume defined in lightning module
    x, y = batch
    y_hat = self(x)  # type: ignore
    loss = self.loss(y_hat, y)
    self.log("train/loss", loss)

    return loss
```

### Prediction Data Module

This module is used for autoregressive training for predictive modelling. It supports number of features for such
problems, including:
* Dynamically changing autoregression length during training via callbacks
* Generating multiple test or prediction sets with different prediction horizons
* Caching datasets for given window size (using Trainer parameters `reload_dataloaders_every_n_epochs=1` is required for autoregression length change)

Creating datamodule is similar to `PredictionDataModule`:

```python
from pydentification.data.datamodules.prediction import PredictionDataModule


dm = PredictionDataModule(
    states,  # numpy array with system states
    test_size=0.5,
    batch_size=32,
    validation_size=0.1,
    n_backward_time_steps=64,  # model hyperparameter
    n_forward_time_steps=16,  # initial prediction horizon, can be changed during or after training
)
```

To change autoregression length during training, using `reload_dataloaders_every_n_epochs=1` is required. This will 
make sure trainer does not cache dataloaders and will reload them every epoch. To change autoregression length, the 
class attribute `n_forward_time_steps` needs to be overwritten. This can be done using callbacks:

```python
from typing import Any

import pytorch_lightning as pl


class ExampleCallback(pl.Callback):
    def __init__(self, overwrite_after_n_epochs: int, overwrite_to: int):
        self.overwrite_after_n_epochs = overwrite_after_n_epochs
        self.overwrite_to = overwrite_to

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if trainer.current_epoch >= self.overwrite_after_n_epochs:
            trainer.datamodule.n_forward_time_steps = self.overwrite_to  # overwrite
```

For testing multiple prediction horizons, the `n_forward_time_steps` can be given as iterable to method for generating
test or predict dataloaders. This will generate multiple dataloaders with different prediction horizons and `pl.Trainer`,
can handle inference on multiple dataloaders, they can be accessed using `dataloader_idx` parameter or `predict` or 
`test` functions. Return value of the `predict` method given more than single dataloader is a list of predictions, for
each of the datalodaers in the same order as they were given.

```python
prediction_horizons = (16, 64, 128, 512, 1024, 4096)

trainer = pl.Trainer(precision=64)
# assumes model and datamodule exist, y_hat is a list with 6 predictions each for different prediction horizon
y_hat = trainer.predict(model, dataloaders=dm.test_dataloader(n_forward_time_steps=prediction_horizons))
```


In complex modelling, where system inputs are used and past outputs are auxiliary features, following datamodule can be
defined and handled in the model.

```python
from pydentification.data.datamodules.prediction import PredictionDataModule


dm = PredictionDataModule.from_csv(
    dataset_path="dataset.csv",  # CSV file with data 
    input_columns=["u"],
    output_columns=["x", "y", "z"],
    test_size=0.5, 
    batch_size=32,
    validation_size=0.1,
    shift=1,
    forward_input_window_size=16,  # 16 step forward core features
    backward_output_window_size=32,  # 32-step back auxiliary features
    forward_output_window_size=16,  # 16-step ahead prediction 
)
```

To handle this in model and trainer, following code can be used:

```python
def training_step(self, batch, batch_idx):  # assume defined in lightning module
    system_inputs, past_system_outputs, targets = batch
    y_hat = self(system_inputs, past_system_outputs)  # handle 2 inputs to the model to get prediction
    loss = self.loss(y_hat, targets)
    self.log("train/loss", loss)

    return loss
```

## References

<a id="1">[1]</a> 
Johan Schoukens and Lennart Ljung (2019). 
*Nonlinear System Identification: A User-Oriented Roadmap.*
https://arxiv.org/abs/1902.00683
