# Scripts

This directory contains sample scripts, which can be used to run W&B experiments. They are not part of the library,
but are useful convenience tool (we found ourselves writing the same code over and over).

## How to Use

Currently, the scripts are not installed with the library, so they need to be manually copied. This can be treated as a 
template for experimentation. This will be updated for  the full release. To use write the required function and import
them into the script.  

## Parametrization

Each training run is parametrized by 5 functions, which take two configurations. Functions are used to define the input,
model architecture, training logic, reporting to W&B and storage logic. For the last three, we provide useful defaults
in `scripts/defaults` directory.

Additionally, two config files can be used, one for storing data and one for model parameters. They are used to abstract
dataset loading and creating model architecture from the code, to quickly iterate through different configurations.
Otherwise `pydentification` library can be used as collection of standalone components, which can be useful for various 
project related to neural system identification.

### Functions

* `input_fn` - function which takes configuration file and returns `pl.DataModule` subclass, typically one of data-modules provided by `pydentification`.
* `model_fn` - function which takes configuration file and returns `pl.LightningModule` and `pl.Trainer` instances.
* `train_fn` - function which takes model, trainer and the data-module and executes training code, typically inside the `Trainer`.
* `report_fn` - function, which takes model, trainer and the data-module, it should run predictions and store relevant metrics in W&B dashboard.
* `save_fn` - function takes in run name (given by W&B `id` or `name`) and model, it saves the model to the disk.

### Configurations

The entrypoint (both training and sweep) are parameterized by two configs, one of them is for the data settings and the
other for the experiment and model, which contains hyperparameters and training settings. 

The data config is stored in `YAML` and it is passed to the `input_fn` function. Not all parameters of the data-module
is stored in the data config, only the static values. The example config looks following:

```yaml
name: Dataset
path: data/dataset.csv
test_size: 10000
input_columns: [x]
output_columns: [y]
```

The experiment config looks in the following way.

```yaml
general:
  project: project-name
  n_runs: 1
  name: placeholder
training:
  n_epochs: 10
  patience: 1
  timeout: "00:00:01:00"
  batch_size: 32
  shift: 1
  validation_size: 0.1
model:
  model_name: MLP
  # generic parameter convention
  n_input_time_steps: 64
  n_output_time_steps: 1
  n_input_state_variables: 1
  n_output_state_variables: 1
  # neural network parameters
  n_hidden_layers: 2
  activation: relu
  n_hidden_time_steps: 32
  n_hidden_state_variables: 4
```

To use sweep add following section to the experiment config.

```yaml
sweep:
  name: sweep
  method: random
  metric: {name: test/root_mean_squared_error, goal: minimize}
sweep_parameters:
  # neural network
  n_hidden_layers: [1, 2, 3, 4, 5]
  n_hidden_time_steps: [32, 16, 8]
  n_hidden_state_variables: [1, 4, 8, 16]
  activation: [leaky_relu, relu, gelu, sigmoid, tanh]
```