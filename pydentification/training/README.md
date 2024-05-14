# Training

This package contains modules used for training any models,  using `lightning.pytorch`. The repository is structured, in
order to decouple model building blocks, defining forward pass etc. from the logic for training them. Not all models
can be used with all training modules, but most training modules is general. Currently implemented training modules:

* `BoundedSimulationTrainingModule` - kernel-regression supported training for SISO simulation models producing networks with bounded predictions
* `LightningPredictionTrainingModule` - autoregressive training for N-step ahead models
* `LightningSimulationTrainingModule` - training for N to M simulation models

## Wrappers

Utility wrappers for training modules, currently implemented:

* `CastWrapper` - torch type case before and after forward pass, used to make stacking modules in Sequential easier
* `ResidualConnectionWrapper` - Wrapper for torch module that adds residual connection to the input
* `AutoregressiveResidualWrapper` -  autoregressive residual wrapper for torch module, which can be used for prediction modelling.

Minimal example using wrappers, assumes input is sequence of integers, like discrete excitation ([0, 0, 1, 1, 0])
and output is some continuous value (the model is example )

```python
torch.nn.Sequential(
    CastWrapper(torch.nn.Linear(1, 20), in_dtype=torch.int64, out_dtype=torch.float32),
    ResidualConnectionWrapper(torch.nn.Linear(20, 20)),  # residual can only be defined for shape-preserving modules
    torch.nn.Linear(20, 1),
)
```

## Callbacks

Prediction datamodules support generating different lengths of target sequences. Callbacks defined in
`lighting/callbacks` sub-module can be used to dynamically change this value during training. To do this trainer
requires to be `LightningPredictionTrainingModule` using `PredictionDataModule`. 

Currently implemented callbacks are:
* `StepAutoRegressionLengthScheduler` - changes n_forward_time_steps every `step_size` epochs by multiplying it by `gamma`
* `MultiStepAutoRegressionLengthScheduler` - Increases the length of auto-regression by `gamma` once the number of epoch reaches one of the milestones.
* `IncreaseAutoRegressionLengthOnPlateau` - Increases the length of auto-regression by factor once the monitored quantity stops improving.
* `CyclicTeacherForcing` - Changes the teacher forcing status cyclically every `cycle_in_epochs` epochs.
* `CombinedAutoRegressionCallback` - Combined callback for auto-regression training, which changes: Auto-regression length, teacher forcing status and learning rate

```python
trainer = pl.Trainer(
    max_epochs=100,
    # example callback for changing n_forward_time_steps
    callbacks=[StepAutoRegressionLengthScheduler(step_size=5, gamma=2, verbose=True)],
    reload_dataloaders_every_n_epochs=1,  # reload dataloaders to get new n_forward_time_steps
)
```

### Combined AutoRegression Callback

Combined callback connects switching between decreasing the learning rate, toggling teacher forcing on and off and 
increasing the auto-regression length. It should not be used with other callbacks that change the same parameters, as
this might lead to some unexpected behaviour. 

The order of switches of the parameters is given as list (with repetitions possible) in the `cycles` parameter. It can
for example decrease learning rate 3 times, switch teacher forcing and increase auto-regression length. The callback
would cause the model to decrease the validation loss, but when it is no longer able to, the length of the sequence
would be increased (both training and validation), datamodule would be reloaded (with caching implemented on its side) 
and the validation would increase (due to harder problem and longer auto-regression). Learning rate can be reset to
initial value at this point (this is controlled by `reset_learning_rate` parameter). Combining those switches helps
auto-regression models to learn longer sequences, while still being able to learn the shorter ones.

```python
callback = callbacks.CombinedAutoRegressionCallback(
    cycles=["learning_rate", "learning_rate", "learning_rate", "teacher_forcing", "ar_length",],
    monitor="val_loss",
    patience=2,
    ar_length_factor=2,
    lr_factor=0.1,
    reset_learning_rate=True,
    max_length=float("inf"),
    verbose=True,
)
```
