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

```python
trainer = pl.Trainer(
    max_epochs=100,
    # example callback for changing n_forward_time_steps
    callbacks=[StepAutoRegressionLengthScheduler(step_size=5, gamma=2, verbose=True)],
    reload_dataloaders_every_n_epochs=1,  # reload dataloaders to get new n_forward_time_steps
)
```
