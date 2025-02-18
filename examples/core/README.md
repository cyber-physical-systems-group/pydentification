# Prediction Experiment

This directory contains example code for prediction experiments based on autonomous system. The code runs W&B sweep for
transformer model, changing different parameters and reporting results to W&B. 

### Running

To run the experiment, use the following command (assuming W&B is installed and logged in, for more details go to
sources section):

```bash
python -m examples.core.prediction
```

# Simulation

This example uses `pydentification` to run training with simulation experiment on example benchmark. 
Only single experiment is run and registered to W&B.

### Running

To run the experiment, use the following command (assuming W&B is installed and logged in, for more details go to
sources section):

```bash
python -m examples.core.simulation
```

### Experiment

This is the example for fully reproducible sweeps, storing snapshot of the code used to run the experiment in ZIP, 
alongside stand-alone function to re-create the model, init parameters in JSON and model weights in safe-tensors format.

It is based on prediction example.

### Sources

* [https://docs.wandb.ai/guides/sweeps](https://docs.wandb.ai/guides/sweeps)
* [https://docs.wandb.ai/guides/track/advanced/distributed-training](https://docs.wandb.ai/guides/track/advanced/distributed-training)
* [https://lightning.ai/docs/pytorch/stable/common/trainer.html](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
