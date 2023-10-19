# Simulation

This example uses `pydentification` to run training with simulation experiment on example benchmark. 
Only single experiment is run and registered to W&B.

### Running

To run the experiment, use the following command (assuming W&B is installed and logged in, for more details go to
sources section):

```bash
python -m examples.prediction.run
```

### Sources

* [https://docs.wandb.ai/guides/sweeps](https://docs.wandb.ai/guides/sweeps)
* [https://docs.wandb.ai/guides/track/advanced/distributed-training](https://docs.wandb.ai/guides/track/advanced/distributed-training)
* [https://lightning.ai/docs/pytorch/stable/common/trainer.html](https://lightning.ai/docs/pytorch/stable/common/trainer.html)