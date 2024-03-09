import logging
from functools import partial
from typing import Any

import click
import wandb
import yaml

from pydentification.experiment.parameters import left_dict_join, prepare_config_for_sweep


def run_sweep_step(project_name: str, dataset_config: dict[str, Any], experiment_config: dict[str, Any]) -> None:
    """
    Runs single experiment in sweep, for more on configuration details see configs/README.md

    :param project_name: name of W&B project
    :param dataset_config: static dataset configuration
    :param experiment_config: dynamic experiment configuration
    """
    with wandb.init(reinit=True):
        parameters = wandb.config

        wandb.log(experiment_config["model"])
        wandb.log(experiment_config["training"])

        try:
            data_parameters = left_dict_join(experiment_config["training"], experiment_config["model"])  # noqa: F821
            dm = input_fn(dataset_config, data_parameters)  # noqa: F821
            model, trainer = model_fn(project_name, experiment_config["training"], parameters)  # noqa: F821
            model, trainer = train_fn(model, trainer, dm)  # noqa: F821
            report_fn(model, trainer, dm)  # noqa: F821
            save_fn(wandb.run.id, model, trainer, dm)  # noqa: F821
        except Exception as e:
            logging.exception(e)  # log traceback, W&B can sometimes lose information
            raise ValueError("Experiment failed.") from e


@click.command()
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--experiment", type=click.Path(exists=True), required=True)
def main(data: str, experiment: str):
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    sweep_parameters = left_dict_join(experiment_config["sweep_parameters"], experiment_config["model"])
    sweep_config = prepare_config_for_sweep(experiment_config["sweep"], sweep_parameters)
    sweep_id = wandb.sweep(sweep_config, project=experiment_config["general"]["project"])

    run_sweep_fn = partial(run_sweep_step, experiment_config["general"]["project"], dataset_config, experiment_config)
    wandb.agent(
        sweep_id,
        function=run_sweep_fn,
        count=experiment_config["general"]["n_runs"],
        project=experiment_config["general"]["project"],
    )
