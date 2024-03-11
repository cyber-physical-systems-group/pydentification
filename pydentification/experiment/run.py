import logging
from typing import Any

import click
import wandb
import yaml

from .context import RuntimeContext
from .parameters import left_dict_join


def run_training(
    context: RuntimeContext, project_name: str, dataset_config: dict[str, Any], experiment_config: dict[str, Any]
) -> None:
    """
    Runs single training without sweep parameters with all parameters used from static configuration file

    :param context: runtime context
    :param project_name: name of W&B project
    :param dataset_config: static dataset configuration
    :param experiment_config: dynamic experiment configuration
    """
    try:
        # some parameters needed for data module are in experiment_config/training, some in experiment_config/model
        data_parameters = left_dict_join(experiment_config["training"], experiment_config["model"])
        dm = context.input_fn(dataset_config, data_parameters)
        model, trainer = context.model_fn(project_name, experiment_config["training"], experiment_config["model"])
        model, trainer = context.train_fn(model, trainer, dm)
        context.report_fn(model, trainer, dm)
        context.save_fn(wandb.run.id, model)

    except Exception as e:
        logging.exception(e)  # log traceback, W&B can sometimes lose information
        raise ValueError("Experiment failed.") from e


@click.command()
@click.option("--data", type=click.Path(exists=True), required=True)
@click.option("--experiment", type=click.Path(exists=True), required=True)
def main(data: str, experiment: str, context: RuntimeContext):
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    with wandb.init(project=experiment_config["general"]["project"], name=experiment_config["general"]["name"]):
        wandb.log(experiment_config["model"])
        wandb.log(experiment_config["training"])
        run_training(context, experiment_config["general"]["project"], dataset_config, experiment_config)
