import logging
from functools import partial
from typing import Any

import wandb
import yaml

from .context import RuntimeContext
from .parameters import left_dict_join, prepare_config_for_sweep


def run_training(
    runtime: RuntimeContext,
    project_name: str,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    model_config: dict[str, Any],
):
    """
    This function is used to run a single training experiment with given configuration. It contains the main
    experimentation logic and parameter passing.
    """
    for config in (dataset_config, model_config, training_config):
        if isinstance(config, dict):  # prevents logging parameters twice in sweep mode
            wandb.log(config)

    try:
        # merge static and dynamic parameters of dataset
        data_parameters = left_dict_join(training_config, model_config)
        # experiment flow
        dm = runtime.input_fn(dataset_config, data_parameters)
        model, trainer = runtime.model_fn(project_name, training_config, model_config)
        model, trainer = runtime.train_fn(model, trainer, dm)
        runtime.report_fn(model, trainer, dm)
        runtime.save_fn(wandb.run.id, model)

    except Exception as e:
        logging.exception(e)  # log traceback, W&B can sometimes lose information
        raise ValueError("Experiment failed.") from e


def run_sweep_step(
    runtime: RuntimeContext, project_name: str, dataset_config: dict[str, Any], experiment_config: dict[str, Any]
):
    with wandb.init(reinit=True):
        parameters = wandb.config  # dynamically generated model settings by W&B sweep

        run_training(
            runtime=runtime,
            project_name=project_name,
            dataset_config=dataset_config,
            model_config=parameters,
            training_config=experiment_config["training"],
        )


def run(data: str, experiment: str, runtime: RuntimeContext):
    """
    Run single experiment with given configuration.

    :param data: dataset configuration
    :param experiment: experiment configuration
    :param runtime: runtime context with code executing the training and all preparations
    """
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    project = experiment_config["general"]["project"]
    name = experiment_config["general"]["name"]

    with wandb.init(project=project, name=name):
        model_config = experiment_config["model"]
        training_config = experiment_config["training"]

        run_training(
            runtime=runtime,
            project_name=project,
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
        )


def sweep(data: str, experiment: str, runtime: RuntimeContext):
    """
    Run a sweep experiment with given configuration.

    :param data: dataset configuration
    :param experiment: experiment configuration
    :param runtime: runtime context with code executing the training and all preparations
    """
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    sweep_parameters = left_dict_join(experiment_config["sweep_parameters"], experiment_config["model"])
    sweep_config = prepare_config_for_sweep(experiment_config["sweep"], sweep_parameters)
    sweep_id = wandb.sweep(sweep_config, project=experiment_config["general"]["project"])

    run_sweep_fn = partial(
        run_sweep_step, runtime, experiment_config["general"]["project"], dataset_config, experiment_config
    )
    wandb.agent(
        sweep_id,
        function=run_sweep_fn,
        count=experiment_config["general"]["n_runs"],
        project=experiment_config["general"]["project"],
    )
