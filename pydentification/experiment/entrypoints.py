import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import wandb
import yaml

from ..stubs import cast_to_path
from .context import RuntimeContext
from .parameters import left_dict_join, prepare_config_for_sweep
from .storage.code import save_code_snapshot


@cast_to_path
def create_code_snapshot(output_dir_name: str | Path, experiment_config: dict[str, Any]):
    """
    Creates the code snapshot for the experiment using experiment config file `storage` key.
    Uses shared output directory for all experiments and list of source code directories from settings.
    """
    for settings in experiment_config["storage"].get("source_code", []):  # skip if key not present
        os.makedirs(settings["target_dir"], exist_ok=True)
        target_path = output_dir_name / settings["target_dir"]
        save_code_snapshot(name=settings["name"], source_dir=settings["source_dir"], target_dir=target_path)


def run_training(
    runtime: RuntimeContext,
    project_name: str,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    checkpoint_path: str | None = None,
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
        model, trainer = runtime.model_fn(project_name, training_config, model_config, checkpoint_path)
        model, trainer = runtime.train_fn(model, trainer, dm, checkpoint_path)
        runtime.report_fn(model, trainer, dm)
        runtime.save_fn(runtime.output_dir_name_fn(), model)

    except Exception as e:
        logging.exception(e)  # log traceback, W&B can sometimes lose information
        raise ValueError("Experiment failed.") from e


def run_sweep_step(
    runtime: RuntimeContext, project_name: str, dataset_config: dict[str, Any], experiment_config: dict[str, Any]
):
    with wandb.init(reinit=True):
        wandb.mark_preempting()
        parameters = dict(wandb.config)  # dynamically generated model settings by W&B sweep
        create_code_snapshot(runtime.output_dir_name_fn(), experiment_config)

        run_training(
            runtime=runtime,
            project_name=project_name,
            dataset_config=dataset_config,
            model_config=parameters,
            training_config=experiment_config["training"],
        )


def run(data: str, experiment: str, resume: str, runtime: RuntimeContext):
    """
    Run single experiment with given configuration.

    :param data: dataset configuration
    :param experiment: experiment configuration
    :param runtime: runtime context with code executing the training and all preparations
    :param resume: resume training from a given run_id, this needs to load existing model from checkpoint and set
                   the training state and optimizer state correctly, otherwise unexpected behavior may occur
    """
    dataset_config = yaml.safe_load(open(data))
    experiment_config = yaml.safe_load(open(experiment))

    if resume:
        resume_config = yaml.safe_load(open(resume))
    else:
        resume_config = {}

    project = experiment_config["general"]["project"]
    name = experiment_config["general"]["name"]
    resume = resume_config.get("resume_mode")  # will be None if resume not used
    run_id = resume_config.get("run_id")

    with wandb.init(project=project, name=name, resume=resume, id=run_id):
        model_config = experiment_config["model"]
        training_config = experiment_config["training"]

        create_code_snapshot(runtime.output_dir_name_fn(), experiment_config)

        run_training(
            runtime=runtime,
            project_name=project,
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path=resume_config.get("checkpoint_path"),
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
