import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable


def _import_function_from_path(module_path: str, function_name: str) -> Callable:
    """Dynamically imports a function from a Python file given the file path and function name."""
    module_name = os.path.basename(module_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    function = getattr(module, function_name)

    return function


def compose_model(path: str | Path, name: str = "model_fn", parameters: str | Path | None = None):
    """
    Compose model from dump, which will contain model generating function, JSON with its parameters and source code
    for module definitions (ZIP of entire `pydentification`).

    :param path: filesystem Path to the model generating function, which will be imported by `import_function_from_path`
    :param name: name of the function to be imported, default is `model_fn`
    :param parameters: filesystem Path to the JSON file with parameters, if None, empty dictionary will be used
    """
    model_fn = _import_function_from_path(path, name)

    if parameters is not None:
        with open(parameters, "r") as f:
            parameters = json.load(f)
    else:
        parameters = {}

    return model_fn(parameters)
