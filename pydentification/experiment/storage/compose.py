import importlib.util
import json
import os
import sys
import zipfile
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


def _safe_unzip(path: Path):
    if path.with_suffix("").exists():
        raise FileExistsError(f"Can't overwrite {path.stem}!")

    with zipfile.ZipFile(path, "r") as zip:
        zip.extractall(str(path.parent))


def compose_model(
    path: str | Path,
    name: str = "model_fn",
    parameters: str | Path | None = None,
    source: str | Path | None = None,
):
    """
    Compose model from dump, which will contain model generating function, JSON with its parameters and source code
    for module definitions (ZIP of entire `pydentification`).

    :param path: filesystem Path to the model generating function, which will be imported by `import_function_from_path`
    :param name: name of the function to be imported, default is `model_fn`
    :param parameters: filesystem Path to the JSON file with parameters, if None, empty dictionary will be used
    :param source: filesystem Path to the ZIP file with source code
                   if None imports are attempted from the current working directory.
    """
    if isinstance(source, str):
        source = Path(source)

    if source is not None:
        _safe_unzip(source)
        sys.path.append(str(source.parent))

    model_fn = _import_function_from_path(path, name)

    if parameters is not None:
        with open(parameters, "r") as f:
            parameters = json.load(f)
    else:
        parameters = {}

    model = model_fn(**parameters)

    if source is not None:
        sys.path.remove(str(source.parent))
    return model
