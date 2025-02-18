import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable


class ReplaceSourceCode:
    """
    ContextManager over-writing imports with given `path` to ZIP with source code created by
    `pydentification.experiment.storage.code.save_code_snapshot`.

    Code is extracted to a temporary directory and added to the `sys.path` for the duration of the context and removed
    afterward on exit. The source code needs to be unique directory to avoid conflicts with other imports.
    """

    def __init__(self, path: Path):
        self.path = path
        self.source_path = path.with_suffix("")

    def __enter__(self):
        if self.source_path.exists():
            raise FileExistsError(f"Can't overwrite {self.source_path.stem}!")

        with zipfile.ZipFile(self.path, "r") as zip:
            zip.extractall(str(self.source_path))

        sys.path.append(str(self.source_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path.remove(str(self.source_path))
        shutil.rmtree(self.source_path)


def _import_function_from_path(module_path: str, function_name: str) -> Callable:
    """Dynamically imports a function from a Python file given the file path and function name."""
    module_name = os.path.basename(module_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    function = getattr(module, function_name)

    return function


def _load_model_and_parameters(path: str | Path, name: str, parameters: dict[str, Any]) -> Any:
    model_fn = _import_function_from_path(path, name)
    return model_fn(**parameters)


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

    if parameters is not None:
        with open(parameters, "r") as f:
            parameters = json.load(f)
    else:
        parameters = {}

    if source is not None:
        with ReplaceSourceCode(source):
            return _load_model_and_parameters(path, name, parameters)
    else:
        return _load_model_and_parameters(path, name, parameters)
