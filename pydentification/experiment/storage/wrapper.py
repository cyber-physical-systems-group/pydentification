import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Callable, Literal

from pydentification.experiment.dumper import parsing


def dump(path: str | Path | Callable[[None], str], param_store: Literal["py", "json", "both"] = "py") -> Callable:
    """
    Decorator for dumping the source code of the decorated function to a file and saving the parameters as JSON.
    The source code is saved with used imports, decorators (except for this one), and optionally replaced parameters,
    so their values are stored in the file.

    :param path: path to the file where the source code will be saved or callable that returns the path (dynamic path)
    :param param_store: whether to store parameters, replace values in source code python file
                        or JSON file with name to value mapping, or both
    """

    def prepare_path(input_path: str | Path | Callable[[None], str]) -> Path:
        """
        Prepare input path for saving the source code and parameters,
        This method runs when the decorated function is called, allowing dynamic paths
        """
        if isinstance(path, str):
            save_path = Path(input_path)
        elif isinstance(path, Callable):
            save_path = Path(input_path())
        else:
            save_path = input_path

        if save_path.exists():  # do not overwrite existing files
            raise FileExistsError(f"File {save_path} already exists!")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    def outer(func: Callable) -> Callable:
        @wraps(func)
        def inner(*args, **kwargs):
            code_path = prepare_path(path)

            arguments = func.__code__.co_varnames[: func.__code__.co_argcount]  # noqa[E203]
            parameters = {name: value for name, value in zip(arguments, args)}
            parameters.update(kwargs)  # add keyword arguments

            source_code = inspect.getsource(func)
            with open(inspect.getfile(func), "r") as f:
                file_source_code = f.read()

            if param_store in {"py", "both"}:
                # optionally re-write function parameters with their values provided in the call
                source_code = parsing.replace_variables(source_code, variables=parameters)

            # remove this decorator from the source code
            source_code = parsing.remove_decorators(source_code, names={"dump"})
            imports_code = parsing.parse_imports(file_source_code)
            source_code = imports_code + "\n" + source_code
            source_code = parsing.format_code(source_code)

            with open(code_path.with_suffix(".py"), "w") as f:
                f.write(source_code)

            if param_store in {"json", "both"}:
                with open(code_path.with_suffix(".json"), "w") as f:
                    f.write(json.dumps(parameters, indent=4))

            return func(*args, **kwargs)

        return inner

    return outer
