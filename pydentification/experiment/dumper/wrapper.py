import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Callable, Literal

from pydentification.experiment.dumper import parsing


def dump(path: str | Path, param_store: Literal["py", "json", "both"] = "py") -> Callable:
    """
    Decorator for dumping the source code of the decorated function to a file and saving the parameters as JSON.
    The source code is saved with used imports, decorators (except for this one), and optionally replaced parameters,
    so their values are stored in the file.

    :param path: path to the file where the source code will be saved
    :param param_store: whether to store parameters, replace values in source code python file
                        or JSON file with name to value mapping, or both
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        raise FileExistsError(f"File {path} already exists!")

    def outer(func: Callable) -> Callable:
        @wraps(func)
        def inner(*args, **kwargs):
            arguments = func.__code__.co_varnames[: func.__code__.co_argcount]  # noqa[E203]
            parameters = {name: value for name, value in zip(arguments, args)}
            parameters.update(kwargs)  # add keyword arguments

            source_code = inspect.getsource(func)
            source_filename = inspect.getfile(func)
            imports = parsing.parse_imports(source_filename)

            if param_store in {"py", "both"}:
                # optionally re-write function parameters with their values provided in the call
                source_code = parsing.replace_variables(source_code, variables=parameters)

            # remove this decorator from the source code
            source_code = imports + "\n" + source_code
            source_code = parsing.format_code(source_code)

            with open(path.with_suffix(".py"), "w") as f:
                f.write(source_code)

            if param_store in {"json", "both"}:
                with open(path.with_suffix(".json"), "w") as f:
                    f.write(json.dumps(parameters, indent=4))

            return func(*args, **kwargs)
        return inner
    return outer
