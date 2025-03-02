from functools import wraps
from pathlib import Path
from typing import Callable, Union, get_type_hints

Print = Callable[[str], None]


def cast_to_path(func: Callable):
    """
    Decorator to cast string arguments to Path in the function signature.
    Uses type hints to determine which arguments should be cast.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        cast_args = []
        for arg, (name, hint) in zip(args, hints.items()):
            if hint == Union[str, Path]:
                if isinstance(arg, str):
                    cast_args.append(Path(arg))
            else:
                cast_args.append(arg)
        new_kwargs = {}
        for name, arg in kwargs.items():
            if hints.get(name) == Union[str, Path]:
                new_kwargs[name] = Path(arg)
            else:
                new_kwargs[name] = arg
        return func(*cast_args, **new_kwargs)

    return wrapper
