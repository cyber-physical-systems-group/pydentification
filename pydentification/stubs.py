from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union, get_type_hints

Print = Callable[[str], None]


def cast_to_path(func: Callable):
    """
    Decorator to cast string arguments to Path in the function signature.
    Uses type hints to determine which arguments should be cast.
    """
    def cast(arg: Any, hint: Any) -> Any:
        if hint == Union[str, Path]:
            return Path(arg)
        return arg

    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        new_args = []
        new_kwargs = {}

        for arg, (name, hint) in zip(args, hints.items()):
            new_args.append(cast(arg, hint))
        for name, arg in kwargs.items():
            new_kwargs[name] = cast(arg, hint=hints.get(name))

        return func(*new_args, **new_kwargs)

    return wrapper
