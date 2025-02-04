import ast
import re
import textwrap
from typing import Any

import black
import isort


def replace_variables(code: str, variables: dict[str, Any]) -> str:
    """
    Function parsing source code of some python function as string and replacing variables from dictionary with their
    values, given as dictionary. This is used to create self-contained code snippets for constructing torch models.

    :note: Variables are only replaced in the function body.
    """
    tree = ast.parse(code)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    pattern = r"\b(" + "|".join(re.escape(var) for var in variables.keys()) + r")\b"

    if not function:
        raise ValueError("No function definition found in the code.")

    def replacer(matched: re.Match) -> str:
        var_name = matched.group(0)
        value = variables[var_name]  # replace name with corresponding value
        return repr(value) if isinstance(value, str) else str(value)

    # find function start position
    func_start = function.body[0].lineno - 1  # adjust line numbers
    # split code into lines and process only the function body
    lines = code.splitlines()
    function_body_lines = lines[func_start:]
    replaced_body = [re.sub(pattern, replacer, line) for line in function_body_lines]

    return "\n".join(lines[:func_start] + replaced_body)  # join lines back together


def remove_decorators(code: str, names: set[str]) -> str:
    """Removes all decorators from function definitions in the given code."""

    class DecoratorRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # loop over all decorators and remove decorators from input set by name
            node.decorator_list = [d for d in node.decorator_list if d.id not in names]  # type: ignore
            return node

    tree = ast.parse(code)
    tree = DecoratorRemover().visit(tree)
    new_code = ast.unparse(tree)

    return textwrap.dedent(new_code)  # `textwrap` ensures consistent indentation


def format_code(code: str) -> str:
    sorted_code = isort.code(code)
    formatted_code = black.format_str(sorted_code, mode=black.Mode(line_length=120))
    return formatted_code
