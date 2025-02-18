import ast
import re
import textwrap
from typing import Any

import autoflake
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

    # get function parameters and remove ones that are replaced from the definition
    original_params = {arg.arg for arg in function.args.args}
    params_to_remove = original_params.intersection(variables.keys())
    new_params = [arg for arg in function.args.args if arg.arg not in params_to_remove]

    function.args.args = new_params  # update function parameters in AST
    updated_code = ast.unparse(tree)

    if not function:
        raise ValueError("No function definition found in the code.")

    def replacer(matched: re.Match) -> str:
        var_name = matched.group(0)
        value = variables[var_name]  # replace name with corresponding value
        return repr(value) if isinstance(value, str) else str(value)

    # find function start position
    func_start = function.body[0].lineno - 1  # adjust line numbers
    # split code into lines and process only the function body
    lines = updated_code.splitlines()
    function_body_lines = lines[func_start:]
    replaced_body = [re.sub(pattern, replacer, line) for line in function_body_lines]

    return textwrap.dedent("\n".join(lines[:func_start] + replaced_body))  # ensure consistent indentation


def remove_decorators(code: str, names: set[str]) -> str:
    """Removes all decorators from function definitions in the given code."""

    class DecoratorRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = [
                decorator
                for decorator in node.decorator_list
                if not (isinstance(decorator, ast.Name) and decorator.id in names)
                and not (
                    isinstance(decorator, ast.Call)  # handle decorators with arguments
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id in names
                )
            ]

            return node

    tree = ast.parse(code)
    tree = DecoratorRemover().visit(tree)
    return ast.unparse(tree)


def parse_imports(code: str) -> str:
    tree = ast.parse(code)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node).strip())

    return "\n".join(imports)


def format_code(code: str) -> str:
    code = autoflake.fix_code(code, remove_all_unused_imports=True)
    code = isort.code(code)
    code = black.format_str(code, mode=black.Mode(line_length=120))
    return code
