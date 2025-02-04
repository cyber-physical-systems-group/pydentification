import ast
import textwrap

import black
import isort


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
