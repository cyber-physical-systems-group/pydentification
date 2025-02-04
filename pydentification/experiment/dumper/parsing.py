import ast
import textwrap


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
