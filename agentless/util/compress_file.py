import libcst as cst
import libcst.matchers as m


class CompressTransformer(cst.CSTTransformer):
    DESCRIPTION = "Replaces function body with ... while preserving docstrings"
    replacement_string = '"$$FUNC_BODY_REPLACEMENT_STRING$$"'

    def __init__(self, keep_constant=True):
        self.keep_constant = keep_constant

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_body = [
            stmt
            for stmt in updated_node.body
            if m.matches(stmt, m.ClassDef())
            or m.matches(stmt, m.FunctionDef())
            or (
                self.keep_constant
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Assign())
            )
            or (
                m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            )
        ]
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        new_body = []
        if updated_node.body.body:
            # Preserve class docstring if present
            if isinstance(updated_node.body.body[0], cst.SimpleStatementLine):
                first_stmt = updated_node.body.body[0].body[0]
                if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
                    new_body.append(updated_node.body.body[0])  # Keep the docstring

            # Process class methods
            for stmt in updated_node.body.body:
                if isinstance(stmt, cst.FunctionDef):
                    new_body.append(self.process_function_def(stmt))

        if not new_body:
            new_body.append(cst.SimpleStatementLine(body=[cst.Expr(value=cst.Ellipsis())]))
        
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return self.process_function_def(updated_node)

    def process_function_def(self, node: cst.FunctionDef) -> cst.FunctionDef:
        new_body = []
        if node.body.body:
            # Preserve function docstring if present
            if isinstance(node.body.body[0], cst.SimpleStatementLine):
                first_stmt = node.body.body[0].body[0]
                if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
                    new_body.append(node.body.body[0])  # Keep the docstring

        if not new_body:
            new_body.append(cst.SimpleStatementLine(body=[cst.Expr(value=cst.Ellipsis())]))

        return node.with_changes(body=cst.IndentedBlock(body=new_body))


def get_skeleton(raw_code, keep_constant: bool = True):
    try:
        tree = cst.parse_module(raw_code)
    except:
        return raw_code

    transformer = CompressTransformer(keep_constant=keep_constant)
    modified_tree = tree.visit(transformer)
    code = modified_tree.code
    code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
    code = code.replace(CompressTransformer.replacement_string, "...\n")
    return code


# Test code and function remain unchanged
code = """
\"\"\"
this is a module
...
\"\"\"
const = {1,2,3}
import os

class fooClass:
    '''this is a class'''

    def __init__(self, x):
        '''initialization.'''
        self.x = x

    def print(self):
        print(self.x)

def test():
    a = fooClass(3)
    a.print()

"""

def test_compress():
    skeleton = get_skeleton(code, True)
    print(skeleton)


if __name__ == "__main__":
    test_compress()