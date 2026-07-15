# ==============================================================================
# File: ast_validator.py
# Description: Core module for SAIR Modular Arithmetic Challenge.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

import ast
import sys
import os

BANNED_MODULES = {'sympy', 'gmpy2', 'math'}

class SandboxValidator(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in BANNED_MODULES:
                self.violations.append(f"Line {node.lineno}: Banned module imported -> '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in BANNED_MODULES:
            self.violations.append(f"Line {node.lineno}: Banned module imported -> '{node.module}'")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # Check for modulo operator Mod()
        if isinstance(node.op, ast.Mod):
            # We must allow string formatting (e.g., "%s" % val), but ban arithmetic modulo.
            # A rigorous check would infer types, but for this sandbox simulator we strictly ban
            # the modulo operator entirely to enforce neural computation.
            if not (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)):
                self.violations.append(f"Line {node.lineno}: Banned operator used -> '%' (Modulo)")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in {'exec', 'eval'}:
                self.violations.append(f"Line {node.lineno}: Banned function called -> '{node.func.id}'")
        self.generic_visit(node)

def validate_file(filepath: str) -> bool:
    print(f"Validating {filepath} against AST Sandbox Rules...")
    if not os.path.exists(filepath):
        print("File not found!")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax Error in file: {e}")
        return False

    validator = SandboxValidator()
    validator.visit(tree)

    if validator.violations:
        print("FAIL: The following violations were found:")
        for v in validator.violations:
            print(f" - {v}")
        return False
    else:
        print("PASS: Code conforms to Sandbox AST rules. No banned operations detected.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ast_validator.py <path_to_script>")
        sys.exit(1)
        
    target = sys.argv[1]
    success = validate_file(target)
    sys.exit(0 if success else 1)
