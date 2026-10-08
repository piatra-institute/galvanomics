from __future__ import annotations

import ast
from pathlib import Path

import tissuebundle.reader

ALLOWED = {"__future__", "json", "pathlib", "typing", "numpy"}


def test_reader_imports_only_stdlib_and_numpy() -> None:
    tree = ast.parse(Path(tissuebundle.reader.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= ALLOWED, f"reader imports {sorted(modules - ALLOWED)}"
