"""Small helpers for the runnable examples in v2/notes/templates (used by render_notes.py)."""
import ast
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
DATA = V2.parent / "house-prices-advanced-regression-techniques"


def load_data():
    """train and test, read exactly like the notebook does."""
    read = lambda name: pd.read_csv(DATA / name, keep_default_na=False, na_values=["NA", ""])
    return read("train.csv"), read("test.csv")


def load_defs(part, names, namespace):
    """Define selected top-level functions / assignments of a notebook part, without running it."""
    tree = ast.parse((V2 / "src" / part).read_text(encoding="utf-8"))
    chosen = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            chosen.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            chosen.append(node)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), part, "exec"), namespace)
