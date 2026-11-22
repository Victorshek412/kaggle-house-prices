"""Render v2/notes/templates/*.md into v2/notes/, running every ```python run``` block.

Each run block is executed in one shared namespace (in order, with cwd = v2/), and its real
output is written underneath it, like a notebook: printed text plus the value of the last line
if that line is an expression.

    python v2/tools/render_notes.py
"""
import ast
import contextlib
import io
import os
import re
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
TEMPLATES = V2 / "notes" / "templates"
BLOCK = re.compile(r"```python run\n(.*?)```", re.S)


def run_block(code, namespace):
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    body = tree.body[:-1] if last else tree.body
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(ast.Module(body=body, type_ignores=[]), "<note>", "exec"), namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<note>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    return buffer.getvalue().rstrip("\n")


def render(template):
    namespace = {}
    text = template.read_text(encoding="utf-8")

    def replace(match):
        code = match.group(1)
        output = run_block(code, namespace)
        shown = f"```python\n{code}```"
        if output:
            shown += f"\n\n跑出嚟會見到：\n\n```text\n{output}\n```"
        return shown

    return BLOCK.sub(replace, text)


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    pd.set_option("display.max_columns", 8)
    os.chdir(V2)
    for template in sorted(TEMPLATES.glob("*.md")):
        target = V2 / "notes" / template.name
        target.write_text(render(template), encoding="utf-8")
        print("wrote", target.relative_to(V2.parent))
