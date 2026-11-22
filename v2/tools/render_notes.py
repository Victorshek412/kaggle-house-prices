"""Render v2/notes/templates/*.md into v2/notes/, running the Python blocks they contain.

Two kinds of runnable block:
- ```python run```   shown in the note, executed, and followed by its real output (printed text,
                     the value of the last line if it is an expression, and any figure drawn).
- ```python setup``` executed but not shown (for loading data the examples need).

All blocks of one template share a namespace and run in order, with cwd = v2/.

    python v2/tools/render_notes.py              # every template
    python v2/tools/render_notes.py part0 part2  # only templates whose name starts with these
"""
import ast
import contextlib
import io
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
TEMPLATES = V2 / "notes" / "templates"
IMAGES = V2 / "notes" / "images"
BLOCK = re.compile(r"```python (run|setup)\n(.*?)```", re.S)


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
    figure_count = 0
    text = template.read_text(encoding="utf-8")

    def replace(match):
        nonlocal figure_count
        kind, code = match.group(1), match.group(2)
        output = run_block(code, namespace)
        if kind == "setup":
            plt.close("all")
            return ""
        shown = f"```python\n{code}```"
        if output:
            shown += f"\n\n跑出嚟會見到：\n\n```text\n{output}\n```"
        for number in plt.get_fignums():
            figure_count += 1
            name = f"{template.stem}_{figure_count}.png"
            IMAGES.mkdir(parents=True, exist_ok=True)
            plt.figure(number).savefig(IMAGES / name, dpi=110, bbox_inches="tight")
            shown += f"\n\n跑出嚟嘅圖：\n\n![example figure](images/{name})"
        plt.close("all")
        return shown

    rendered = BLOCK.sub(replace, text)
    return re.sub(r"\n{3,}", "\n\n", rendered)


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    pd.set_option("display.max_columns", 10)
    os.chdir(V2)
    prefixes = sys.argv[1:]
    for template in sorted(TEMPLATES.glob("*.md")):
        if prefixes and not template.stem.startswith(tuple(prefixes)):
            continue
        target = V2 / "notes" / template.name
        target.write_text(render(template), encoding="utf-8")
        print("wrote", target.relative_to(V2.parent))
