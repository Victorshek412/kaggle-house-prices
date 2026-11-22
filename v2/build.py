"""Stitch v2/src/part*.py into house_prices_v2.ipynb; --execute also runs it.

Cell markers: a line "# %%" starts a code cell, "# %% [markdown]" starts a
markdown cell whose lines have their leading "# " removed.
"""
import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "house_prices_v2.ipynb"


def parse_cells(text):
    cells, kind, lines = [], None, []

    def flush():
        body = "\n".join(lines).strip("\n")
        if kind and body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        else:
            lines.append(line)
    flush()
    return cells


def build(execute=False):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for part in sorted((HERE / "src").glob("part*.py")):
        for kind, body in parse_cells(part.read_text(encoding="utf-8")):
            new = nbf.v4.new_markdown_cell if kind == "markdown" else nbf.v4.new_code_cell
            nb.cells.append(new(body))
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1200, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the notebook after building it")
    print("wrote", build(parser.parse_args().execute))
