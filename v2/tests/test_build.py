import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build import parse_cells


def test_parse_cells():
    text = "# %% [markdown]\n# # Title\n# %%\nx = 1\n"
    assert parse_cells(text) == [("markdown", "# Title"), ("code", "x = 1")]


def test_parse_cells_keeps_blank_markdown_lines():
    text = "# %% [markdown]\n# para one\n#\n# para two\n"
    assert parse_cells(text) == [("markdown", "para one\n\npara two")]
