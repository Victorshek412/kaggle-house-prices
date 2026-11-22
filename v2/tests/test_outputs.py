"""Checks on what the executed notebook produces. Run `python v2/build.py --execute` first."""
import os
import runpy
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
RESULTS = V2 / "results"


def run_part(name, monkeypatch):
    monkeypatch.chdir(V2)
    os.environ.setdefault("MPLBACKEND", "Agg")
    return runpy.run_path(str(V2 / "src" / name))


def test_eda_figures_exist():
    for name in ("01_saleprice_distribution.png", "02_missing_values.png",
                 "03_outliers.png", "04_top_correlations.png"):
        assert (RESULTS / "figures" / name).is_file(), name


def test_none_category_kept(monkeypatch):
    ns = run_part("part1_eda.py", monkeypatch)
    train = ns["train"]
    assert train.shape == (1460, 81)
    assert "None" in train.MasVnrType.unique()
    assert ns["test"].shape == (1459, 80)
