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


def read_results():
    import pandas as pd
    return pd.read_csv(RESULTS / "cv_results.csv", dtype={"step": str})


def test_cv_results_steps():
    import numpy as np
    results = read_results()
    assert results.step.tolist()[:2] == ["0", "1"]
    assert np.isfinite(results.cv_rmse).all()
    scores = results.set_index("step").cv_rmse
    assert scores["1"] < scores["0"]


def test_submissions_valid():
    import numpy as np
    import pandas as pd
    sample = pd.read_csv(V2.parent / "house-prices-advanced-regression-techniques" / "sample_submission.csv")
    files = sorted((RESULTS / "submissions").glob("*.csv"))
    assert "s1_ridge_raw.csv" in [f.name for f in files]
    for path in files:
        sub = pd.read_csv(path)
        assert sub.columns.tolist() == ["Id", "SalePrice"], path.name
        assert len(sub) == 1459, path.name
        assert sub.Id.tolist() == sample.Id.tolist(), path.name
        assert np.isfinite(sub.SalePrice).all() and (sub.SalePrice > 0).all(), path.name
