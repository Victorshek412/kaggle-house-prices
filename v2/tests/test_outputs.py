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


def test_feature_steps():
    steps = read_results().step.tolist()
    expected = ["2", "3", "4", "5", "6"]
    assert [s for s in steps if s in expected] == expected
    for name in ("s2_new_features.csv", "s3_log_features.csv", "s4_no_outliers.csv"):
        assert (RESULTS / "submissions" / name).is_file(), name


def test_feature_functions(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    ns = run_part("part1_eda.py", monkeypatch)
    ns["OUT_DIR"] = tmp_path  # don't overwrite the real notebook outputs
    for part in ("part2_baseline.py", "part3_features.py"):  # quick: Ridge-only parts
        exec((V2 / "src" / part).read_text(), ns)
    rows = ns["train"].iloc[:2].copy()
    rows["GarageArea"] = [0, 400]
    rows["GarageType"] = pd.Series([None, None], index=rows.index, dtype=object)
    filled = ns["fill_missing_meaning"](rows)
    assert filled.GarageType.iloc[0] == "NoGarage"
    assert pd.isna(filled.GarageType.iloc[1])  # has a garage, type unknown: leave for the imputer
    assert filled.MSSubClass.dtype == object
    added = ns["add_new_features"](filled)
    assert (added.TotalSF == rows.GrLivArea + rows.TotalBsmtSF).all()


def test_outlier_step_changes_score():
    # Regression: Step 6 once had no effect because GrLivArea was already log-transformed.
    scores = read_results().set_index("step").cv_rmse
    assert abs(scores["6"] - scores["5"]) > 1e-6


def test_model_steps():
    scores = read_results().set_index("step").cv_rmse
    for step in ("7", "8", "9", "10", "11"):
        assert step in scores.index, step
        assert scores[step] < scores["1"], step
    assert (RESULTS / "submissions" / "s5_catboost.csv").is_file()
    assert (RESULTS / "figures" / "05_model_comparison.png").is_file()


def test_blend():
    scores = read_results().set_index("step").cv_rmse
    singles = scores[[s for s in ("7", "7b", "8", "9", "10", "11") if s in scores.index]]
    assert "12" in scores.index
    assert scores["12"] <= singles.min() + 0.002
    assert (RESULTS / "submissions" / "s6_blend.csv").is_file()
    for name in ("06_oof_correlation.png", "07_improvement_staircase.png"):
        assert (RESULTS / "figures" / name).is_file(), name
    header = (RESULTS / "kaggle_log.csv").read_text().splitlines()[0]
    assert header == "submission,step,cv_rmse,kaggle_score,submitted_on,notes"


def _parts_1_2(monkeypatch, tmp_path):
    ns = run_part("part1_eda.py", monkeypatch)
    ns["OUT_DIR"] = tmp_path
    exec((V2 / "src" / "part2_baseline.py").read_text().split("# %% [markdown]\n# ### Step 0")[0], ns)
    return ns


def test_record_rerun_replaces_row(monkeypatch, tmp_path):
    ns = _parts_1_2(monkeypatch, tmp_path)
    ns["record"]("1", "first", 0.2)
    ns["record"]("2", "second", 0.15)
    ns["record"]("1", "first again", 0.19)  # re-running a cell must not add a duplicate
    results = ns["results"]
    assert results.step.tolist() == ["1", "2"]
    assert results.cv_rmse.tolist() == [0.19, 0.15]


def test_missing_data_gives_clear_error(monkeypatch, tmp_path):
    import pytest
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="train.csv"):
        runpy.run_path(str(V2 / "src" / "part1_eda.py"))


def test_notebook_matches_source():
    import nbformat
    import sys
    sys.path.insert(0, str(V2))
    from build import parse_cells
    expected = [body for part in sorted((V2 / "src").glob("part*.py"))
                for _, body in parse_cells(part.read_text(encoding="utf-8"))]
    notebook = nbformat.read(V2 / "house_prices_v2.ipynb", as_version=4)
    assert [cell.source for cell in notebook.cells] == expected, "run: python v2/build.py --execute"


def _functions_from(part, names):
    """Define only the named top-level functions of a notebook part (without running its cells)."""
    import ast
    import numpy as np
    import pandas as pd
    tree = ast.parse((V2 / "src" / part).read_text(encoding="utf-8"))
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in chosen} == set(names), "function(s) missing from " + part
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), part, "exec"), namespace)
    return namespace


def test_part6_helpers():
    import numpy as np
    import pandas as pd
    ns = _functions_from("part6_error_analysis.py", ["clean_years", "add_huge_partial_flag"])
    frame = pd.DataFrame({"GarageYrBlt": [2207.0, 2008.0, 1990.0], "YrSold": [2007, 2007, 2008]})
    cleaned = ns["clean_years"](frame)
    assert np.isnan(cleaned.GarageYrBlt.iloc[0])          # a typo like 2207 becomes missing
    assert cleaned.GarageYrBlt.iloc[1:].tolist() == [2008.0, 1990.0]
    assert frame.GarageYrBlt.iloc[0] == 2207.0             # the input frame is not changed

    raw = pd.DataFrame({"GrLivArea": [5095, 5095, 1500], "SaleCondition": ["Partial", "Normal", "Partial"]})
    logged = pd.DataFrame({"GrLivArea": np.log1p(raw.GrLivArea)})  # like X_final after Step 5
    flagged = ns["add_huge_partial_flag"](logged, raw)
    assert flagged.HugePartial.tolist() == [1, 0, 0]


def test_step13_submission():
    assert "13" in read_results().step.tolist()
    assert (RESULTS / "submissions" / "s7_outlier_flag.csv").is_file()


def test_s7_lowers_huge_partial_prediction():
    import pandas as pd
    s6 = pd.read_csv(RESULTS / "submissions" / "s6_blend.csv").set_index("Id").SalePrice
    s7 = pd.read_csv(RESULTS / "submissions" / "s7_outlier_flag.csv").set_index("Id").SalePrice
    assert s7[2550] < 0.6 * s6[2550]
