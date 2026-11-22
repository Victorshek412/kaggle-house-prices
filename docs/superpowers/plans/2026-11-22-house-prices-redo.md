# House Prices Redo (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, beginner-readable Kaggle notebook (`v2/house_prices_v2.ipynb`) that walks from a mean baseline to a blended model in 12 measured steps, plus one Cantonese learning note per Part.

**Architecture:** Notebook source lives as five plain `# %%`-cell Python files (one per Part) that a small build script stitches into one `.ipynb` and optionally executes locally. Every step's CV score goes into one shared results table. Row-wise feature code is plain pandas; anything that learns from column statistics lives inside an sklearn `Pipeline`, so it is fit per fold.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, xgboost, catboost, matplotlib, nbformat/nbclient, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-house-prices-redo-design.md`

## Global Constraints

- Notebook must run unchanged on Kaggle: data at `/kaggle/input/house-prices-advanced-regression-techniques/`, outputs to `/kaggle/working/`. Locally it falls back to `house-prices-advanced-regression-techniques/` (repo root) and `v2/results/`.
- The notebook never imports repo `.py` files; everything it needs is defined in its own cells.
- Read CSVs with `keep_default_na=False, na_values=["NA", ""]` (keeps `MasVnrType == "None"` as a category).
- CV: `KFold(n_splits=5, shuffle=True, random_state=42)` for every step. The metric is RMSE between log(true price) and log(predicted price).
- Test predictions = mean of the 5 fold models' log predictions, exponentiated once.
- Outlier rule (Step 6 onward): drop training-fold rows with `GrLivArea > 4000 and SalePrice < 300000`. Validation rows are always scored.
- Notebook text and code comments are in English. Learning notes (`v2/notes/*.md`) are in Cantonese and follow the 4-part format: 概念 / 逐行解釋 / 點睇結果 / 講師可能會問.
- Total notebook runtime on Kaggle should stay under ~10 minutes. Keep CatBoost/XGBoost iteration counts modest.
- Competition data is never committed (already in `.gitignore`).
- Hand off one Part at a time. After each Part task, stop and give the student the cells and notes. Do not start the next Part until the student says they understand.

## Review Focus

1. **Test rows with missing values that train never has** (e.g. `MSZoning`, `BsmtFullBath`, `GarageCars` in test). Expect finite predictions for all 1459 rows. Pinned by `test_submissions_valid` (Task 3+).
2. **Categories in test that never appear in train** (e.g. `MSSubClass` 150). Expect one-hot to ignore them and no crash. Covered by the same test.
3. **Raw-price Ridge (Step 1) predicting ≤ 0.** Expect predictions to be clipped to a positive floor before the log. Pinned by `test_submissions_valid` on `s1_ridge_raw.csv`.
4. **Notebook executed outside Kaggle.** Expect the path fallback to work, not a `FileNotFoundError`. Pinned by every task's build-and-execute step.
5. **The `"None"` string in `MasVnrType` read as missing.** Expect it to stay a category. Pinned by `test_none_category_kept` (Task 2).

---

## File Structure

```
v1_reference/                 old solution, moved with git mv, unchanged
v2/
  build.py                    stitches src/part*.py → house_prices_v2.ipynb; --execute runs it
  src/part1_eda.py … part5_blend.py   notebook cells ("# %%" code, "# %% [markdown]" text)
  house_prices_v2.ipynb       generated, committed (what the student uploads/copies to Kaggle)
  notes/part1_eda.md … part5_blend.md
  results/                    cv_results.csv, figures/, submissions/, kaggle_log.csv
  tests/test_build.py, tests/test_outputs.py
README.md                     rewritten: what v1 and v2 are
```

Shared notebook names, defined once and reused by later Parts:

| Name | Defined in | Signature / meaning |
|---|---|---|
| `DATA_DIR`, `OUT_DIR` | Part 1 | `Path`, with Kaggle-first fallback |
| `train`, `test`, `sample` | Part 1 | DataFrames |
| `kf` | Part 2 | the KFold above |
| `make_linear_preprocessor()` | Part 2 | → `ColumnTransformer` (numeric: median impute + StandardScaler; categorical: constant `"Missing"` + `OneHotEncoder(handle_unknown="ignore")`) |
| `cv_evaluate(model, X, y, X_test, target="log", drop_outliers=False)` | Part 2 | → `(cv_rmse: float, oof_log: np.ndarray, test_log: np.ndarray)`; clones the model per fold; for `target="raw"` it fits on price and clips predictions to ≥ 1 before the log |
| `record(step, description, cv_rmse)` | Part 2 | appends to global `results` DataFrame, prints the change versus the previous row, writes `OUT_DIR/cv_results.csv` |
| `save_submission(name, test_log)` | Part 2 | writes `OUT_DIR/submissions/{name}.csv` with columns `Id,SalePrice`; asserts 1459 rows, Ids equal `sample.Id`, all prices finite and > 0 |
| `fill_missing_meaning(df)`, `add_new_features(df)`, `log_skewed(df)` | Part 3 | DataFrame → new DataFrame, row-wise only |
| `X_final`, `X_test_final` | Part 3 | feature matrix after Step 6 settings, reused by Parts 4–5 |
| `oof`, `test_preds` | Part 4 | `dict[str, np.ndarray]` of each model's OOF / test log predictions |

---

### Task 1: Restructure folders and build script

**Files:**
- Move: every tracked top-level item except `docs/`, `.gitignore`, `README.md` → `v1_reference/`
- Create: `v2/build.py`, `v2/tests/test_build.py`, `README.md` (rewrite)

**Interfaces:**
- Produces: `python v2/build.py [--execute]` builds `v2/house_prices_v2.ipynb` from `v2/src/part*.py` in name order. A line `# %%` starts a code cell; `# %% [markdown]` starts a markdown cell whose lines have their leading `# ` stripped.

- [ ] **Step 1:** Set up the venv (`python3 -m venv .venv && .venv/bin/pip install -r v1_reference/requirements.txt pytest`) after the `git mv`.
- [ ] **Step 2: Write the failing test** `test_parse_cells`: for source `"# %% [markdown]\n# # Title\n# %%\nx = 1\n"`, `parse_cells(text)` returns `[("markdown", "# Title"), ("code", "x = 1")]`.
- [ ] **Step 3:** Run `.venv/bin/pytest v2/tests/test_build.py -v`. Expected: FAIL (no module).
- [ ] **Step 4:** Implement `parse_cells(text: str) -> list[tuple[str, str]]` and `main()` in `v2/build.py`. `--execute` runs the notebook with nbclient, using cwd `v2/` and a 1200 s timeout.
- [ ] **Step 5:** Run the test. Expected: PASS.
- [ ] **Step 6:** Rewrite `README.md`: two short sections, v1 (old reference) and v2 (current work, how to run on Kaggle and locally).
- [ ] **Step 7:** Commit with message `Move v1 to v1_reference; add v2 notebook build script`.

### Task 2: Part 1 — EDA

**Files:**
- Create: `v2/src/part1_eda.py`, `v2/notes/part1_eda.md`
- Test: `v2/tests/test_outputs.py`

**Interfaces:**
- Produces: `DATA_DIR`, `OUT_DIR`, `train`, `test`, `sample`; figures in `OUT_DIR/figures/`.

- [ ] **Step 1: Write the failing test** `test_none_category_kept`: after execution, `v2/results/figures/` contains `01_saleprice_distribution.png`, `02_missing_values.png`, `03_outliers.png`, `04_top_correlations.png`. Also run `part1_eda.py`'s loading code through `runpy` with the local path and assert `"None" in train.MasVnrType.unique()` and `train.shape == (1460, 81)`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Write Part 1 cells:
  - title and intro (the task, the metric, the Kaggle grading);
  - setup with path fallback and the loader;
  - shape and dtypes summary;
  - price distribution: raw vs log histogram, with skewness printed;
  - missing-value bar chart, split into "missing means absent" and "truly unknown";
  - GrLivArea vs SalePrice scatter with the two outliers circled;
  - top 10 numeric correlations with SalePrice;
  - a closing markdown cell, "What EDA tells us to do next", linking each finding to Steps 2, 3, 5 and 6.
- [ ] **Step 4:** Run `.venv/bin/python v2/build.py --execute && .venv/bin/pytest v2/tests -v`. Expected: PASS.
- [ ] **Step 5:** Write `v2/notes/part1_eda.md` in the 4-part format. Quote the actual numbers from the executed run (skewness, missing counts).
- [ ] **Step 6:** Commit with message `v2 Part 1: EDA`. **Stop and hand off to the student.**

### Task 3: Part 2 — Baseline and shared helpers

**Files:**
- Create: `v2/src/part2_baseline.py`, `v2/notes/part2_baseline.md`
- Modify: `v2/tests/test_outputs.py`

**Interfaces:**
- Consumes: Part 1 names.
- Produces: `kf`, `make_linear_preprocessor`, `cv_evaluate`, `record`, `save_submission`, `results`.

- [ ] **Step 1: Write the failing tests**
  - `test_cv_results_steps`: `cv_results.csv` steps begin with `["0", "1"]`, every `cv_rmse` is finite, and step 1 < step 0.
  - `test_submissions_valid`: for every CSV in `results/submissions/`, there are 1459 rows, `Id` equals the sample's `Id`, and `SalePrice` is finite and > 0. It must include `s1_ridge_raw.csv`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write Part 2 cells:
  - a markdown cell on what CV is and why the test set can't be used;
  - the helper definitions above;
  - Step 0, `DummyRegressor(strategy="mean")` on raw price, through `cv_evaluate(target="raw")`;
  - Step 1, `Pipeline([preprocessor, Ridge(alpha=10)])` with `target="raw"`, then `save_submission("s1_ridge_raw", ...)`;
  - a markdown cell explaining what the numbers mean.
- [ ] **Step 4:** Build, execute and run the tests. Expected: PASS.
- [ ] **Step 5:** Write the notes. They must cover data leakage, and why the preprocessor sits inside the Pipeline.
- [ ] **Step 6:** Commit with message `v2 Part 2: baseline and CV helpers`. **Stop and hand off.**

### Task 4: Part 3 — Feature engineering (Steps 2–6)

**Files:**
- Create: `v2/src/part3_features.py`, `v2/notes/part3_features.md`
- Modify: `v2/tests/test_outputs.py`

**Interfaces:**
- Consumes: Part 2 helpers.
- Produces: `fill_missing_meaning`, `add_new_features`, `log_skewed`, `X_final`, `X_test_final`.

- [ ] **Step 1: Write the failing test** `test_feature_steps`:
  - steps 2–6 are present in order;
  - submissions `s2_new_features.csv`, `s3_log_features.csv` and `s4_no_outliers.csv` exist;
  - add a unit check via `runpy` on a 2-row frame: a row with `GarageArea == 0` and NaN `GarageType` gets `"NoGarage"`, and `TotalSF == GrLivArea + TotalBsmtSF`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write the cells:
  - Step 2: same Ridge, `target="log"`.
  - Step 3: `fill_missing_meaning`, covering basement, garage, fireplace, pool, alley, fence and misc. The "absent" fill applies only when the matching size/count column is 0; otherwise the value stays NaN for the Pipeline to impute.
  - Step 4: `add_new_features`, adding TotalSF, TotalBathrooms, HouseAge, RemodelAge and TotalPorchSF. Save `s2_new_features`.
  - Step 5: `log_skewed`, applying log1p to area columns whose train skew is > 0.75. Choose that column list once from the training data and hard-code it into the function, and explain this choice in markdown. Save `s3_log_features`.
  - Step 6: `drop_outliers=True`. Save `s4_no_outliers`.
  - Re-tune Ridge alpha over `[1, 3, 10, 30]` at Step 6 and use the best value.
- [ ] **Step 4:** Build, execute and run the tests. Expected: PASS.
- [ ] **Step 5:** Write the notes. "點睇結果" should cover the student's most-useful-feature question, using the real numbers.
- [ ] **Step 6:** Commit with message `v2 Part 3: feature engineering steps 2-6`. **Stop and hand off.**

### Task 5: Part 4 — Model comparison (Steps 7–11)

**Files:**
- Create: `v2/src/part4_models.py`, `v2/notes/part4_models.md`
- Modify: `v2/tests/test_outputs.py`

**Interfaces:**
- Consumes: `X_final`, `X_test_final`, helpers.
- Produces: `oof`, `test_preds` with keys `"ridge"`, `"lasso"`, `"elasticnet"`, `"rf"`, `"xgb"`, `"catboost"` and `"svr"`.

- [ ] **Step 1: Write the failing test** `test_model_steps`: steps 7–11 are present; `s5_catboost.csv` exists; every model's CV < step 1's CV.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write the cells, one per model, each with its own markdown explaining the idea:
  - Step 7: Ridge, Lasso and ElasticNet, each tuned over a small alpha grid with `cv_evaluate`.
  - Step 8: `RandomForestRegressor(n_estimators=300, n_jobs=-1)` on the one-hot data.
  - Step 9: XGBoost (`max_depth=3`, `learning_rate=0.03`, about 1500 trees, subsample 0.8).
  - Step 10: CatBoost with native categoricals. Fill categorical NaN with `"Missing"`. Use the unlogged features from Steps 3–4, because trees don't need log. Save `s5_catboost`.
  - Step 11: SVR (RBF) inside the linear preprocessor pipeline, with a small `C`/`epsilon` grid.
  - End with a comparison bar chart, `figures/05_model_comparison.png`.
- [ ] **Step 4:** Build, execute and run the tests. Expected: PASS. Note the runtime of each model in the notes.
- [ ] **Step 5:** Write the notes. Cover bagging vs boosting, why trees don't need scaling or log, and why SVR needs scaling.
- [ ] **Step 6:** Commit with message `v2 Part 4: model comparison`. **Stop and hand off.**

### Task 6: Part 5 — Blend and summary figures

**Files:**
- Create: `v2/src/part5_blend.py`, `v2/notes/part5_blend.md`, `v2/results/kaggle_log.csv` (header only)
- Modify: `v2/tests/test_outputs.py`

**Interfaces:**
- Consumes: `oof`, `test_preds`, `results`.

- [ ] **Step 1: Write the failing test** `test_blend`:
  - step 12 is present, and its CV ≤ the best single model's CV + 0.002;
  - `s6_blend.csv` exists;
  - the figures `06_oof_correlation.png` and `07_improvement_staircase.png` exist;
  - `kaggle_log.csv` has the header `submission,step,cv_rmse,kaggle_score,submitted_on,notes`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Write the cells:
  - a heatmap of the correlation between OOF residuals;
  - pick 2–4 models with low CV and lower residual correlation, and blend them with equal weights. State the picks in markdown with the reason;
  - save `s6_blend`;
  - a staircase chart of every step's CV;
  - a final summary table that maps each submission file to its step and CV, ready for the student to fill in the Kaggle scores.
- [ ] **Step 4:** Build, execute and run the full test suite. Expected: all PASS.
- [ ] **Step 5:** Write the notes. Cover why blending works and why the weights aren't tuned.
- [ ] **Step 6:** Commit with message `v2 Part 5: blend and summary` and push. **Hand off.**
