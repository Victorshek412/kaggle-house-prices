# Kaggle House Prices

Predicting Ames house sale prices for the Kaggle competition
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## `v2/`: current work

This is a beginner-readable notebook that builds the solution one measured step at a time, from a mean baseline through feature engineering and model comparison to a blend.

- `v2/house_prices_v2.ipynb` is the notebook. It is generated from `v2/src/part*.py`.
- `v2/notes/` has one study note per Part. Part 0 covers the basics; Part 6 is the error analysis made after the first Kaggle results.
- `v2/results/` holds CV scores, figures and submission CSVs.

**On Kaggle:** create a notebook in the competition, upload `v2/house_prices_v2.ipynb` (or paste in its cells), and select **Run All**.

**Locally:** put `train.csv`, `test.csv`, `sample_submission.csv` and `data_description.txt` in `house-prices-advanced-regression-techniques/` at the repo root. Then run:

```sh
python3 -m venv .venv
.venv/bin/pip install -r v2/requirements.txt
.venv/bin/python v2/build.py --execute
.venv/bin/pytest v2/tests
```

## `v1_reference/`: earlier solution, kept for reference only

This is an engineered pipeline: 27 experiments, with a CV log-RMSE of 0.121 for its ElasticNet + CatBoost blend. It is not maintained, and its scripts expect to be run from inside that folder.

## Design and plan

- Design: `docs/superpowers/specs/2026-11-22-house-prices-redo-design.md`
- Plan: `docs/superpowers/plans/2026-11-22-house-prices-redo.md`
