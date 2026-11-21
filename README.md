# Kaggle House Prices

Predicts house sale prices using the Ames housing data. See the [notebook](House_Prices_Project.ipynb) for the analysis, model comparison and plots.

| Model | Local 5-fold log-RMSE |
| --- | ---: |
| Ridge baseline | 0.15091 |
| CatBoost | 0.12505 |
| ElasticNet + CatBoost (50/50) | 0.12117 |

The blend reduced validation error by 19.7% versus Ridge. Applying `log1p` to skewed inputs such as floor area was the most useful feature change in the Ridge experiments. See the [full results](results/experiments.csv) and [feature comparison](results/feature_ablation.csv).

## Run

Download `train.csv`, `test.csv`, `sample_submission.csv` and `data_description.txt` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Put them in `house-prices-advanced-regression-techniques/`, then run with Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh run_python.sh run_experiments.py
sh run_python.sh build_notebook.py --execute
```

On Kaggle, import the notebook, attach the competition data and select **Run All**.

## Predictions

- [Ridge baseline CSV](submissions/01_ridge_baseline.csv)
- [ElasticNet + CatBoost CSV](submissions/recommended.csv)

Kaggle scores and rank have yet to be recorded.
