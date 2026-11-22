# %% [markdown]
# ## Part 2: Baseline and cross-validation
#
# ### Why we need cross-validation (CV)
# We can't score ourselves on `test`: its prices are hidden, and Kaggle limits daily
# submissions. Scoring on the rows we trained on would be cheating, because the model has
# already seen the answers. So we split `train` into **5 folds**. Each fold takes a turn as
# "pretend test data", while a model is trained on the other 4. Every house is predicted exactly
# once by a model that never saw it. These are the **out-of-fold (OOF)** predictions, and their
# error is our **CV score**.
#
# Every step in this notebook uses the **same 5 folds** (`random_state=42`), so differences
# between steps come from our changes, not from luck in the split.
#
# ### Avoiding data leakage
# Some preprocessing *learns from the data*. Filling gaps with the median, scaling, and
# listing the categories for one-hot encoding all do. If we did those on all 1,460 rows before
# splitting, every fold's "unseen" houses would already have influenced the training. That is
# **data leakage**, and it makes CV look better than reality. Wrapping preprocessing and model
# in one sklearn `Pipeline` means each fold re-learns the preprocessing from its own training
# rows only.

# %%
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def make_linear_preprocessor():
    """Numbers: fill gaps with the median, then scale. Text: fill gaps, then one-hot encode."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])


def cv_evaluate(model, X, y, X_test, target="log", drop_outliers=False):
    """5-fold CV. Returns (cv_rmse, out-of-fold log predictions, averaged test log predictions)."""
    y = np.asarray(y, dtype=float)
    y_log = np.log(y)
    oof_log = np.zeros(len(X))
    test_log = np.zeros(len(X_test))
    for train_idx, valid_idx in kf.split(X):
        if drop_outliers:  # remove outliers from the training rows only; validation keeps them
            keep = ~((X.GrLivArea.values[train_idx] > 4000) & (y[train_idx] < 300000))
            train_idx = train_idx[keep]
        fold_model = clone(model)  # a fresh, untrained copy for every fold
        if target == "log":
            fold_model.fit(X.iloc[train_idx], y_log[train_idx])
            oof_log[valid_idx] = fold_model.predict(X.iloc[valid_idx])
            test_log += fold_model.predict(X_test) / kf.n_splits
        else:  # target == "raw": learn dollars, convert to log only for scoring
            fold_model.fit(X.iloc[train_idx], y[train_idx])
            floor = y[train_idx].min() / 2  # a raw-price model can predict <= 0; log needs > 0
            oof_log[valid_idx] = np.log(np.clip(fold_model.predict(X.iloc[valid_idx]), floor, None))
            test_log += np.log(np.clip(fold_model.predict(X_test), floor, None)) / kf.n_splits
    cv_rmse = np.sqrt(np.mean((y_log - oof_log) ** 2))
    return cv_rmse, oof_log, test_log


results = pd.DataFrame(columns=["step", "description", "cv_rmse"])


def record(step, description, cv_rmse):
    """Add one row to the results table, print the change, and save the table."""
    global results
    if len(results):
        change = cv_rmse - results.cv_rmse.iloc[-1]
        print(f"Step {step}: CV RMSE {cv_rmse:.5f} ({change:+.5f} vs previous step)")
    else:
        print(f"Step {step}: CV RMSE {cv_rmse:.5f}")
    row = pd.DataFrame([{"step": str(step), "description": description, "cv_rmse": cv_rmse}])
    results = row if results.empty else pd.concat([results, row], ignore_index=True)
    results.to_csv(OUT_DIR / "cv_results.csv", index=False)


def save_submission(name, test_log):
    """Turn log predictions back into dollars and write a Kaggle submission file."""
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(test_log)})
    assert len(submission) == 1459 and submission.Id.tolist() == sample.Id.tolist()
    assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice > 0).all()
    (OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_DIR / "submissions" / f"{name}.csv", index=False)
    print("saved", name + ".csv")

# %% [markdown]
# ### Step 0: predict the average price for every house
# This is the "no model" reference point. Any real model must beat it.

# %%
X = train.drop(columns=["Id", "SalePrice"])
y = train.SalePrice
X_test = test.drop(columns="Id")

score, _, _ = cv_evaluate(DummyRegressor(strategy="mean"), X, y, X_test, target="raw")
record("0", "Predict the mean price", score)

# %% [markdown]
# ### Step 1: Ridge regression on the raw data
# Ridge is linear regression with a penalty on large coefficients, $\alpha\sum_j \beta_j^2$.
# With about 270 columns after one-hot encoding and only 1,460 houses, plain linear
# regression would overfit. The penalty keeps the coefficients modest.
#
# Here we train on **raw dollar prices** with only the minimum preprocessing. Every later step
# changes one thing and is compared with this.

# %%
ridge_raw = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=10))])
score, oof_step1, test_step1 = cv_evaluate(ridge_raw, X, y, X_test, target="raw")
record("1", "Ridge, raw price, minimal preprocessing", score)
save_submission("s1_ridge_raw", test_step1)

# %% [markdown]
# ### Reading the numbers
# A CV RMSE of 0.40 (Step 0) means a typical miss of about $e^{0.40}-1 \approx 49\%$.
# Ridge cuts that to about 0.155, a typical miss of roughly 17%. But Step 1 still trains on
# dollars while being scored in logs. Nothing stops a dollar model from predicting a price of
# zero or less, so `cv_evaluate` puts a floor under raw predictions. It wasn't triggered here,
# but it guards against a crash. Part 3 fixes this mismatch first.

# %%
results
