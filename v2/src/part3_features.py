# %% [markdown]
# ## Part 3: Feature engineering, one change at a time
#
# Each step below changes **exactly one thing** and re-scores with the same 5 folds. That way,
# the change in CV RMSE can be credited to that one change.
#
# Row-wise feature code (sums, differences, relabelling) is plain pandas. It looks at one house
# at a time and never at column statistics, so it can't leak information between folds. That
# means it is safe to apply to `train` and `test` before CV. Anything that *learns* from a
# column (medians, scaling, category lists) stays inside the Pipeline from Part 2.
#
# One deliberate exception: in Step 5 we choose *which* columns to log-transform once, by
# looking at their skewness in `train`. That choice uses the features only, never the price,
# and it learns no numbers that go into the model, so its effect on CV is negligible.

# %% [markdown]
# ### Step 2: predict log(SalePrice) instead of the price
# Same Ridge, same features. The only change is `target="log"`: the model now learns
# log(price), the scale Kaggle scores on and where the price is roughly symmetric (Part 1.2).

# %%
ridge = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=10))])
score, _, _ = cv_evaluate(ridge, X, y, X_test, target="log")
record("2", "Predict log(SalePrice)", score)

# %% [markdown]
# ### Step 3: give missing values their real meaning
# For many columns NA means "this house has none" (Part 1.3). We write that meaning down
# explicitly, but **only when another column confirms it**. For example, a missing
# `GarageType` becomes `"NoGarage"` only if `GarageArea` is 0. If the house *has* a garage
# but its type is missing, it stays missing, and the Pipeline fills it as an unknown.
#
# We also fix one column type. `MSSubClass` stores codes (20 = 1-storey 1946+, 60 = 2-storey
# 1946+, ...). They are labels, not quantities: code 60 isn't "three times" code 20. As a
# number, Ridge would treat it as one; as text, it gets one-hot encoded like any category.

# %%
ABSENT_RULES = [  # (column that proves absence, columns to relabel, label)
    ("TotalBsmtSF", ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], "NoBasement"),
    ("GarageArea", ["GarageType", "GarageFinish", "GarageQual", "GarageCond"], "NoGarage"),
    ("Fireplaces", ["FireplaceQu"], "NoFireplace"),
    ("PoolArea", ["PoolQC"], "NoPool"),
]


def fill_missing_meaning(df):
    df = df.copy()  # never change the caller's DataFrame
    for size_column, columns, label in ABSENT_RULES:
        absent = df[size_column] == 0
        for column in columns:
            df.loc[absent & df[column].isna(), column] = label
    # No size column exists for these; the data description says NA means "none".
    for column, label in [("Alley", "NoAlley"), ("Fence", "NoFence"), ("MiscFeature", "NoMisc")]:
        df[column] = df[column].fillna(label)
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


X3, X3_test = fill_missing_meaning(X), fill_missing_meaning(X_test)
score, _, _ = cv_evaluate(ridge, X3, y, X3_test, target="log")
record("3", "Missing values labelled by meaning; MSSubClass as category", score)

# %% [markdown]
# ### Step 4: new features that combine existing columns
# A buyer thinks "how big is the house in total?", but the data splits size across several
# columns. So we build the combined ideas directly.
#
# **Expect little change for Ridge here.** A sum like `TotalSF` is a *linear* combination of
# columns Ridge already has, so Ridge could already weight them that way. The totals start
# to matter in Step 5: log(total) is *not* a sum of the parts' logs. They also matter for the
# tree models in Part 4, which split on one column at a time.
#
# | Feature | Formula | Idea |
# |---|---|---|
# | `TotalSF` | above-ground living area + basement area | overall size |
# | `TotalBathrooms` | full baths + 0.5 × half baths (house + basement) | one comfort number |
# | `HouseAge` | year sold − year built | age at the time of sale |
# | `RemodelAge` | year sold − year remodelled | how fresh the interior is |
# | `TotalPorchSF` | sum of the five deck/porch areas | outdoor space |

# %%
def add_new_features(df):
    df = df.copy()
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodelAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)  # one row was remodelled after sale
    porches = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["TotalPorchSF"] = df[porches].sum(axis=1)
    return df


X4, X4_test = add_new_features(X3), add_new_features(X3_test)
score, _, test_log = cv_evaluate(ridge, X4, y, X4_test, target="log")
record("4", "New features: TotalSF, TotalBathrooms, ages, TotalPorchSF", score)
save_submission("s2_new_features", test_log)

# %% [markdown]
# ### Step 5: log-transform skewed area columns
# Area columns are right-skewed, just like the price was (skew table below). Price grows
# roughly in *proportion* to size: an extra 100 sq ft matters more for a 600 sq ft house than
# for a 4,000 sq ft one. A straight line can't express that on raw areas, but on log(area)
# it can. `log1p(x) = log(1 + x)` is used because many areas are 0 (no porch), and log(0) is
# undefined.
#
# **Which columns?** Every area column with skewness above 0.75 in `train`. The skew table
# is printed below. We fixed the list once and wrote it into the code. It uses only the
# features, never the price, so it carries no target information.

# %%
AREA_COLUMNS = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
                "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
                "PoolArea", "MiscVal", "TotalSF", "TotalPorchSF"]
skewness = X4[AREA_COLUMNS].skew().sort_values(ascending=False)
print(skewness.round(2).to_string())

SKEWED_COLUMNS = ["MiscVal", "PoolArea", "LotArea", "3SsnPorch", "LowQualFinSF", "BsmtFinSF2",
                  "ScreenPorch", "EnclosedPorch", "MasVnrArea", "OpenPorchSF", "LotFrontage",
                  "TotalSF", "BsmtFinSF1", "WoodDeckSF", "TotalBsmtSF", "1stFlrSF", "GrLivArea",
                  "TotalPorchSF", "BsmtUnfSF", "2ndFlrSF"]  # all skew > 0.75 (GarageArea is 0.18)
assert set(SKEWED_COLUMNS) == set(skewness[skewness > 0.75].index)


def log_skewed(df):
    df = df.copy()
    df[SKEWED_COLUMNS] = np.log1p(df[SKEWED_COLUMNS])
    return df

# %%
X5, X5_test = log_skewed(X4), log_skewed(X4_test)
score, _, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log")
record("5", "log1p of skewed area columns", score)
save_submission("s3_log_features", test_log)

# %% [markdown]
# ### Step 6: drop the two huge-but-cheap houses from training
# They are removed **from each fold's training rows only** (Part 1.4). They are still predicted
# and still count in the CV score. Removing them from scoring as well would flatter us,
# because the hidden test set may contain houses like them.

# %%
score, _, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log", drop_outliers=True)
record("6", "Drop 2 outliers from training folds", score)

# %% [markdown]
# ### Tuning Ridge's alpha
# `alpha=10` was a guess. We try a few values with the same CV and keep the best. Each value
# is a separate experiment on the same folds, so this is a fair comparison. If the best value
# lands on the edge of the grid, the true best might lie beyond it, so the grid should widen.

# %%
alpha_scores = {}
for alpha in [1, 3, 10, 30, 100]:
    candidate = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=alpha))])
    alpha_scores[alpha], _, _ = cv_evaluate(candidate, X5, y, X5_test, target="log", drop_outliers=True)
    print(f"alpha={alpha:>3}: CV RMSE {alpha_scores[alpha]:.5f}")
best_alpha = min(alpha_scores, key=alpha_scores.get)
print("best alpha:", best_alpha)

ridge = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=best_alpha))])
score, oof_ridge, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log", drop_outliers=True)
record("6b", f"Ridge alpha tuned to {best_alpha}", score)
save_submission("s4_no_outliers", test_log)

# These are the features that Parts 4 and 5 build on.
X_final, X_test_final = X5, X5_test

# %% [markdown]
# ### Which feature engineering helped most?

# %%
part3 = results[results.step.isin(["1", "2", "3", "4", "5", "6", "6b"])].copy()
part3["improvement"] = -part3.cv_rmse.diff()
part3
