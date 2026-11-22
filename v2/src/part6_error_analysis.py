# %% [markdown]
# ## Part 6: Error analysis — what the leaderboard taught us
#
# After submitting S1–S6, we compared the Kaggle scores with our CV scores. Two things did not
# match: CatBoost (S5) beat Ridge (S4) on CV but not on Kaggle, and the final blend (S6) scored
# worse than our team's first attempt (an earlier ElasticNet + CatBoost blend). This Part
# investigates why, and makes one more principled change (Step 13).
#
# *The Kaggle scores below were copied from the competition's submissions page. They can't be
# computed in the notebook, because the test prices are hidden.*

# %%
kaggle = pd.DataFrame({
    "submission": ["s1_ridge_raw", "s2_new_features", "s3_log_features", "s4_no_outliers",
                   "s5_catboost", "s6_blend", "first attempt (ElasticNet + CatBoost)"],
    "cv_rmse": [cv_by_step[s] for s in ["1", "4", "5", "6b", "10", "12"]] + [0.12117],
    "kaggle_score": [0.14188, 0.13044, 0.12253, 0.12236, 0.12310, 0.12037, 0.11745],
})
kaggle["kaggle_change"] = kaggle.kaggle_score.diff()
kaggle.loc[6, "kaggle_change"] = np.nan  # the first attempt is not part of the S1-S6 sequence
kaggle.round(5)

# %% [markdown]
# ### Where does our CV error come from?
# Two training houses, the huge-but-cheap Partial sales from Part 1 (Ids 524 and 1299), have
# enormous errors. In our 5-fold split **both land in the same fold**. So whenever one of them
# is being predicted, the other is missing from training too. CV can therefore never test
# whether a model could learn this pattern from one of them and use it on the other.

# %%
blend_error = y_log - blend_oof
fold_of = np.zeros(len(train), dtype=int)
for fold, (_, valid_idx) in enumerate(kf.split(X), start=1):
    fold_of[valid_idx] = fold

is_outlier = outlier.to_numpy()
print(pd.DataFrame({"Id": train.Id[is_outlier].to_numpy(), "fold": fold_of[is_outlier],
                    "true price": train.SalePrice[is_outlier].to_numpy(),
                    "Step 12 log error": blend_error[is_outlier].round(2)}).to_string(index=False))
print(f"\nStep 12 blend CV RMSE, all 1460 houses:        {np.sqrt(np.mean(blend_error ** 2)):.4f}")
print(f"Step 12 blend CV RMSE, without these 2 houses:  {np.sqrt(np.mean(blend_error[~is_outlier] ** 2)):.4f}")

# %% [markdown]
# ### The test set has a third house like them
# Looking at the test **features** (not prices, which are hidden) shows one more huge Partial
# sale: Id 2550. Steps 6–12 removed its two look-alikes from training, so every model learned
# only that "huge house = expensive", and predicts a high price for it.

# %%
huge_test = (test.GrLivArea > 4000).to_numpy()
print(test.loc[huge_test, ["Id", "GrLivArea", "TotalBsmtSF", "OverallQual", "SaleCondition"]].to_string(index=False))
print()
for label, predictions in [("S4 Ridge", test_preds["ridge"]), ("S5 CatBoost", test_preds["catboost"]),
                           ("S6 blend", blend_test)]:
    print(f"{label:12s} predicts ${np.exp(predictions[huge_test][0]):,.0f} for Id 2550")

# %% [markdown]
# ### Step 13 idea: keep the look-alikes and tell the models what they are
# Instead of deleting the two training houses, we keep them and add a feature,
# `HugePartial` = 1 for houses over 4,000 sq ft sold as Partial sales. That lets a model learn
# a separate adjustment for this rare kind of sale. Like the Step 6 outlier rule, the flag must
# use the **raw** living area (`X_final` already holds log1p of it; see the Part 3 bug).
#
# We also clean one impossible value: a test house has `GarageYrBlt` = 2207, clearly a typo.
# Garage years later than the year after the sale become missing, and the Pipeline (or
# CatBoost) then treats them like any other gap.

# %%
def clean_years(df):
    """Impossible garage years (e.g. 2207, a typo) become missing."""
    df = df.copy()
    df.loc[df["GarageYrBlt"] > df["YrSold"] + 1, "GarageYrBlt"] = np.nan
    return df


def add_huge_partial_flag(df, raw):
    """1 for houses over 4,000 sq ft sold as Partial sales, else 0 (uses the raw, unlogged area)."""
    df = df.copy()
    df["HugePartial"] = ((raw["GrLivArea"] > 4000) & (raw["SaleCondition"] == "Partial")).astype(int).to_numpy()
    return df


X13, X13_test = add_huge_partial_flag(clean_years(X_final), train), add_huge_partial_flag(clean_years(X_test_final), test)
X13_cat, X13_cat_test = add_huge_partial_flag(clean_years(X_cat), train), add_huge_partial_flag(clean_years(X_cat_test), test)
print("flagged in train:", train.Id[X13.HugePartial == 1].tolist(), "  flagged in test:", test.Id[X13_test.HugePartial == 1].tolist())
print("garage years cleaned in test:", test.Id[X_test_final.GarageYrBlt > X_test_final.YrSold + 1].tolist())

# %% [markdown]
# ### Checking the idea where normal CV can't
# Because both look-alikes share a fold, we test the idea directly: hold out **one** of them,
# train on every other house, and predict it. We do this twice (once per house), with the same
# four models and settings as the Step 12 blend.
# - **Current method (Steps 6–12):** both look-alikes removed from training, no flag.
# - **Step 13 method:** the other look-alike kept, with the `HugePartial` flag.

# %%
def make_xgb(seed=42):
    """The XGBoost model from Step 9 (same settings), with a choice of random seed."""
    return linear_pipeline(XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3,
                                        subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                                        n_jobs=-1, random_state=seed))


step13_models = {  # the Step 12 blend members, unchanged
    "lasso": (linear_pipeline(Lasso(alpha=lasso_alpha, max_iter=50000)), "linear"),
    "xgb": (make_xgb(), "linear"),
    "catboost": (catboost, "tree"),
    "svr": (linear_pipeline(SVR(C=best_C, epsilon=0.01, gamma=best_gamma)), "linear"),
}

rows = []
for held_out in [524, 1299]:
    is_target = (train.Id == held_out).to_numpy()
    for setting, X_linear, X_tree, keep in [
            ("current: drop both, no flag", X_final, X_cat, ~is_outlier),
            ("Step 13: keep the other + flag", X13, X13_cat, ~is_target)]:
        predictions = []
        for name, (model, kind) in step13_models.items():
            X_use = X_linear if kind == "linear" else X_tree
            fitted = clone(model).fit(X_use[keep], y_log[keep])
            predictions.append(fitted.predict(X_use[is_target])[0])
        blend = np.mean(predictions)
        rows.append({"held-out Id": held_out, "setting": setting,
                     "true price": int(train.SalePrice[is_target].iloc[0]),
                     "blend predicts": int(round(np.exp(blend))),
                     "log error": round(y_log[is_target][0] - blend, 2)})
pd.DataFrame(rows)

# %% [markdown]
# ### Step 13: refit the same blend on the new features
# Same four models, same settings, same equal weights. The only changes are the cleaned garage
# year, keeping the two look-alikes in training (`drop_outliers=False`), and the flag. CV can't
# reward the flag (both look-alikes share a fold), so expect the CV score to stay about the
# same. The check above is the evidence for this step.

# %%
oof13, test13 = {}, {}
for name, (model, kind) in step13_models.items():
    X_train_use, X_test_use = (X13, X13_test) if kind == "linear" else (X13_cat, X13_cat_test)
    score, oof13[name], test13[name] = cv_evaluate(model, X_train_use, y, X_test_use,
                                                   target="log", drop_outliers=False)
    print(f"{name:9s} CV RMSE {score:.5f}")

blend13_oof = np.mean(list(oof13.values()), axis=0)
blend13_test = np.mean(list(test13.values()), axis=0)
record("13", "Keep outliers + HugePartial flag, clean garage years; same 4-model blend",
       np.sqrt(np.mean((y_log - blend13_oof) ** 2)))
save_submission("s7_outlier_flag", blend13_test)

for label, predictions in [("S6 blend (Step 12)", blend_test), ("S7 blend (Step 13)", blend13_test)]:
    print(f"{label}: Id 2550 predicted at ${np.exp(predictions[huge_test][0]):,.0f}")
change = np.abs(blend13_test - blend_test)[~huge_test]
print(f"other test houses: median change {np.median(change):.4f}, largest change {change.max():.4f} (log scale)")

# %% [markdown]
# ### Other ideas we tried and did not keep
# Every idea is scored with the same CV on the Step 13 features. Stacking is scored with its own
# CV over the out-of-fold predictions, so it can't grade its own homework.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


def rmse(log_predictions):
    return np.sqrt(np.mean((y_log - log_predictions) ** 2))


QUALITY = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}


def add_quality_scores(df, raw):
    """The first attempt's extra features: quality grades as numbers, and quality x area."""
    df = df.copy()
    for column in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                   "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]:
        df[column + "Score"] = raw[column].map(QUALITY).fillna(0).to_numpy()
    df["QualityLivingArea"] = np.log1p(raw["OverallQual"] * raw["GrLivArea"]).to_numpy()
    return df


tried = {"Step 13 blend (kept)": rmse(blend13_oof)}

oof_matrix = np.column_stack(list(oof13.values()))
stacked = cross_val_predict(LinearRegression(positive=True), oof_matrix, y_log,
                            cv=KFold(n_splits=5, shuffle=True, random_state=7))
tried["stacking: weights learned from OOF predictions"] = rmse(stacked)

tried["first attempt's recipe: Lasso + CatBoost 50/50"] = rmse((oof13["lasso"] + oof13["catboost"]) / 2)

score, _, _ = cv_evaluate(linear_pipeline(Lasso(alpha=lasso_alpha, max_iter=50000)),
                          add_quality_scores(X13, X4), y, add_quality_scores(X13_test, X4_test),
                          drop_outliers=False)
tried["Lasso alone, Step 13 features"] = rmse(oof13["lasso"])
tried["Lasso alone, + first attempt's quality scores"] = score

xgb_runs = [oof13["xgb"]]
for seed in (7, 2026):
    _, oof_seed, _ = cv_evaluate(make_xgb(seed), X13, y, X13_test, drop_outliers=False)
    xgb_runs.append(oof_seed)
tried["XGBoost alone, seed 42"] = rmse(oof13["xgb"])
tried["XGBoost alone, average of 3 seeds"] = rmse(np.mean(xgb_runs, axis=0))
tried["Step 13 blend, but with the 3-seed XGBoost"] = rmse(np.mean(
    [oof13["lasso"], np.mean(xgb_runs, axis=0), oof13["catboost"], oof13["svr"]], axis=0))

pd.Series(tried).round(5).to_frame("cv_rmse")

# %% [markdown]
# ### Updated submission list

# %%
summary = pd.concat([summary, pd.DataFrame([{"submission": "s7_outlier_flag", "step": "13",
                                             "cv_rmse": results.set_index("step").cv_rmse["13"]}])],
                    ignore_index=True)
summary.to_csv(OUT_DIR / "submission_summary.csv", index=False)
summary.round(5)

# %% [markdown]
# ## Part 6 summary
# - **CV and the leaderboard disagreed** by amounts (around 0.001–0.003) that are small next to
#   the fold-to-fold spread of our CV, so single comparisons like S4 vs S5 are mostly noise.
# - **A blind spot in our CV:** the two huge Partial sales share a fold, so CV could never
#   reward learning from them. Yet they dominate the error. The blend's CV RMSE drops a lot
#   without them.
# - **Step 13** keeps them in training with a `HugePartial` flag, and cleans an impossible
#   garage year. CV stays about the same, as expected. In the hold-one-out check, the error on
#   these houses shrinks sharply, and the prediction for the one such test house (Id 2550)
#   falls to roughly the price range of its two training look-alikes.
# - **Not kept:** stacking, seed averaging, and the first attempt's recipe and features did not
#   beat the Step 13 blend on CV.
# - **Honest caveat:** Step 13 was prompted by the leaderboard. We made exactly one extra
#   submission (S7) to check it, rather than trying many variants against the leaderboard.
