# %% [markdown]
# ## Part 4: Comparing models
#
# The features are now fixed (`X_final` from Part 3, plus a tree-friendly version for
# CatBoost). Every model below uses the **same folds, the log target, and the outlier rule**.
# So the only thing that changes is the model. Each model's out-of-fold (OOF) and test
# predictions are kept in two dictionaries, `oof` and `test_preds`, for blending in Part 5.
#
# | Family | Models | Idea |
# |---|---|---|
# | Linear | Ridge, Lasso, ElasticNet | price = weighted sum of features, with a penalty |
# | Tree ensembles | Random Forest, XGBoost, CatBoost | many decision trees, combined |
# | Kernel | SVR | a smooth curve fitted in a transformed space |

# %%
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

oof, test_preds, seconds = {}, {}, {}


def run_model(step, name, description, model, X_train, X_predict):
    """Score one model with the shared CV, keep its predictions, and record it."""
    started = time.time()
    score, oof[name], test_preds[name] = cv_evaluate(model, X_train, y, X_predict,
                                                      target="log", drop_outliers=True)
    seconds[name] = time.time() - started
    record(step, description, score)
    print(f"   ({seconds[name]:.0f} s)")


def linear_pipeline(model):
    return Pipeline([("prep", make_linear_preprocessor()), ("model", model)])


run_model("6b", "ridge", "Ridge (reference, re-run)", linear_pipeline(Ridge(alpha=best_alpha)),
          X_final, X_test_final)
results = results.iloc[:-1]  # the re-run duplicates step 6b; keep the table tidy

# %% [markdown]
# ### Step 7: Lasso and ElasticNet, two other penalties
# - **Ridge** penalises `alpha × Σ coefficient²`. It shrinks every coefficient but never to
#   exactly zero.
# - **Lasso** penalises `alpha × Σ |coefficient|`. That pushes weak features to **exactly
#   zero**, so Lasso also does feature selection.
# - **ElasticNet** mixes the two (`l1_ratio` = share of the Lasso penalty).
#
# Their alphas live on a much smaller scale than Ridge's, because the penalty is written
# differently in sklearn. We try a small grid for each and keep the best.

# %%
for alpha in [0.0002, 0.0005, 0.001]:
    score, _, _ = cv_evaluate(linear_pipeline(Lasso(alpha=alpha, max_iter=50000)),
                              X_final, y, X_test_final, drop_outliers=True)
    print(f"Lasso alpha={alpha}: {score:.5f}")

# %%
run_model("7", "lasso", "Lasso (alpha=0.0005)",
          linear_pipeline(Lasso(alpha=0.0005, max_iter=50000)), X_final, X_test_final)
run_model("7b", "elasticnet", "ElasticNet (alpha=0.001, l1_ratio=0.5)",
          linear_pipeline(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000)),
          X_final, X_test_final)

# %%
lasso_fit = linear_pipeline(Lasso(alpha=0.0005, max_iter=50000)).fit(X_final[~outlier], np.log(y[~outlier]))
coefficients = lasso_fit.named_steps["model"].coef_
print(f"Lasso kept {np.sum(coefficients != 0)} of {len(coefficients)} columns (the rest are exactly 0)")

# %% [markdown]
# ### Step 8: Random Forest (bagging)
# A **decision tree** asks yes/no questions ("OverallQual ≥ 8?", "TotalSF > 2,500?") and
# predicts the average price of the houses that end up in each leaf. A single deep tree
# memorises the training data (overfits). A **Random Forest** grows hundreds of trees, each on
# a random bootstrap sample of the houses and a random subset of the columns, and averages
# them. Averaging many different noisy trees cancels much of the noise. This is **bagging**.
#
# Trees only split on thresholds, so scaling and log transforms don't matter to them. They
# also find interactions (e.g. "large **and** high quality") without being told.

# %%
run_model("8", "rf", "Random Forest (300 trees)",
          linear_pipeline(RandomForestRegressor(n_estimators=300, max_features=0.3,
                                                min_samples_leaf=2, n_jobs=-1, random_state=42)),
          X_final, X_test_final)

# %% [markdown]
# ### Step 9: XGBoost (boosting)
# **Boosting** builds trees **one after another**. Each new small tree is fitted to the errors
# the previous trees still make, and only a small fraction of it is added (the learning
# rate). Bagging averages independent trees to cut noise; boosting adds dependent trees to
# cut bias. Settings: 1,500 shallow trees (depth 3), learning rate 0.03, and each tree sees
# 80% of houses and 50% of columns (more randomness, less overfitting).

# %%
run_model("9", "xgb", "XGBoost (1500 trees, depth 3, lr 0.03)",
          linear_pipeline(XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3,
                                       subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                                       n_jobs=-1, random_state=42)),
          X_final, X_test_final)

# %% [markdown]
# ### Step 10: CatBoost (boosting with native categories)
# CatBoost is also gradient boosting, but it handles **text columns directly**. It replaces
# each category by a smoothed average of the target, computed in a careful order to avoid
# leakage, instead of exploding it into hundreds of one-hot columns. It also builds
# *symmetric* trees, which tend to overfit less on small data.
#
# Input: the Step 4 features (meaningful NA labels + new features). Logs aren't needed for
# trees. Remaining text gaps are filled with `"Missing"`, because CatBoost requires text in
# category columns.

# %%
categorical_columns = X4.select_dtypes(exclude="number").columns.tolist()
X_cat = X4.copy()
X_cat_test = X4_test.copy()
X_cat[categorical_columns] = X_cat[categorical_columns].fillna("Missing")
X_cat_test[categorical_columns] = X_cat_test[categorical_columns].fillna("Missing")

catboost = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                             # a tuple, not a list: sklearn's clone() (used in cv_evaluate)
                             # cannot copy CatBoost when cat_features is a list
                             cat_features=tuple(categorical_columns), random_seed=42,
                             verbose=0, allow_writing_files=False)
run_model("10", "catboost", "CatBoost (1500 trees, depth 6, native categories)",
          catboost, X_cat, X_cat_test)
save_submission("s5_catboost", test_preds["catboost"])

# %% [markdown]
# ### Step 11: Support Vector Regression (SVR)
# SVR with an RBF kernel predicts a house's price from its **similarity** to the training
# houses. Similarity is based on distance in feature space, which is why scaling (done in our
# Pipeline) is essential. Errors smaller than `epsilon` are ignored. `C` sets how hard the
# model tries to fit the rest, and `gamma` sets how local "similar" is.
#
# The grid was widened twice, because the best pair kept landing on its edge. The final best,
# C=1 and gamma=0.003, sits on a corner of this grid. A separate check beyond that corner
# (C=0.3; gamma=0.005 and 0.01) scored worse (0.124–0.130), so it is a true minimum.

# %%
svr_scores = {}
for C in [1, 3, 10, 30]:
    for gamma in [0.0001, 0.0003, 0.001, 0.003]:
        svr_scores[(C, gamma)], _, _ = cv_evaluate(
            linear_pipeline(SVR(C=C, epsilon=0.01, gamma=gamma)),
            X_final, y, X_test_final, drop_outliers=True)
        print(f"SVR C={C}, gamma={gamma}: {svr_scores[(C, gamma)]:.5f}")
best_C, best_gamma = min(svr_scores, key=svr_scores.get)

run_model("11", "svr", f"SVR (C={best_C}, gamma={best_gamma})",
          linear_pipeline(SVR(C=best_C, epsilon=0.01, gamma=best_gamma)), X_final, X_test_final)

# %% [markdown]
# ### Model comparison

# %%
names = ["ridge", "lasso", "elasticnet", "rf", "xgb", "catboost", "svr"]
y_log = np.log(y.values)
comparison = pd.DataFrame({
    "cv_rmse": [np.sqrt(np.mean((y_log - oof[name]) ** 2)) for name in names],
    "seconds": [seconds[name] for name in names],
}, index=names).sort_values("cv_rmse")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(comparison.index[::-1], comparison.cv_rmse[::-1], color="steelblue")
ax.set(title="Model comparison (5-fold CV, lower is better)", xlabel="CV RMSE (log price)",
       xlim=(comparison.cv_rmse.min() * 0.95, comparison.cv_rmse.max() * 1.02))
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "05_model_comparison.png", dpi=120)
plt.show()
comparison.round(5)
