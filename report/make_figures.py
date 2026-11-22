"""Column-width figures for the report. Run from the repo root:
    .venv/bin/python report/make_figures.py
Numbers come from the data, v2/results/*.csv, and the reference notebook run (noted inline)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 3.3  # inches, one column

train = pd.read_csv(ROOT / "house-prices-advanced-regression-techniques" / "train.csv",
                    keep_default_na=False, na_values=["NA", ""])
test = pd.read_csv(ROOT / "house-prices-advanced-regression-techniques" / "test.csv",
                   keep_default_na=False, na_values=["NA", ""])
cv = pd.read_csv(ROOT / "v2" / "results" / "cv_results.csv", dtype={"step": str}).set_index("step").cv_rmse
kaggle = pd.read_csv(ROOT / "v2" / "results" / "kaggle_log.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1 target distribution
fig, axes = plt.subplots(2, 1, figsize=(W, 3.0))
axes[0].hist(train.SalePrice / 1000, bins=50, color="steelblue")
axes[0].set(title=f"SalePrice (skewness {train.SalePrice.skew():.2f})", xlabel="price ($1,000)")
axes[1].hist(np.log(train.SalePrice), bins=50, color="seagreen")
axes[1].set(title=f"log(SalePrice) (skewness {np.log(train.SalePrice).skew():.2f})", xlabel="log price")
save(fig, "f01_target.png")

# 2 missing values
absent = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",
          "GarageQual", "GarageCond", "GarageYrBlt", "BsmtQual", "BsmtCond", "BsmtExposure",
          "BsmtFinType1", "BsmtFinType2"]
missing = train.drop(columns="SalePrice").isna().sum()
missing = missing[missing > 0].sort_values().tail(12)
fig, ax = plt.subplots(figsize=(W, 2.6))
ax.barh(missing.index, missing.values, color=["orange" if c in absent else "gray" for c in missing.index])
ax.set(xlabel="houses with NA (training set)", title="orange: NA = has none, gray: unknown")
save(fig, "f02_missing.png")

# 3 outliers
out = (train.GrLivArea > 4000) & (train.SalePrice < 300000)
fig, ax = plt.subplots(figsize=(W, 2.4))
ax.scatter(train.GrLivArea, train.SalePrice / 1000, s=4, alpha=0.5)
ax.scatter(train.GrLivArea[out], train.SalePrice[out] / 1000, s=80, facecolors="none", edgecolors="red")
for _, r in train[out].iterrows():
    ax.annotate(f"Id {r.Id}", (r.GrLivArea, r.SalePrice / 1000), xytext=(-38, 12), textcoords="offset points", fontsize=7)
ax.set(xlabel="GrLivArea (sq ft)", ylabel="SalePrice ($1,000)")
save(fig, "f03_outliers.png")

# 4 correlations
corr = train.select_dtypes("number").drop(columns="Id").corr()["SalePrice"].drop("SalePrice").sort_values().tail(8)
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.barh(corr.index, corr.values, color="steelblue")
for i, v in enumerate(corr.values):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
ax.set(xlabel="correlation with SalePrice", xlim=(0, 0.9))
save(fig, "f04_correlation.png")

# 5 pathway flow
boxes = ["Explore the data (Section 2)",
         "Fixed 5-fold CV + Pipeline\nSteps 0-1: baselines",
         "Feature engineering, one change per step\nSteps 2-6",
         "Compare 7 models\nSteps 7-11",
         "Equal-weight blend of 4 models\nStep 12",
         "Submit S1-S6, compare CV with Kaggle",
         "Error analysis -> Step 13\nS7: 0.11624"]
fig, ax = plt.subplots(figsize=(W, 3.6))
ax.axis("off")
ys = np.linspace(0.93, 0.07, len(boxes))
for i, (y, text) in enumerate(zip(ys, boxes)):
    ax.text(0.5, y, text, ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f0fa" if i < 5 else "#fdebd3", ec="#555555"))
    if i < len(boxes) - 1:
        ax.annotate("", xy=(0.5, ys[i + 1] + 0.055), xytext=(0.5, y - 0.055),
                    arrowprops=dict(arrowstyle="->", color="#555555"))
save(fig, "f05_pathway.png")

# 6 fold assignment of the first 20 houses
folds = np.zeros(len(train), dtype=int)
for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), start=1):
    folds[va] = k
grid = np.array([[1 if folds[h] == k else 0 for h in range(20)] for k in range(1, 6)])
fig, ax = plt.subplots(figsize=(W, 1.7))
ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#cfe0f3", "#e8743b"]), aspect="auto")
ax.set_xticks(range(20), [str(i) for i in train.Id[:20]], fontsize=6)
ax.set_yticks(range(5), [f"fold {k}" for k in range(1, 6)])
ax.set(xlabel="house Id (first 20 of 1,460)", title="orange: validation, blue: training")
ax.spines[:].set_visible(False)
save(fig, "f06_folds.png")

# 7 feature engineering staircase
steps = ["1", "2", "3", "4", "5", "6"]
labels = ["1 raw", "2 log y", "3 NA", "4 new", "5 log x", "6 outl."]
fig, ax = plt.subplots(figsize=(W, 2.3))
bars = ax.bar(labels, [cv[s] for s in steps], color=["gray"] + ["orange"] * 5)
for b, s in zip(bars, steps):
    ax.text(b.get_x() + b.get_width() / 2, cv[s] + 0.0006, f"{cv[s]:.4f}", ha="center", fontsize=6.5)
ax.set(ylim=(0.12, 0.16), ylabel="CV RMSE")
save(fig, "f07_staircase.png")

# 8 Ridge alpha (reference run, Part 3)
alphas = [1, 3, 10, 30, 100]
scores = [0.13027, 0.12843, 0.12766, 0.12797, 0.12980]
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.plot(alphas, scores, "o-", color="steelblue")
ax.set_xscale("log")
ax.set_xticks(alphas, [str(a) for a in alphas])
ax.annotate("best: 10", (10, 0.12766), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=7)
ax.set(xlabel="Ridge alpha", ylabel="CV RMSE")
save(fig, "f08_alpha.png")

# 9 model comparison
models = {"XGBoost": "9", "CatBoost": "10", "SVR": "11", "Lasso": "7", "ElasticNet": "7b", "Ridge": "6b", "Random Forest": "8"}
vals = pd.Series({m: cv[s] for m, s in models.items()}).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.barh(vals.index, vals.values, color=["#8c8c8c" if m == "Random Forest" else "steelblue" for m in vals.index])
for i, v in enumerate(vals.values):
    ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=7)
ax.set(xlim=(0.115, 0.142), xlabel="CV RMSE (x-axis starts at 0.115)")
save(fig, "f09_models.png")

# 11 CV vs Kaggle
k = kaggle.set_index("submission")
names = ["s1_ridge_raw", "s2_new_features", "s3_log_features", "s4_no_outliers", "s5_catboost", "s6_blend", "s7_outlier_flag"]
x = np.arange(1, 8)
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.plot(x, [k.loc[n, "cv_rmse"] for n in names], "o-", label="our CV")
ax.plot(x, [k.loc[n, "kaggle_score"] for n in names], "s-", label="Kaggle")
ax.axhline(0.11745, ls="--", color="gray", lw=1)
ax.text(1.05, 0.1182, "first attempt on Kaggle (0.11745)", fontsize=6.5, color="gray")
ax.set_xticks(x, [f"S{i}" for i in x])
ax.set(ylabel="RMSE (log price)")
ax.legend(frameon=False)
save(fig, "f11_cv_vs_kaggle.png")

# 12 predictions for test Id 2550 (submission files)
subs = ROOT / "v2" / "results" / "submissions"
preds = {f"S{i}": pd.read_csv(subs / f"{n}.csv").set_index("Id").SalePrice[2550] for i, n in enumerate(names, start=1)}
preds["First\nattempt"] = pd.read_csv(ROOT / "v1_reference" / "submissions" / "recommended.csv").set_index("Id").SalePrice[2550]
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.bar(list(preds), [v / 1000 for v in preds.values()],
       color=["steelblue"] * 6 + ["seagreen", "gray"])
for price in (160, 184.75):
    ax.axhline(price, ls="--", color="red", lw=0.8)
ax.set(ylabel="predicted price ($1,000)", ylim=(0, 1000),
       title="Test house Id 2550\nred lines: 524 and 1299 sold for \\$160k, \\$185k")
ax.tick_params(axis="x", labelsize=6.5)
save(fig, "f12_id2550.png")

# 13 leave-one-outlier-out check (Part 6)
houses = ["Id 524", "Id 1299"]
true = [184.75, 160.0]
earlier = [665.634, 708.551]
step13 = [242.533, 303.257]
xi = np.arange(2)
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.bar(xi - 0.25, true, 0.25, label="true price", color="gray")
ax.bar(xi, earlier, 0.25, label="earlier method", color="steelblue")
ax.bar(xi + 0.25, step13, 0.25, label="Step 13", color="seagreen")
ax.set_xticks(xi, houses)
ax.set(ylabel="price ($1,000)")
ax.set_ylim(0, 950)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
save(fig, "f13_loo.png")

# 16 bagging vs boosting on a small hold-out experiment (same as the Part 4 notes)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
cols = ["OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt", "GarageCars"]
X_tr, X_va, y_tr, y_va = train_test_split(train[cols].fillna(0), np.log(train.SalePrice), test_size=0.2, random_state=0)
rmse = lambda p: np.sqrt(np.mean((y_va - p) ** 2))
forest_sizes = [1, 3, 10, 30, 100, 300]
forest = [rmse(RandomForestRegressor(n_estimators=n, random_state=0).fit(X_tr, y_tr).predict(X_va)) for n in forest_sizes]
booster = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, random_state=0).fit(X_tr, y_tr)
staged = [rmse(p) for p in booster.staged_predict(X_va)]
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.plot(forest_sizes, forest, "o-", label="Random Forest (more trees, averaged)")
ax.plot(range(1, 501), staged, "-", label="Boosting (trees added one by one)")
ax.set_xscale("log")
ax.set(xlabel="number of trees", ylabel="validation RMSE", ylim=(0.13, 0.4))
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
save(fig, "f16_bagging_boosting.png")
print("forest", [round(v, 4) for v in forest], "boost min", round(min(staged), 4), "at", int(np.argmin(staged)) + 1)

# 14 console output of the feature steps (reference run, Parts 2-3)
lines = ["Step 0: CV RMSE 0.40776",
         "Step 1: CV RMSE 0.15534 (-0.25243 vs previous step)",
         "Step 2: CV RMSE 0.15206 (-0.00327 vs previous step)",
         "Step 3: CV RMSE 0.15106 (-0.00100 vs previous step)",
         "Step 4: CV RMSE 0.15121 (+0.00014 vs previous step)",
         "Step 5: CV RMSE 0.13239 (-0.01882 vs previous step)",
         "Step 6: CV RMSE 0.12766 (-0.00473 vs previous step)"]
fig, ax = plt.subplots(figsize=(W, 1.35))
ax.axis("off")
fig.patch.set_facecolor("#f4f4f4")
ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=6.3, va="top", color="#333333")
fig.savefig(OUT / "f14_console.png", dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("figures written to", OUT)
