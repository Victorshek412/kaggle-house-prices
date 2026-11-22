# %% [markdown]
# # House Prices: from baseline to blend, one measured step at a time
#
# **Task.** Predict the sale price of 1,459 houses in Ames, Iowa, from 79 descriptive columns
# (size, age, quality, location, ...). We learn from 1,460 houses whose prices we know.
#
# **Metric.** Kaggle scores the RMSE between the *log* of our predicted price and the *log* of the
# true price. Working in logs means a 10% miss counts the same on a cheap house as on an
# expensive one.
#
# **How this notebook is organised.** Part 1 explores the data. Part 2 builds a baseline and the
# cross-validation tools. Part 3 adds feature engineering one step at a time. Part 4 compares
# models. Part 5 blends them. Every step's score goes into one results table, so the whole
# pathway can be read off at the end.
#
# *Numbers quoted in the text come from the reference run. A run on Kaggle may differ slightly
# (around the 4th decimal) because library versions differ.*
#
# ## Part 1: Exploratory data analysis (EDA)

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On Kaggle the data sits under /kaggle/input; locally it sits next to this repo's v2 folder.
CANDIDATES = [Path("/kaggle/input/house-prices-advanced-regression-techniques"),
              *sorted(path.parent for path in Path("/kaggle/input").glob("**/train.csv")),
              Path("../house-prices-advanced-regression-techniques"),
              Path("house-prices-advanced-regression-techniques")]
DATA_DIR = next((path for path in CANDIDATES if (path / "train.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Could not find train.csv. Looked in: " + ", ".join(map(str, CANDIDATES))
                            + ". On Kaggle, attach the competition data with 'Add Input'.")

ON_KAGGLE = Path("/kaggle/working").exists()
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else DATA_DIR.parent / "v2" / "results"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def read_csv(name):
    # The string "None" is a real category (e.g. MasVnrType = no masonry veneer).
    # Only "NA" and empty cells mean missing.
    return pd.read_csv(DATA_DIR / name, keep_default_na=False, na_values=["NA", ""])


train = read_csv("train.csv")
test = read_csv("test.csv")
sample = read_csv("sample_submission.csv")
print("train:", train.shape, " test:", test.shape)

# %% [markdown]
# ### 1.1 First look
# `train` has 81 columns: `Id`, 79 descriptive columns, and the target `SalePrice`.
# `test` has the same columns except `SalePrice`, which is what we must predict.

# %%
print(train.dtypes.value_counts())
train.head()

# %% [markdown]
# ### 1.2 The target: SalePrice
# The middle half of houses sell for $130k–$214k, but a long tail runs up to $755k. A *right-skewed*
# target like this pulls a linear model towards the few expensive houses. Taking the log
# makes the distribution almost symmetric, and it is also the scale Kaggle scores on.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(train.SalePrice, bins=50, color="steelblue")
axes[0].set(title=f"SalePrice (skew = {train.SalePrice.skew():.2f})", xlabel="price ($)")
axes[1].hist(np.log(train.SalePrice), bins=50, color="seagreen")
axes[1].set(title=f"log(SalePrice) (skew = {np.log(train.SalePrice).skew():.2f})", xlabel="log price")
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "01_saleprice_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ### 1.3 Missing values: two very different kinds
# `data_description.txt` says that for many columns **NA means "this house doesn't have one"**:
# no pool, no garage, no basement, no fence. That is information, not a gap.
# For other columns (e.g. `LotFrontage`), NA really means *we don't know*.
# Part 3 treats the two kinds differently.
#
# A few columns are missing **only in test** (e.g. `MSZoning`, `GarageCars`). Our pipeline
# must cope with gaps it never saw during training.

# %%
ABSENT_MEANING = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

missing = pd.DataFrame({"train": train.drop(columns="SalePrice").isna().sum(),
                        "test": test.isna().sum()})
missing = missing[(missing.train + missing.test) > 0]
missing["kind"] = np.where(missing.index.isin(ABSENT_MEANING), "NA = absent", "truly unknown")
missing = missing.sort_values("train", ascending=False)
print("Columns missing only in test:", missing.index[missing.train.eq(0)].tolist())

top = missing.head(20).iloc[::-1]
colors = np.where(top.kind.eq("NA = absent"), "orange", "gray")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top.index, top.train, color=colors)
ax.set(title="Missing values in train (orange = NA means absent)", xlabel="rows missing")
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "02_missing_values.png", dpi=120)
plt.show()
missing

# %% [markdown]
# ### 1.4 Outliers
# Living area (`GrLivArea`) is one of the strongest predictors. But two houses are huge
# (over 4,000 sq ft) yet sold cheaply (under $300k). Both were *Partial* sales of new builds,
# so they don't follow the normal price pattern. A model that tries to fit them bends its
# line for every other house. Part 3 (Step 6) tests removing them from training.

# %%
outlier = (train.GrLivArea > 4000) & (train.SalePrice < 300000)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(train.GrLivArea, train.SalePrice, s=8, alpha=0.5)
ax.scatter(train.GrLivArea[outlier], train.SalePrice[outlier], s=150,
           facecolors="none", edgecolors="red", linewidths=2, label="outliers")
ax.set(title="Living area vs price", xlabel="GrLivArea (sq ft)", ylabel="SalePrice ($)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "03_outliers.png", dpi=120)
plt.show()
train.loc[outlier, ["Id", "GrLivArea", "SalePrice", "OverallQual", "SaleCondition"]]

# %% [markdown]
# ### 1.5 Which numeric columns move with price?
# Correlation only sees straight-line relationships, and it ignores the 43 text columns.
# It is a quick map, not a feature-selection rule.

# %%
correlations = (train.select_dtypes("number").drop(columns="Id")
                .corr()["SalePrice"].drop("SalePrice").sort_values())
top10 = correlations.tail(10)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top10.index, top10.values, color="steelblue")
ax.set(title="Top 10 correlations with SalePrice", xlabel="Pearson correlation")
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "04_top_correlations.png", dpi=120)
plt.show()
top10.iloc[::-1].round(3)

# %% [markdown]
# ### What EDA tells us to do next
# | Finding | Action | Step |
# |---|---|---|
# | SalePrice is right-skewed; the metric is on log scale | Predict log(SalePrice) | Step 2 |
# | Many NAs mean "doesn't have one" | Fill them with an explicit "absent" label | Step 3 |
# | Quality and size dominate; size is spread across many columns | Build total-size and age features | Step 4 |
# | Area columns are skewed like the price | Take log1p of skewed areas | Step 5 |
# | Two huge-but-cheap houses | Test dropping them from training | Step 6 |
