"""Slide-sized figures in the deck's palette (background #F6F4EF / #ECE8DF, sans-serif, large type).
Run from the repo root: .venv/bin/python report/make_slide_figures.py"""
import contextlib, io, os, runpy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "slide_figures"
INK, SOFT, ACCENT, BLUE, GREY = "#1B2A41", "#44516A", "#C4511C", "#2F5F94", "#9AA3B2"
LIGHT, ALT = "#F6F4EF", "#ECE8DF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.edgecolor": GREY,
                     "axes.spines.top": False, "axes.spines.right": False})
train = pd.read_csv(ROOT / "house-prices-advanced-regression-techniques" / "train.csv",
                    keep_default_na=False, na_values=["NA", ""])

def save(fig, name, bg):
    fig.patch.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(bg)
    fig.savefig(OUT / name, dpi=200, facecolor=bg)
    plt.close(fig)

# price histograms: 1664 x 500 px
fig, axes = plt.subplots(1, 2, figsize=(8.32, 2.5))
axes[0].hist(train.SalePrice / 1000, bins=45, color=GREY)
axes[0].set(xlabel="sale price ($1,000)", yticks=[])
axes[1].hist(np.log(train.SalePrice), bins=45, color=ACCENT)
axes[1].set(xlabel="log of sale price", yticks=[])
for ax in axes:
    ax.spines["left"].set_visible(False)
fig.tight_layout(w_pad=4)
save(fig, "price_hist.png", LIGHT)

# outliers scatter: 900 x 640 px
out = (train.GrLivArea > 4000) & (train.SalePrice < 300000)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.scatter(train.GrLivArea, train.SalePrice / 1000, s=6, color=BLUE, alpha=0.35, linewidths=0)
ax.scatter(train.GrLivArea[out], train.SalePrice[out] / 1000, s=110, facecolors="none", edgecolors=ACCENT, linewidths=2.2)
ax.set(xlabel="living area (sq ft)", ylabel="price ($1,000)", xlim=(0, 6000))
fig.tight_layout()
save(fig, "outliers.png", ALT)

# CV vs Kaggle by submission: 1000 x 660 px
k = pd.read_csv(ROOT / "v2" / "results" / "kaggle_log.csv")
x = np.arange(1, len(k) + 1)
fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.axhline(0.11745, ls=(0, (4, 3)), color=GREY, lw=1.5)
ax.text(0.85, 0.1168, "first attempt 0.11745", va="top", fontsize=11, color=SOFT)
ax.plot(x, k.cv_rmse, "o-", color=BLUE, lw=2, ms=7, label="our CV")
ax.plot(x, k.kaggle_score, "s-", color=ACCENT, lw=2, ms=7, label="Kaggle")
ax.annotate("0.12037", (6, 0.12037), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=11, color=INK)
ax.annotate("0.11624", (7, 0.11624), xytext=(-14, -4), textcoords="offset points", ha="right", va="top", fontsize=11, color=INK, weight="bold")
ax.set_xticks(x, [f"S{i}" for i in x])
ax.set(ylabel="RMSE (log price)", ylim=(0.110, 0.16), xlim=(0.6, 7.4))
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
save(fig, "leaderboard.png", LIGHT)

# residual correlation heatmap: 920 x 800 px, recomputed from the notebook's OOF predictions
os.chdir(ROOT / "v2")
# the notebook parts save their own figures; skip that here so v2/results/figures stays untouched
_savefig = matplotlib.figure.Figure.savefig
matplotlib.figure.Figure.savefig = lambda *a, **k: None
ns = runpy.run_path("src/part1_eda.py")
ns["OUT_DIR"] = ROOT / "report" / "slide_figures" / "_tmp"
(ns["OUT_DIR"] / "figures").mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    for part in ("part2_baseline.py", "part3_features.py", "part4_models.py"):
        exec(open("src/" + part).read(), ns)
matplotlib.figure.Figure.savefig = _savefig
names = ["ridge", "lasso", "elasticnet", "rf", "xgb", "catboost", "svr"]
labels = ["Ridge", "Lasso", "ElasticNet", "R. Forest", "XGBoost", "CatBoost", "SVR"]
res = pd.DataFrame({n: ns["y_log"] - ns["oof"][n] for n in names}).corr().to_numpy()
fig, ax = plt.subplots(figsize=(4.6, 4.0))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("deck", [ALT, BLUE, INK])
ax.imshow(res, cmap=cmap, vmin=0.7, vmax=1)
ax.set_xticks(range(7), labels, rotation=40, ha="right", fontsize=12)
ax.set_yticks(range(7), labels, fontsize=12)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{res[i, j]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if res[i, j] > 0.8 else INK)
ax.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=ACCENT, lw=3.5))
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
save(fig, "heatmap.png", ALT)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

# cropped leaderboard row for the result slide (our row and the "new personal best" banner)
from PIL import Image
Image.open(ROOT / "report" / "figures" / "kaggle_screenshot.png").crop((62, 852, 1792, 1078)).save(OUT / "kaggle_row.png")
