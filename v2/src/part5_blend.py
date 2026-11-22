# %% [markdown]
# ## Part 5: Blending, and the whole pathway at a glance
#
# ### Why averaging models can beat the best model
# Each model makes errors. If two models make **different** errors (one guesses too high where
# the other guesses too low), averaging them cancels part of both. If they make the **same**
# errors, averaging changes nothing. So the useful question is not just "how good is each
# model?" but "**how differently do they go wrong?**"
#
# We measure that with the correlation between their out-of-fold **residuals** (true log price
# minus predicted log price). 1.0 means identical mistakes. Lower means more to gain from
# combining.

# %%
residuals = pd.DataFrame({name: y_log - oof[name] for name in names})
residual_corr = residuals.corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
image = ax.imshow(residual_corr, cmap="Blues", vmin=0.7, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        value = residual_corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.93 else "black")  # readable on dark cells
fig.colorbar(image, label="correlation of OOF residuals")
ax.set_title("How similarly do the models go wrong?")
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "06_oof_correlation.png", dpi=120)
plt.show()

# %% [markdown]
# **Reading the heatmap.**
# - Ridge, Lasso and ElasticNet correlate at 0.99–1.00: they make almost the same mistakes, so
#   blending them with each other gains little.
# - Linear models vs boosted trees: about 0.88, the most different pair among the good models.
# - XGBoost vs CatBoost: 0.93. They are both boosting, but they treat categories differently
#   (one-hot vs target statistics).
# - SVR sits in between (0.89–0.92).
#
# ### Choosing the blend
# **Rule, fixed before looking at blend scores:** take the best model from each family, plus
# both boosters because they handle categories differently. That gives **Lasso + XGBoost +
# CatBoost + SVR**, with **equal weights**. We skip Random Forest (much weaker), and we skip
# Ridge and ElasticNet (near-copies of Lasso).
#
# **Why equal weights instead of tuning them?** Weights tuned on the same 1,460 OOF
# predictions would fit those predictions' noise. That is overfitting again, one level up.
# Equal weights are a choice we can defend without looking at the score.

# %%
blend_names = ["lasso", "xgb", "catboost", "svr"]
blend_oof = np.mean([oof[name] for name in blend_names], axis=0)
blend_test = np.mean([test_preds[name] for name in blend_names], axis=0)
blend_score = np.sqrt(np.mean((y_log - blend_oof) ** 2))
record("12", "Blend: Lasso + XGBoost + CatBoost + SVR, equal weights", blend_score)
save_submission("s6_blend", blend_test)

# %% [markdown]
# For context only (these did **not** choose the blend), here is what simpler or larger
# blends score. Throwing in every model is worse, because three near-identical linear models
# then take three of the six votes.

# %%
def blend_rmse(chosen):
    return np.sqrt(np.mean((y_log - np.mean([oof[n] for n in chosen], axis=0)) ** 2))

pd.Series({
    f"best single model ({comparison.index[0]})": comparison.cv_rmse.min(),
    "XGBoost + CatBoost": blend_rmse(["xgb", "catboost"]),
    "Lasso + XGBoost": blend_rmse(["lasso", "xgb"]),
    "chosen 4-model blend": blend_score,
    "all models except Random Forest": blend_rmse([n for n in names if n != "rf"]),
}).round(5).to_frame("cv_rmse")

# %% [markdown]
# ### The whole pathway: improvement staircase

# %%
steps = results[results.step != "0"].reset_index(drop=True)  # step 0 (0.41) would flatten the chart
colors = ["gray" if s == "1" else "orange" if s in {"2", "3", "4", "5", "6", "6b"}
          else "steelblue" if s != "12" else "seagreen" for s in steps.step]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(steps.step, steps.cv_rmse, color=colors)
for i, value in enumerate(steps.cv_rmse):
    ax.text(i, value + 0.0005, f"{value:.4f}", ha="center", fontsize=7)
ax.set(title="CV RMSE by step (gray: baseline, orange: feature engineering, "
             "blue: models, green: blend)",
       xlabel="step", ylabel="CV RMSE (log price)", ylim=(0.11, 0.16))
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "07_improvement_staircase.png", dpi=120)
plt.show()
print(f"Step 0 (predict the mean) scored {results.cv_rmse.iloc[0]:.4f}; it is left off the chart.")

# %% [markdown]
# ### Submissions made in this notebook
# Upload them in this order and record each Kaggle score. The jump from **S2 to S3** changes one
# thing (log of skewed areas), which makes it the cleanest "best improvement" case to explain.

# %%
submission_steps = {"s1_ridge_raw": "1", "s2_new_features": "4", "s3_log_features": "5",
                    "s4_no_outliers": "6b", "s5_catboost": "10", "s6_blend": "12"}
cv_by_step = results.set_index("step").cv_rmse
summary = pd.DataFrame({
    "submission": list(submission_steps),
    "step": list(submission_steps.values()),
    "cv_rmse": [cv_by_step[s] for s in submission_steps.values()],
})
summary.to_csv(OUT_DIR / "submission_summary.csv", index=False)
summary.round(5)

# %% [markdown]
# ## Summary
# - **Most useful feature engineering:** log1p of skewed area columns (Step 5), −12.4% CV RMSE
#   on its own. Price scales with size in proportion, so log price is close to linear in log
#   area. Together with the log target, that matches what a linear model can express.
# - **Best single models:** gradient boosting (XGBoost, CatBoost). Bagging (Random Forest) was
#   the weakest, because averaging can't remove the bias that every tree shares.
# - **Blend (Step 12):** an equal-weight blend of four models from different families. It beats
#   the best single model, because their errors are only partly correlated.
# - **Limits:** every choice was made with the same 5-fold CV, so the CV scores are slightly
#   optimistic. The Kaggle leaderboard is the independent check.
