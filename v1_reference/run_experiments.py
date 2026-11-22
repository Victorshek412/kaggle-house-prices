"""Run a bounded, reproducible experimental session and save coursework evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import house_prices as hp


def find_data_dir():
    candidates = [Path("house-prices-advanced-regression-techniques"),
                  Path("/kaggle/input/house-prices-advanced-regression-techniques")]
    for path in candidates:
        if (path / "train.csv").exists():
            return path
    raise FileNotFoundError("Set --data-dir to the folder containing the four official files.")


def specifications():
    groups = hp.ALL_GROUPS
    experiments = [dict(experiment="00_constant", model="dummy", groups=(), parameters={})]
    for index, active in enumerate([(), groups[:1], groups[:2], groups[:3], groups[:4], groups], start=1):
        suffix = "baseline" if not active else active[-1]
        experiments.append(dict(experiment=f"{index:02d}_ridge_{suffix}", model="ridge",
                                groups=active, parameters={"alpha": 20.0}))
    for alpha in (5., 50., 100.):
        experiments.append(dict(experiment=f"ridge_alpha_{int(alpha)}", model="ridge",
                                groups=groups, parameters={"alpha": alpha}))
    for alpha in (0.0005, 0.001):
        experiments.append(dict(experiment=f"elasticnet_{alpha}", model="elasticnet",
                                groups=groups, parameters={"alpha": alpha, "l1_ratio": 0.9}))
    experiments.append(dict(experiment="svr", model="svr", groups=groups, parameters={}))
    for depth in (4, 6):
        experiments.append(dict(experiment=f"catboost_depth_{depth}", model="catboost",
                                groups=groups[:-1], parameters={"depth": depth}))
    for depth in (2, 3):
        experiments.append(dict(experiment=f"xgboost_depth_{depth}", model="xgboost",
                                groups=groups[:-1], parameters={"max_depth": depth}))
    return experiments


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_session(data_dir=None, output_dir=Path("."), include_outliers=True):
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    predictions_dir = results_dir / "predictions"
    figures_dir = results_dir / "figures"
    submission_dir = output_dir / "submissions"
    for path in (results_dir, predictions_dir, figures_dir, submission_dir):
        path.mkdir(parents=True, exist_ok=True)
    train, test, sample = hp.load_data(data_dir)
    X, y = train.drop(columns="SalePrice"), train.SalePrice
    folds = hp.make_folds(len(train), seed=42)
    fold_assignment = np.zeros(len(train), dtype=int)
    for fold, (_, valid) in enumerate(folds, 1):
        fold_assignment[valid] = fold
    pd.DataFrame({"Id": train.Id, "fold": fold_assignment}).to_csv(results_dir / "fold_assignments.csv", index=False)
    package_names = ("numpy", "pandas", "scikit-learn", "catboost", "xgboost", "scipy", "matplotlib")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "packages": {name: importlib.metadata.version(name) for name in package_names},
                  "data_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                                  for name in ("train.csv", "test.csv", "sample_submission.csv", "data_description.txt")},
                  "seed": 42, "folds": 5, "metric": "RMSE(log(SalePrice), predicted_log_price)",
                  "training_rows": len(train), "test_rows": len(test),
                  "test_prediction_method": "Mean of five fold-model log predictions, exponentiated once",
                  "outlier_rule": "Training rows only: GrLivArea > 4000 and SalePrice < 300000",
                  "external_price_labels_used": False, "kaggle_submitted": False}
    # In a standalone notebook the embedded modules have no on-disk __file__.
    source_paths = [Path("house_prices.py"), Path("run_experiments.py")]
    provenance["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_paths if path.exists()}
    save_json(results_dir / "provenance.json", provenance)
    (output_dir / "requirements-resolved.txt").write_text("\n".join(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata['Name'].lower())) + "\n")

    records, fold_records, predictions, configs = [], [], {}, {}

    def run_one(spec, exclude=False, seed=42, label=None):
        name = label or spec["experiment"]
        print(f"START {name}: {spec['model']} | training exclusion={exclude} | seed={seed}", flush=True)
        model = hp.make_model(spec["model"], groups=spec["groups"], parameters=spec["parameters"], seed=seed)
        result = hp.evaluate_model(model, X, y, test, hp.make_folds(len(train), seed), exclude_outliers=exclude)
        row = {"experiment": name, "model": spec["model"], "feature_groups": "+".join(spec["groups"]),
               "exclude_outliers": exclude, "seed": seed,
               **{key: result[key] for key in ("cv_rmse", "fold_mean", "fold_std", "seconds")}}
        records.append(row)
        for fold_row in result["folds"]:
            fold_records.append({"experiment": name, **fold_row})
        predictions[name] = result
        configs[name] = {**spec, "exclude_outliers": exclude, "seed": seed}
        np.savez_compressed(predictions_dir / f"{name}.npz", oof_log=result["oof_log_predictions"],
                            test_log=result["test_log_predictions"])
        hp.make_submission(test.Id, result["test_log_predictions"], sample).to_csv(submission_dir / f"{name}.csv", index=False)
        pd.DataFrame(records).to_csv(results_dir / "experiments.csv", index=False)
        pd.DataFrame(fold_records).to_csv(results_dir / "fold_scores.csv", index=False)
        save_json(results_dir / "configurations.json", configs)
        print(f"DONE  {name}: OOF log-RMSE={result['cv_rmse']:.6f}, fold SD={result['fold_std']:.6f}, {result['seconds']:.1f}s", flush=True)
        return row

    experiment_specs = specifications()
    for spec in experiment_specs:
        run_one(spec)

    primary = pd.DataFrame(records)
    best_linear_name = primary[primary.model.isin(["ridge", "elasticnet", "svr"])].sort_values("cv_rmse").iloc[0].experiment
    best_tree_name = primary[primary.model.isin(["catboost", "xgboost"])].sort_values("cv_rmse").iloc[0].experiment
    if include_outliers:
        for name in (best_linear_name, best_tree_name):
            run_one(configs[name], exclude=True, label=name + "_exclude")

    # Leave-one-group-out measures conditional contribution with a fixed model/settings.
    full_ridge = next(spec for spec in experiment_specs if spec["experiment"] == "06_ridge_log")
    ablations = []
    full_score = predictions["06_ridge_log"]["cv_rmse"]
    for group in hp.ALL_GROUPS:
        spec = {**full_ridge, "experiment": "ablation_without_" + group,
                "groups": tuple(item for item in hp.ALL_GROUPS if item != group)}
        row = run_one(spec)
        ablations.append({"feature_group": group, "full_model_rmse": full_score,
                          "without_group_rmse": row["cv_rmse"],
                          "benefit_of_group": row["cv_rmse"] - full_score,
                          "interpretation": "Positive benefit means removal worsened this fixed Ridge model."})
    pd.DataFrame(ablations).sort_values("benefit_of_group", ascending=False).to_csv(results_dir / "feature_ablation.csv", index=False)

    # Choose one linear family and one booster using development folds, then use fixed 50/50 weights.
    candidates = pd.DataFrame(records)
    candidates = candidates[~candidates.experiment.str.startswith("ablation_") & (candidates.model != "dummy")]
    linear = candidates[candidates.model.isin(["ridge", "elasticnet", "svr"])].sort_values("cv_rmse").iloc[0]
    tree = candidates[candidates.model.isin(["catboost", "xgboost"])].sort_values("cv_rmse").iloc[0]
    blend_names = [linear.experiment, tree.experiment]
    blend_oof = np.mean([predictions[name]["oof_log_predictions"] for name in blend_names], axis=0)
    blend_test = np.mean([predictions[name]["test_log_predictions"] for name in blend_names], axis=0)
    blend_fold_errors = [float(np.sqrt(np.mean((np.log(y.iloc[valid]) - blend_oof[valid]) ** 2))) for _, valid in folds]
    blend_score = float(np.sqrt(np.mean((np.log(y) - blend_oof) ** 2)))
    records.append({"experiment": "blend_50_50", "model": "blend", "feature_groups": "selected linear + booster",
                    "exclude_outliers": "component-specific", "seed": 42, "cv_rmse": blend_score,
                    "fold_mean": float(np.mean(blend_fold_errors)), "fold_std": float(np.std(blend_fold_errors)), "seconds": 0.})
    predictions["blend_50_50"] = {"oof_log_predictions": blend_oof, "test_log_predictions": blend_test, "cv_rmse": blend_score}
    configs["blend_50_50"] = {"components": blend_names, "weights": [0.5, 0.5], "space": "natural log"}
    np.savez_compressed(predictions_dir / "blend_50_50.npz", oof_log=blend_oof, test_log=blend_test)
    hp.make_submission(test.Id, blend_test, sample).to_csv(submission_dir / "blend_50_50.csv", index=False)
    for fold, error in enumerate(blend_fold_errors, 1):
        fold_records.append({"experiment": "blend_50_50", "fold": fold, "training_rows": None,
                             "removed_training_rows": None, "validation_rows": len(folds[fold-1][1]), "rmse": error})
    print(f"BLEND {blend_names}: {blend_score:.6f}", flush=True)

    # Second split: a stability diagnostic after selection, explicitly not an untouched test set.
    stability = []
    alternate_predictions = []
    for name in blend_names:
        spec = configs[name]
        row = run_one(spec, exclude=spec["exclude_outliers"], seed=2026, label=name + "_seed2026")
        stability.append({"experiment": name, "seed42_rmse": predictions[name]["cv_rmse"], "seed2026_rmse": row["cv_rmse"]})
        alternate_predictions.append(predictions[name + "_seed2026"]["oof_log_predictions"])
    alternate_blend = np.mean(alternate_predictions, axis=0)
    stability.append({"experiment": "blend_50_50", "seed42_rmse": blend_score,
                      "seed2026_rmse": float(np.sqrt(np.mean((np.log(y) - alternate_blend) ** 2)))})
    pd.DataFrame(stability).to_csv(results_dir / "stability.csv", index=False)

    final_table = pd.DataFrame(records)
    final_table.to_csv(results_dir / "experiments.csv", index=False)
    pd.DataFrame(fold_records).to_csv(results_dir / "fold_scores.csv", index=False)
    save_json(results_dir / "configurations.json", configs)
    eligible = final_table[(final_table.seed == 42) & ~final_table.experiment.str.startswith("ablation_")]
    chosen = eligible.sort_values("cv_rmse").iloc[0].experiment
    hp.make_submission(test.Id, predictions[chosen]["test_log_predictions"], sample).to_csv(submission_dir / "recommended.csv", index=False)
    oof_table = pd.DataFrame({"Id": train.Id, "SalePrice": y, "fold": fold_assignment,
                             "predicted_log_price": predictions[chosen]["oof_log_predictions"]})
    oof_table["predicted_price"] = np.exp(oof_table.predicted_log_price)
    oof_table["log_residual"] = np.log(y) - oof_table.predicted_log_price
    oof_table["absolute_log_error"] = oof_table.log_residual.abs()
    oof_table.to_csv(results_dir / "recommended_oof.csv", index=False)
    oof_table.sort_values("absolute_log_error", ascending=False).head(25).merge(
        X[["Id", "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition"]], on="Id").to_csv(results_dir / "largest_errors.csv", index=False)
    summary = {"recommended_experiment": chosen, "cv_log_rmse": float(predictions[chosen]["cv_rmse"]),
               "baseline_ridge_rmse": predictions["01_ridge_baseline"]["cv_rmse"],
               "blend_components": blend_names, "blend_cv_rmse": blend_score,
               "experiments_run": len(records), "validation_rows": len(train), "prediction_rows": len(test),
               "recommended_submission": "submissions/recommended.csv", "kaggle_score": None,
               "kaggle_rank": None, "kaggle_submitted": False,
               "limitations": ["Models and features were selected using development cross-validation; selection optimism remains.",
                               "Seed 2026 is a stability check, not an independent test set or confidence interval.",
                               "Feature ablations are conditional on this fixed Ridge model; effects can interact.",
                               "Outlier rule came from exploratory training-data analysis; all validation rows remain scored.",
                               "Historical top-percentile notebook claims do not establish a current leaderboard cutoff."],
               "stability": stability}
    save_json(results_dir / "session_summary.json", summary)
    leaderboard_log = submission_dir / "kaggle_submission_log.csv"
    if not leaderboard_log.exists():
        pd.DataFrame(columns=["submitted_at", "file", "experiment", "kaggle_score", "rank", "active_teams",
                              "previous_score", "improvement", "notes"]).to_csv(leaderboard_log, index=False)
    make_figures(train, eligible, pd.DataFrame(ablations), oof_table, figures_dir)
    print("SESSION COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def make_figures(train, scores, ablations, oof, directory):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(train.SalePrice / 1000, bins=35, color="#3875a9")
    axes[0].set(xlabel="Sale price ($ thousands)", ylabel="Number of homes", title="Original target")
    axes[1].hist(np.log(train.SalePrice), bins=35, color="#289c8e")
    axes[1].set(xlabel="Natural log of sale price", ylabel="Number of homes", title="Modeling target")
    fig.tight_layout(); fig.savefig(directory / "01_target_distribution.png"); plt.close(fig)
    view = scores[scores.model != "dummy"].sort_values("cv_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(view) * .30)))
    ax.barh(view.experiment, view.cv_rmse, color="#3875a9")
    ax.set(xlabel="Out-of-fold log-RMSE (lower is better)", title="Development comparison: every validation row scored")
    fig.tight_layout(); fig.savefig(directory / "02_model_comparison.png"); plt.close(fig)
    view = ablations.sort_values("benefit_of_group")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(view.feature_group, view.benefit_of_group,
            color=["#289c8e" if value > 0 else "#cc6b50" for value in view.benefit_of_group])
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="RMSE without group − RMSE with all groups", title="Conditional feature contribution (fixed Ridge model)")
    fig.tight_layout(); fig.savefig(directory / "03_feature_ablation.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    actual = np.log(oof.SalePrice)
    axes[0].scatter(actual, oof.predicted_log_price, s=12, alpha=.5, color="#3875a9")
    bounds = [min(actual.min(), oof.predicted_log_price.min()), max(actual.max(), oof.predicted_log_price.max())]
    axes[0].plot(bounds, bounds, color="black", lw=1)
    axes[0].set(xlabel="Actual log price", ylabel="OOF predicted log price", title="Recommended model: unseen-fold predictions")
    axes[1].scatter(oof.predicted_log_price, oof.log_residual, s=12, alpha=.5, color="#289c8e")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(xlabel="OOF predicted log price", ylabel="Actual − predicted log price", title="Where the model misses")
    fig.tight_layout(); fig.savefig(directory / "04_error_analysis.png"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--no-outlier-comparison", action="store_true")
    args = parser.parse_args()
    run_session(args.data_dir, args.output_dir, include_outliers=not args.no_outlier_comparison)
