"""Leakage-aware building blocks for the House Prices coursework experiments."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


ALL_GROUPS = ("semantic", "size", "age", "quality", "log")


def read_competition_csv(path):
    # `None` is a real masonry category. Only the documented NA token is missing.
    return pd.read_csv(path, keep_default_na=False, na_values=["NA", ""])


def load_data(path):
    path = Path(path)
    train = read_competition_csv(path / "train.csv")
    test = read_competition_csv(path / "test.csv")
    sample = read_competition_csv(path / "sample_submission.csv")
    if "SalePrice" not in train or "SalePrice" in test:
        raise ValueError("Expected labeled train.csv and unlabeled test.csv.")
    if train.drop(columns="SalePrice").columns.tolist() != test.columns.tolist():
        raise ValueError("Train/test predictor schemas differ.")
    if not train.Id.is_unique or not test.Id.is_unique or not sample.Id.is_unique:
        raise ValueError("Ids must be unique in each file.")
    if set(train.Id) & set(test.Id):
        raise ValueError("Training and test Ids overlap.")
    if test.Id.tolist() != sample.Id.tolist():
        raise ValueError("Sample submission and test Id order differ.")
    if not np.isfinite(train.SalePrice).all() or (train.SalePrice <= 0).any():
        raise ValueError("Training prices must be positive and finite.")
    return train, test, sample


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Deterministic row-wise features; no target or population statistics."""
    def __init__(self, groups=ALL_GROUPS):
        self.groups = groups

    def fit(self, X, y=None):
        invalid = set(self.groups) - set(ALL_GROUPS)
        if invalid:
            raise ValueError(f"Unknown feature groups: {invalid}")
        self.feature_names_in_ = [c for c in X.columns if c not in ("Id", "SalePrice")]
        return self

    def transform(self, X):
        data = X.loc[:, self.feature_names_in_].copy()
        if "MSSubClass" in data:
            data["MSSubClass"] = data["MSSubClass"].map(
                lambda value: str(int(value)) if pd.notna(value) else np.nan)
        if "semantic" in self.groups:
            absent_groups = [
                ("TotalBsmtSF", ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"), "NoBasement"),
                ("Fireplaces", ("FireplaceQu",), "NoFireplace"),
                ("PoolArea", ("PoolQC",), "NoPool"),
            ]
            for measure, columns, absence in absent_groups:
                if measure in data:
                    for column in columns:
                        if column in data:
                            fill = pd.Series(np.where(data[measure].eq(0), absence, "Unknown"), index=data.index)
                            data[column] = data[column].fillna(fill)
            if {"GarageCars", "GarageArea"}.issubset(data.columns):
                no_garage = data.GarageCars.eq(0) & data.GarageArea.eq(0)
                for column in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
                    if column in data:
                        data[column] = data[column].fillna(pd.Series(
                            np.where(no_garage, "NoGarage", "Unknown"), index=data.index))
            for column, absence in (("Alley", "NoAlley"), ("Fence", "NoFence"),
                                    ("MiscFeature", "NoMiscFeature"), ("MasVnrType", "Unknown")):
                if column in data:
                    data[column] = data[column].fillna(absence)
            if {"GarageYrBlt", "YrSold"}.issubset(data.columns):
                # A reported 2207 garage date is impossible for a 2000s sale;
                # keep a flag and impute inside each fold instead of guessing 2007.
                impossible = (data.GarageYrBlt > data.YrSold + 2) | (data.GarageYrBlt < 1700)
                data["GarageYearInvalid"] = impossible.astype(int)
                data.loc[impossible, "GarageYrBlt"] = np.nan

        if "size" in self.groups:
            totals = {
                "TotalSF": ("GrLivArea", "TotalBsmtSF"),
                "TotalFinishedSF": ("GrLivArea", "BsmtFinSF1", "BsmtFinSF2"),
                "TotalPorchSF": ("WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"),
            }
            for name, columns in totals.items():
                if set(columns).issubset(data.columns):
                    data[name] = data[list(columns)].sum(axis=1, min_count=len(columns))
            bathrooms = ("FullBath", "BsmtFullBath", "HalfBath", "BsmtHalfBath")
            if set(bathrooms).issubset(data.columns):
                data["TotalBathrooms"] = (data.FullBath + data.BsmtFullBath +
                                            0.5 * (data.HalfBath + data.BsmtHalfBath))
            for column, name in (("TotalBsmtSF", "HasBasement"), ("GarageArea", "HasGarage"),
                                 ("PoolArea", "HasPool"), ("Fireplaces", "HasFireplace"),
                                 ("2ndFlrSF", "HasSecondFloor")):
                if column in data:
                    data[name] = np.where(data[column].isna(), np.nan, data[column].gt(0).astype(float))

        if "age" in self.groups and "YrSold" in data:
            for column, name in (("YearBuilt", "HouseAge"), ("YearRemodAdd", "RemodelAge"),
                                 ("GarageYrBlt", "GarageAge")):
                if column in data:
                    data[name] = (data.YrSold - data[column]).clip(lower=0)
            if "YearRemodAdd" in data:
                data["RemodelAfterSale"] = data.YearRemodAdd.gt(data.YrSold).astype(int)
                if "YearBuilt" in data:
                    data["WasRemodeled"] = data.YearRemodAdd.gt(data.YearBuilt).astype(int)

        if "quality" in self.groups:
            quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1,
                       "NoBasement": 0, "NoGarage": 0, "NoFireplace": 0, "NoPool": 0}
            for column in ("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                           "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"):
                if column in data:
                    data[column + "Score"] = data[column].map(quality)
            if {"OverallQual", "GrLivArea"}.issubset(data.columns):
                data["QualityLivingArea"] = data.OverallQual * data.GrLivArea
            if {"OverallQual", "TotalBsmtSF", "GrLivArea"}.issubset(data.columns):
                data["QualityTotalArea"] = data.OverallQual * (data.GrLivArea + data.TotalBsmtSF)

        if "log" in self.groups:
            columns = ("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
                       "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
                       "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                       "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "TotalSF",
                       "TotalFinishedSF", "TotalPorchSF", "QualityLivingArea", "QualityTotalArea")
            for column in columns:
                if column in data:
                    if data[column].dropna().lt(0).any():
                        raise ValueError(f"Unexpected negative area/value in {column}")
                    data[column] = np.log1p(data[column])
        return data


class NativeCatBoost(RegressorMixin, BaseEstimator):
    """Keep native categories, with unknown category markers and numeric NaNs."""
    def __init__(self, parameters=None, seed=42):
        self.parameters = parameters
        self.seed = seed

    def _prepare(self, X):
        data = X.copy()
        for column in self.categorical_columns_:
            data[column] = data[column].fillna("Unknown").astype(str)
        return data

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.categorical_columns_ = X.select_dtypes(exclude=np.number).columns.tolist()
        settings = dict(iterations=1800, depth=5, learning_rate=0.035, l2_leaf_reg=5,
                        loss_function="RMSE", random_seed=self.seed, thread_count=4,
                        verbose=False, allow_writing_files=False)
        settings.update(self.parameters or {})
        self.model_ = CatBoostRegressor(**settings, cat_features=self.categorical_columns_)
        self.model_.fit(self._prepare(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._prepare(X))


def make_model(kind, groups=ALL_GROUPS, parameters=None, seed=42):
    parameters = parameters or {}
    features = FeatureBuilder(groups=groups)
    if kind == "catboost":
        return Pipeline([("features", features), ("model", NativeCatBoost(parameters, seed))])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                                   keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown",
                                                       keep_empty_features=True)),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    if kind == "dummy":
        model = DummyRegressor(strategy="mean")
    elif kind == "ridge":
        model = Ridge(**({"alpha": 20.0} | parameters))
    elif kind == "elasticnet":
        model = ElasticNet(**({"alpha": 0.0005, "l1_ratio": 0.9, "max_iter": 30000,
                               "tol": 1e-5, "random_state": seed} | parameters))
    elif kind == "svr":
        model = SVR(**({"C": 20, "epsilon": 0.008, "gamma": 0.0005} | parameters))
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        settings = dict(n_estimators=1800, max_depth=3, learning_rate=0.025,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                        reg_alpha=0.02, reg_lambda=5, tree_method="hist", n_jobs=4,
                        random_state=seed, objective="reg:squarederror")
        model = XGBRegressor(**(settings | parameters))
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([("features", features), ("preprocess", preprocess), ("model", model)])


def make_folds(n_rows, seed=42):
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_rows)))


def training_keep_mask(X, y, exclude_outliers=False):
    if exclude_outliers:
        return ~((X.GrLivArea.to_numpy() > 4000) & (np.asarray(y) < 300000))
    return np.ones(len(X), dtype=bool)


def evaluate_model(model, X, y, X_test, folds, exclude_outliers=False):
    started = time.perf_counter()
    y = np.asarray(y, dtype=float)
    y_log = np.log(y)
    oof = np.full(len(X), np.nan)
    coverage = np.zeros(len(X), dtype=int)
    test_predictions, fold_records = [], []
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        if set(train_idx) & set(valid_idx):
            raise ValueError("Training and validation indices overlap.")
        keep = training_keep_mask(X.iloc[train_idx], y[train_idx], exclude_outliers)
        fit_idx = train_idx[keep]
        estimator = clone(model)
        estimator.fit(X.iloc[fit_idx], y_log[fit_idx])
        pred = estimator.predict(X.iloc[valid_idx])
        oof[valid_idx] = pred
        coverage[valid_idx] += 1
        test_predictions.append(estimator.predict(X_test))
        fold_records.append({"fold": fold, "training_rows": len(fit_idx),
                             "removed_training_rows": int((~keep).sum()),
                             "validation_rows": len(valid_idx),
                             "rmse": float(np.sqrt(np.mean((y_log[valid_idx] - pred) ** 2)))})
    if not (coverage == 1).all() or not np.isfinite(oof).all():
        raise ValueError("Every training row must have exactly one finite OOF prediction.")
    test_log = np.mean(test_predictions, axis=0)
    if not np.isfinite(test_log).all():
        raise ValueError("Non-finite test predictions.")
    errors = [row["rmse"] for row in fold_records]
    return {"cv_rmse": float(np.sqrt(np.mean((y_log - oof) ** 2))),
            "fold_mean": float(np.mean(errors)), "fold_std": float(np.std(errors)),
            "seconds": time.perf_counter() - started, "folds": fold_records,
            "oof_log_predictions": oof, "test_log_predictions": test_log}


def make_submission(ids, log_predictions, sample):
    ids = pd.Index(ids)
    predictions = np.asarray(log_predictions, dtype=float)
    if (predictions.ndim != 1 or len(ids) != len(predictions) or len(ids) != len(sample)
            or not ids.is_unique or not sample.Id.is_unique or set(ids) != set(sample.Id)):
        raise ValueError("Submission Ids must match the sample exactly, once each.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite.")
    with np.errstate(over="ignore", under="ignore"):
        prices = np.exp(predictions)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Dollar predictions must be positive and finite.")
    aligned = pd.Series(prices, index=ids).reindex(sample.Id)
    return pd.DataFrame({"Id": sample.Id.to_numpy(), "SalePrice": aligned.to_numpy()})
