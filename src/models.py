import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

from src.features import FEATURE_COLS, CATEGORICAL_COLS

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
}


def build_training_data(df, h, target_col="demand"):
    df = df.copy().sort_values(["sku_id", "week"])
    df["target"] = df.groupby("sku_id", group_keys=False)[target_col].shift(-h)

    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    feats = [c for c in FEATURE_COLS if c in df.columns] + cats

    mask = df["target"].notna() & df[FEATURE_COLS[0]].notna()
    train = df[mask].copy()
    for col in cats:
        train[col] = train[col].astype("category")

    X = train[feats]
    y = train["target"]
    w = train["sample_weight"] if "sample_weight" in train.columns else None
    return X, y, w


def train_lgbm(X_train, y_train, X_val, y_val, weights=None):
    cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    w = weights.values if weights is not None else None

    ds_train = lgb.Dataset(X_train, label=y_train, weight=w, categorical_feature=cats)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

    return lgb.train(
        LGBM_PARAMS, ds_train,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def train_catboost(X_train, y_train, X_val, y_val, weights=None):
    cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    w = weights.values if weights is not None else None

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
        loss_function="RMSE", eval_metric="MAE",
        cat_features=cats, early_stopping_rounds=50, verbose=False,
    )
    model.fit(X_train, y_train, sample_weight=w, eval_set=(X_val, y_val))
    return model


def train_lgbm_quantile(X_train, y_train, X_val, y_val, alpha=0.833, weights=None):
    cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    w = weights.values if weights is not None else None

    params = {**LGBM_PARAMS, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
    ds_train = lgb.Dataset(X_train, label=y_train, weight=w, categorical_feature=cats)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

    return lgb.train(
        params, ds_train,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def train_catboost_quantile(X_train, y_train, X_val, y_val, alpha=0.833, weights=None):
    cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    w = weights.values if weights is not None else None

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
        loss_function=f"Quantile:alpha={alpha}",
        eval_metric=f"Quantile:alpha={alpha}",
        cat_features=cats, early_stopping_rounds=50, verbose=False,
    )
    model.fit(X_train, y_train, sample_weight=w, eval_set=(X_val, y_val))
    return model


def predict(model, X):
    X = X.copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return np.maximum(model.predict(X), 0.0)


def blend_forecasts(preds_a, preds_b, w_a, w_b):
    return np.maximum(w_a * preds_a + w_b * preds_b, 0.0)


def optimize_blend_weights(preds_a, preds_b, y_val):
    def loss(w):
        return np.mean(np.abs(w[0] * preds_a + w[1] * preds_b - y_val))

    res = minimize(loss, [0.5, 0.5], bounds=[(0, 1), (0, 1)],
                   constraints={"type": "eq", "fun": lambda w: w[0] + w[1] - 1})
    return float(res.x[0]), float(res.x[1])
