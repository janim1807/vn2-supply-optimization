import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4, 8, 13, 52]
ROLLING_WINDOWS = [4, 8, 13]


def mask_censored_demand(df):
    df = df.copy()
    df["demand"] = df["sales"].where(df["in_stock"], other=np.nan)
    return df


def add_features(df):
    df = df.copy()
    df = df.sort_values(["sku_id", "week"])

    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year

    for period in [1, 2]:
        df[f"fourier_sin_{period}"] = np.sin(2 * np.pi * period * df["week_of_year"] / 52)
        df[f"fourier_cos_{period}"] = np.cos(2 * np.pi * period * df["week_of_year"] / 52)

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("sku_id", group_keys=False)["demand"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = df.groupby("sku_id", group_keys=False)["demand"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    df["intermittency_rate"] = df.groupby("sku_id", group_keys=False)["demand"].transform(
        lambda s: (s.shift(1) == 0).rolling(13, min_periods=1).mean()
    )

    return df


def add_per_series_scaling(df):
    df = df.copy()
    df["scale"] = (
        df.groupby("sku_id")["demand"]
        .transform(lambda s: s.shift(1).rolling(53, min_periods=4).mean())
        .clip(lower=1.0)
        .fillna(1.0)
    )
    df["demand_scaled"] = df["demand"] / df["scale"]
    return df


def add_time_decay_weights(df, reference_week=None):
    df = df.copy()
    if reference_week is None:
        reference_week = df["week"].max()

    weeks_ago = (reference_week - df["week"]).dt.days / 7
    df["sample_weight"] = np.where(weeks_ago <= 52, 1.0, np.where(weeks_ago <= 104, 0.5, 0.25))
    return df


FEATURE_COLS = (
    ["week_of_year", "month", "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2"]
    + [f"lag_{lag}" for lag in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + ["intermittency_rate"]
)

CATEGORICAL_COLS = ["Store", "Product", "ProductGroup", "Department", "StoreFormat", "Format"]
