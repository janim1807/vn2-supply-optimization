import numpy as np
import pandas as pd

from src.policy import Q_STAR

MIN_SAMPLES = 8
DECAY_HALFLIFE = 26


class AdaptiveConformal:
    def __init__(self, alpha_init=Q_STAR, gamma=0.05):
        self.alpha = alpha_init
        self.gamma = gamma
        self._history = []

    def update(self, actuals, targets):
        sku_ids = [s for s in actuals if s in targets]
        if not sku_ids:
            return
        covered = np.mean([float(actuals[s]) <= targets[s] for s in sku_ids])
        self.alpha = float(np.clip(self.alpha + self.gamma * (Q_STAR - covered), 0.50, 0.99))
        self._history.append({"coverage": covered, "alpha": self.alpha})

    def log(self):
        if not self._history:
            return f"α={self.alpha:.3f} (no updates)"
        return f"α={self.alpha:.3f}  coverage={self._history[-1]['coverage']:.1%}"


def compute_all_residuals(df, models, start_week=None):
    from src.features import FEATURE_COLS, CATEGORICAL_COLS
    from src.models import predict, blend_forecasts

    cols = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in df.columns]
    weeks = sorted(df["week"].unique())
    per_sku = {}

    for week in weeks:
        if start_week is not None and week < start_week:
            continue
        future = [w for w in weeks if w > week]
        if len(future) < 3:
            continue

        row = df[df["week"] == week].set_index("sku_id")
        if row.empty:
            continue

        X = row[cols].copy()
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")

        m = models[3]
        preds = blend_forecasts(predict(m["lgbm"], X), predict(m["catboost"], X), m["w_lgbm"], m["w_cb"])
        actuals = df[df["week"] == future[2]].set_index("sku_id")["demand"]

        for sku_id, pred in zip(X.index, preds):
            actual = actuals.get(sku_id, np.nan)
            if np.isnan(actual):
                continue
            rel = (float(actual) - float(pred)) / max(float(pred), 1.0)
            entry = per_sku.setdefault(sku_id, {"weeks": [], "residuals": []})
            entry["weeks"].append(week)
            entry["residuals"].append(rel)

    return {
        sku_id: {
            "weeks": np.array(d["weeks"], dtype="datetime64[ns]"),
            "relative_residuals": np.array(d["residuals"]),
        }
        for sku_id, d in per_sku.items()
    }


def compute_safety_margins(residuals_by_sku, reference_week, current_forecasts, alpha=Q_STAR):
    all_rel = np.concatenate([
        v["relative_residuals"] for v in residuals_by_sku.values()
    ]) if residuals_by_sku else np.array([0.0])
    global_q = float(_weighted_quantile(all_rel, None, alpha))

    margins = {}
    for sku_id, fc in current_forecasts.items():
        fc = max(fc, 0.0)
        data = residuals_by_sku.get(sku_id)

        if data is None or len(data["relative_residuals"]) < MIN_SAMPLES:
            ratio = global_q
        else:
            weeks = pd.to_datetime(data["weeks"])
            w = _decay_weights(weeks, reference_week)
            ratio = float(_weighted_quantile(data["relative_residuals"], w, alpha))

        margins[sku_id] = fc * ratio
    return margins


def _decay_weights(weeks, ref_week):
    ago = np.array([(ref_week - w).days / 7 for w in weeks], dtype=float)
    w = np.exp(-ago * np.log(2) / DECAY_HALFLIFE)
    return w / w.sum()


def _weighted_quantile(values, weights, q):
    n = len(values)
    q_adj = min(q * (1 + 1 / n), 1.0)

    if weights is None:
        return float(np.quantile(values, q_adj))

    idx = np.argsort(values)
    cumw = np.cumsum(weights[idx])
    i = min(np.searchsorted(cumw, q_adj), n - 1)
    return float(values[idx[i]])
