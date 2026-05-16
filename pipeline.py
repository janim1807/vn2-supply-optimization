from pathlib import Path
import pandas as pd
import numpy as np

from src.loader import load_training_data, load_initial_state, load_competition_actuals, DATA_DIR
from src.features import (
    mask_censored_demand, add_features, add_per_series_scaling,
    add_time_decay_weights, FEATURE_COLS, CATEGORICAL_COLS,
)
from src.models import (
    build_training_data, train_lgbm, train_catboost, predict, blend_forecasts,
    optimize_blend_weights, train_lgbm_quantile, train_catboost_quantile,
)
from src.policy import compute_order, Q_STAR
from src.conformal import AdaptiveConformal, compute_all_residuals, compute_safety_margins
from src.evaluate import summarize_costs, worst_skus

MODE = "conformal"


def build_features(df):
    df = mask_censored_demand(df)
    df = add_features(df)
    df = add_per_series_scaling(df)
    df = add_time_decay_weights(df)
    return df


def train_horizon_models(df):
    models = {}
    for h in [1, 2, 3]:
        X, y, w = build_training_data(df, h=h)
        split = int(len(X) * 0.9)

        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]
        w_tr = w.iloc[:split] if w is not None else None

        print(f"  h={h}: {len(X):,} rows — training LightGBM...")
        lgbm = train_lgbm(X_tr, y_tr, X_val, y_val, w_tr)
        print(f"    LightGBM best iter = {lgbm.best_iteration}")

        print(f"  h={h}: training CatBoost...")
        cb = train_catboost(X_tr, y_tr, X_val, y_val, w_tr)

        p_lgbm = predict(lgbm, X_val)
        p_cb = predict(cb, X_val)
        w_l, w_c = optimize_blend_weights(p_lgbm, p_cb, y_val.values)
        print(f"    Blend weights: LightGBM={w_l:.2f} / CatBoost={w_c:.2f}")

        models[h] = {"lgbm": lgbm, "catboost": cb, "w_lgbm": w_l, "w_cb": w_c}
    return models


def forecast_for_week(df_feat, models, current_week):
    cols = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in df_feat.columns]
    latest = df_feat[df_feat["week"] == current_week].set_index("sku_id")
    if latest.empty:
        return {}

    X = latest[cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    forecasts = {}
    for h in [1, 2, 3]:
        m = models[h]
        preds = blend_forecasts(predict(m["lgbm"], X), predict(m["catboost"], X), m["w_lgbm"], m["w_cb"])
        for sku_id, p in zip(X.index, preds):
            forecasts.setdefault(sku_id, {})[h] = float(p)
    return forecasts


def place_orders(forecasts, inv_state, safety_margins):
    orders, targets = {}, {}
    for sku_id, f in forecasts.items():
        st = inv_state.get(sku_id, {"inventory_on_hand": 0.0, "in_transit_1": 0.0, "in_transit_2": 0.0})
        orders[sku_id] = compute_order(
            f[1], f[2], f[3],
            st["inventory_on_hand"], st["in_transit_1"], st["in_transit_2"],
            safety_margin=safety_margins.get(sku_id),
        )
        targets[sku_id] = max(f[3] + (safety_margins.get(sku_id) or 0.0), 0.0)
    return orders, targets


def simulate_one_week(inv_state, orders, actuals):
    results = []
    new_state = {}

    for sku_id, st in inv_state.items():
        available = st["inventory_on_hand"] + st["in_transit_1"]
        demand = float(actuals.get(sku_id, 0.0))
        sales = min(available, demand)
        lost = demand - sales
        ending = available - sales

        results.append({
            "sku_id": sku_id,
            "shortage_cost": 1.0 * lost,
            "holding_cost": 0.2 * ending,
            "total_cost": 1.0 * lost + 0.2 * ending,
            "ending_inventory": ending,
            "actual_demand": demand,
        })

        new_state[sku_id] = {
            "inventory_on_hand": ending,
            "in_transit_1": st["in_transit_2"],
            "in_transit_2": float(orders.get(sku_id, 0)),
        }

    return new_state, results


def _append_actuals(df_rolling, df_raw, actuals_df, week):
    new_rows = actuals_df[actuals_df["week"] == week][["sku_id", "week"]].copy()
    new_rows = new_rows.merge(
        actuals_df[actuals_df["week"] == week][["sku_id", "actual_sales"]], on="sku_id"
    ).rename(columns={"actual_sales": "sales"})
    new_rows["in_stock"] = True
    new_rows = new_rows.merge(
        df_raw[["sku_id", "Store", "Product", "ProductGroup", "Division",
                "Department", "DepartmentGroup", "StoreFormat", "Format"]].drop_duplicates("sku_id"),
        on="sku_id", how="left"
    )
    return pd.concat([df_rolling, new_rows], ignore_index=True)


def run_conformal(df_raw, df_features, inv_state, actuals_df, weeks):
    print("\nTraining models...")
    models = train_horizon_models(df_features)

    print("\nComputing conformal residuals...")
    residuals = compute_all_residuals(df_features, models)
    n_valid = sum(1 for v in residuals.values() if len(v["relative_residuals"]) >= 8)
    print(f"  {n_valid}/{df_raw['sku_id'].nunique()} SKUs have >=8 calibration residuals")

    all_results = []
    df_rolling = df_raw.copy()
    aci = AdaptiveConformal(alpha_init=0.65, gamma=0.10)

    print(f"\nSimulating {len(weeks)} weeks...")

    for rnd, week in enumerate(weeks, 1):
        df_feat = build_features(df_rolling)
        last_week = df_feat["week"].max()

        fc = forecast_for_week(df_feat, models, last_week)
        h3_fc = {s: f[3] for s, f in fc.items()}
        margins = compute_safety_margins(residuals, last_week, h3_fc, alpha=aci.alpha)

        if rnd == 1:
            m_arr = np.array(list(margins.values()))
            print(f"  Margins (r1) — mean:{m_arr.mean():.2f} med:{np.median(m_arr):.2f} p10:{np.percentile(m_arr, 10):.2f} p90:{np.percentile(m_arr, 90):.2f}")

        orders, targets = place_orders(fc, inv_state, margins)
        actuals_week = actuals_df[actuals_df["week"] == week].set_index("sku_id")["actual_sales"]
        inv_state, week_results = simulate_one_week(inv_state, orders, actuals_week)

        aci.update(actuals_week.to_dict(), targets)

        for r in week_results:
            r["week"] = week
            r["competition_round"] = rnd
        all_results.extend(week_results)
        df_rolling = _append_actuals(df_rolling, df_raw, actuals_df, week)

        avg = np.mean([r["total_cost"] for r in week_results])
        print(f"  Round {rnd} | {week.date()} | avg: {avg:.4f}€ | {aci.log()}")

    return all_results


def run_quantile(df_raw, df_features, inv_state, actuals_df, weeks):
    print("\nTraining mean models...")
    mean_models = train_horizon_models(df_features)

    print("\nTraining quantile model (h=3)...")
    X, y, w = build_training_data(df_features, h=3)
    split = int(len(X) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    w_tr = w.iloc[:split] if w is not None else None

    lgbm_q = train_lgbm_quantile(X_tr, y_tr, X_val, y_val, Q_STAR, w_tr)
    cb_q = train_catboost_quantile(X_tr, y_tr, X_val, y_val, Q_STAR, w_tr)
    p_l = predict(lgbm_q, X_val)
    p_c = predict(cb_q, X_val)
    w_l, w_c = optimize_blend_weights(p_l, p_c, y_val.values)
    print(f"    Blend: LightGBM={w_l:.2f} / CatBoost={w_c:.2f}")
    q_model = {"lgbm": lgbm_q, "catboost": cb_q, "w_lgbm": w_l, "w_cb": w_c}

    all_results = []
    df_rolling = df_raw.copy()

    print(f"\nSimulating {len(weeks)} weeks...")

    for rnd, week in enumerate(weeks, 1):
        df_feat = build_features(df_rolling)
        last_week = df_feat["week"].max()

        fc = forecast_for_week(df_feat, mean_models, last_week)

        cols = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in df_feat.columns]
        latest = df_feat[df_feat["week"] == last_week].set_index("sku_id")
        X_pred = latest[cols].copy()
        for col in CATEGORICAL_COLS:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category")

        q_preds = blend_forecasts(
            predict(q_model["lgbm"], X_pred), predict(q_model["catboost"], X_pred),
            q_model["w_lgbm"], q_model["w_cb"],
        )
        for sku_id, qp in zip(X_pred.index, q_preds):
            if sku_id in fc:
                fc[sku_id][3] = float(qp)

        margins = {s: 0.0 for s in fc}
        orders, targets = place_orders(fc, inv_state, margins)

        actuals_week = actuals_df[actuals_df["week"] == week].set_index("sku_id")["actual_sales"]
        inv_state, week_results = simulate_one_week(inv_state, orders, actuals_week)

        for r in week_results:
            r["week"] = week
            r["competition_round"] = rnd
        all_results.extend(week_results)
        df_rolling = _append_actuals(df_rolling, df_raw, actuals_df, week)

        avg = np.mean([r["total_cost"] for r in week_results])
        n_ok = sum(1 for r in week_results if r["shortage_cost"] == 0)
        print(f"  Round {rnd} | {week.date()} | avg: {avg:.4f}€ | coverage={n_ok/len(week_results):.1%}")

    return all_results


def main():
    print(f"=== VN2 Pipeline [{MODE}] ===\n")
    print("Loading data...")
    df_raw = load_training_data(DATA_DIR)
    inv_df = load_initial_state(DATA_DIR)
    actuals_df = load_competition_actuals(DATA_DIR)

    print(f"  {df_raw['sku_id'].nunique()} SKUs | {df_raw['week'].nunique()} weeks")
    print(f"  In-stock: {df_raw['in_stock'].mean():.1%} | Zero-sales: {(df_raw['sales'] == 0).mean():.1%}")

    print("\nBuilding features...")
    df_features = build_features(df_raw)

    inv_state = inv_df.set_index("sku_id")[
        ["inventory_on_hand", "in_transit_1", "in_transit_2"]
    ].to_dict(orient="index")
    competition_weeks = sorted(actuals_df["week"].unique())

    if MODE == "conformal":
        all_results = run_conformal(df_raw, df_features, inv_state, actuals_df, competition_weeks)
    else:
        all_results = run_quantile(df_raw, df_features, inv_state, actuals_df, competition_weeks)

    results = pd.DataFrame(all_results)
    summary = summarize_costs(results)

    print(f"\n=== Results ===")
    print(f"  Avg cost/SKU/week: {summary['avg_cost_per_sku_per_week']:.4f}€")
    print(f"  Shortage/Holding:  {summary['shortage_fraction']:.1%} / {summary['holding_fraction']:.1%}")
    print(f"\nWorst 10 SKUs:")
    print(worst_skus(results).to_string(index=False))


if __name__ == "__main__":
    main()
