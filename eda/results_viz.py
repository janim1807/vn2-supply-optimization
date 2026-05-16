import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.loader import load_training_data, load_initial_state, load_competition_actuals, DATA_DIR
from src.models import build_training_data, predict, blend_forecasts, optimize_blend_weights
from src.policy import Q_STAR, Z_Q
from src.conformal import AdaptiveConformal, compute_all_residuals, compute_safety_margins
from pipeline import build_features, train_horizon_models, forecast_for_week, place_orders, simulate_one_week

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALPHA_INIT = 0.65
GAMMA = 0.10


def run_with_tracking():
    print("Loading & training...")
    df_raw = load_training_data(DATA_DIR)
    inv_df = load_initial_state(DATA_DIR)
    actuals_df = load_competition_actuals(DATA_DIR)

    df_feat = build_features(df_raw)
    models = train_horizon_models(df_feat)
    residuals = compute_all_residuals(df_feat, models)

    inv_state = inv_df.set_index("sku_id")[
        ["inventory_on_hand", "in_transit_1", "in_transit_2"]
    ].to_dict(orient="index")

    weeks = sorted(actuals_df["week"].unique())
    aci = AdaptiveConformal(alpha_init=ALPHA_INIT, gamma=GAMMA)

    all_results = []
    round_data = []
    alphas = [ALPHA_INIT]
    coverages = []
    fc_h3_all = {}
    margins_all = {}
    df_rolling = df_raw.copy()

    for rnd, week in enumerate(weeks, 1):
        df_f = build_features(df_rolling)
        last = df_f["week"].max()

        fc = forecast_for_week(df_f, models, last)
        h3 = {s: f[3] for s, f in fc.items()}
        margins = compute_safety_margins(residuals, last, h3, alpha=aci.alpha)

        orders, targets = place_orders(fc, inv_state, margins)
        act = actuals_df[actuals_df["week"] == week].set_index("sku_id")["actual_sales"]
        inv_state, wk_res = simulate_one_week(inv_state, orders, act)

        aci.update(act.to_dict(), targets)

        for r in wk_res:
            r["week"] = week
            r["round"] = rnd
        all_results.extend(wk_res)

        costs = [r["total_cost"] for r in wk_res]
        round_data.append({
            "round": rnd, "week": week,
            "avg_cost": np.mean(costs),
            "avg_shortage": np.mean([r["shortage_cost"] for r in wk_res]),
            "avg_holding": np.mean([r["holding_cost"] for r in wk_res]),
        })
        alphas.append(aci.alpha)
        coverages.append(aci._history[-1]["coverage"] if aci._history else Q_STAR)
        fc_h3_all[rnd] = h3
        margins_all[rnd] = margins

        new = actuals_df[actuals_df["week"] == week][["sku_id", "week"]].copy()
        new = new.merge(actuals_df[actuals_df["week"] == week][["sku_id", "actual_sales"]], on="sku_id")
        new = new.rename(columns={"actual_sales": "sales"})
        new["in_stock"] = True
        new = new.merge(
            df_raw[["sku_id", "Store", "Product", "ProductGroup", "Division",
                    "Department", "DepartmentGroup", "StoreFormat", "Format"]].drop_duplicates("sku_id"),
            on="sku_id", how="left")
        df_rolling = pd.concat([df_rolling, new], ignore_index=True)

    return pd.DataFrame(all_results), pd.DataFrame(round_data), alphas, coverages, fc_h3_all, margins_all


def main():
    results, rounds, alphas, coverages, fc_h3, margins = run_with_tracking()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"VN2 Results — {results['total_cost'].mean():.4f}€/SKU/week | "
        f"Shortage {results['shortage_cost'].sum()/results['total_cost'].sum():.1%} / "
        f"Holding {results['holding_cost'].sum()/results['total_cost'].sum():.1%}",
        fontsize=11, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    r = rounds["round"]
    ax.bar(r - 0.2, rounds["avg_shortage"], width=0.4, label="Shortage", color="#e74c3c", alpha=0.85)
    ax.bar(r + 0.2, rounds["avg_holding"], width=0.4, label="Holding", color="#3498db", alpha=0.85)
    ax.plot(r, rounds["avg_cost"], "ko-", linewidth=1.5, markersize=5, label="Total")
    ax.set_xlabel("Round")
    ax.set_ylabel("€/SKU")
    ax.set_title("Cost by round")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    ax.plot(r, coverages, "s--", color="#9b59b6", linewidth=1.5, markersize=6, label="Coverage")
    ax.axhline(Q_STAR, color="#9b59b6", linestyle=":", alpha=0.5)
    ax.set_ylabel("Coverage", color="#9b59b6")
    ax.set_ylim(0.5, 1.0)
    ax2.plot(r, alphas[1:], "o-", color="#e67e22", linewidth=2, markersize=6, label="α")
    ax2.set_ylabel("α", color="#e67e22")
    ax2.set_ylim(0.50, 0.90)
    ax.set_xlabel("Round")
    ax.set_title("ACI evolution")
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    costs = results["total_cost"]
    p95 = np.percentile(costs, 95)
    ax.hist(costs[costs <= p95], bins=50, color="#2ecc71", alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.axvline(costs.mean(), color="red", linestyle="--", label=f"Mean: {costs.mean():.3f}€")
    ax.axvline(np.median(costs), color="blue", linestyle="--", label=f"Median: {np.median(costs):.3f}€")
    ax.set_xlabel("Cost (€)")
    ax.set_title("Cost distribution (< p95)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    worst = results.groupby("sku_id")["total_cost"].mean().sort_values(ascending=False).head(12)
    colors = ["#e74c3c" if c > 10 else "#e67e22" if c > 7 else "#f39c12" for c in worst.values]
    ax.barh(worst.index[::-1], worst.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Avg weekly cost (€)")
    ax.set_title("Worst 12 SKUs")

    ax = fig.add_subplot(gs[2, 0])
    by_sku = results.groupby("sku_id")[["shortage_cost", "holding_cost"]].mean()
    ax.scatter(by_sku["holding_cost"], by_sku["shortage_cost"], alpha=0.4, s=20, color="#3498db")
    lim = max(by_sku["shortage_cost"].max(), by_sku["holding_cost"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("Holding cost")
    ax.set_ylabel("Shortage cost")
    ax.set_title("Shortage vs Holding")

    ax = fig.add_subplot(gs[2, 1])
    skus = list(fc_h3[1].keys())
    fcs = np.array([fc_h3[1][s] for s in skus])
    mgs = np.array([margins[1][s] for s in skus])
    ax.scatter(fcs, mgs, alpha=0.35, s=15, color="#9b59b6")
    x = np.linspace(0, fcs.max(), 200)
    ax.plot(x, Z_Q * np.sqrt(x), "r--", linewidth=1.5, label="Poisson baseline")
    ax.set_xlabel("h=3 forecast")
    ax.set_ylabel("Safety margin")
    ax.set_title("Conformal margin vs forecast (R1)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    out = PLOTS_DIR / "results_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
