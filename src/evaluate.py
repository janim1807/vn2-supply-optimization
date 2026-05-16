import numpy as np
import pandas as pd
from src.policy import CS, CH

BENCHMARK_COST = 4.334
WINNER_COST = 3.763


def summarize_costs(results):
    total = results["total_cost"].sum()
    return {
        "total_cost": round(total, 4),
        "avg_cost_per_sku_per_week": round(results["total_cost"].mean(), 4),
        "shortage_fraction": round(results["shortage_cost"].sum() / total, 3) if total > 0 else 0,
        "holding_fraction": round(results["holding_cost"].sum() / total, 3) if total > 0 else 0,
    }


def worst_skus(results, n=10):
    return (
        results.groupby("sku_id")["total_cost"].mean()
        .sort_values(ascending=False).head(n)
        .reset_index().rename(columns={"total_cost": "avg_weekly_cost"})
    )
