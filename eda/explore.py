import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.loader import load_training_data, load_initial_state

OUTPUT_DIR = Path("eda/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUTPUT_DIR / name}.png")


def plot_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Dataset Overview", fontweight="bold")

    zero_rates = df.groupby("sku_id").apply(lambda g: (g["sales"] == 0).mean())
    axes[0].hist(zero_rates, bins=30, color="#4878CF", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Zero-sales rate per SKU")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Zero-Sales Distribution")
    axes[0].axvline(zero_rates.mean(), color="red", linestyle="--", label=f"Mean {zero_rates.mean():.1%}")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(1))
    axes[0].legend()

    oos_rates = df.groupby("sku_id").apply(lambda g: (~g["in_stock"]).mean())
    axes[1].hist(oos_rates, bins=30, color="#D65F5F", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("OOS rate per SKU")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Out-of-Stock Distribution")
    axes[1].axvline(oos_rates.mean(), color="navy", linestyle="--", label=f"Mean {oos_rates.mean():.1%}")
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(1))
    axes[1].legend()

    avg_sales = df[df["in_stock"]].groupby("sku_id")["sales"].mean()
    axes[2].hist(np.log1p(avg_sales), bins=40, color="#6ACC65", edgecolor="white", linewidth=0.5)
    axes[2].set_xlabel("log(1 + avg sales)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Demand Scale")

    fig.tight_layout()
    save(fig, "01_dataset_overview")


def plot_portfolio_trends(df):
    weekly = df.groupby("week")["sales"].sum().reset_index()
    oos = df.groupby("week")["in_stock"].apply(lambda x: (~x).mean()).reset_index()
    oos.columns = ["week", "oos_rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Portfolio Trends", fontweight="bold")

    ax1.plot(weekly["week"], weekly["sales"], color="#4878CF", linewidth=1.2)
    ax1.fill_between(weekly["week"], weekly["sales"], alpha=0.15, color="#4878CF")
    ax1.set_ylabel("Total weekly sales")
    ax1.set_title("Aggregate Demand")

    ax2.plot(oos["week"], oos["oos_rate"] * 100, color="#D65F5F", linewidth=1.2)
    ax2.fill_between(oos["week"], oos["oos_rate"] * 100, alpha=0.15, color="#D65F5F")
    ax2.set_ylabel("OOS rate (%)")
    ax2.set_xlabel("Week")

    fig.tight_layout()
    save(fig, "02_portfolio_trends")


def plot_seasonality(df):
    df = df.copy()
    df["woy"] = df["week"].dt.isocalendar().week.astype(int)
    seasonal = df[df["in_stock"]].groupby("woy")["sales"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(seasonal["woy"], seasonal["sales"], color="#4878CF", linewidth=2, marker="o", markersize=3)
    ax.fill_between(seasonal["woy"], seasonal["sales"], alpha=0.15, color="#4878CF")
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Avg sales/SKU (in-stock)")
    ax.set_title("Seasonality", fontweight="bold")
    ax.set_xticks(range(1, 53, 4))
    save(fig, "03_seasonality")


def plot_censoring(df):
    df = df.copy()
    df["demand"] = df["sales"].where(df["in_stock"], other=np.nan)

    comp = df.groupby("sku_id").agg(avg_sales=("sales", "mean"), avg_demand=("demand", "mean")).dropna()
    comp["bias"] = (comp["avg_sales"] - comp["avg_demand"]) / comp["avg_demand"].clip(lower=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Censoring Impact", fontweight="bold")

    axes[0].scatter(comp["avg_demand"], comp["avg_sales"], alpha=0.4, s=15, color="#4878CF")
    lim = max(comp["avg_demand"].max(), comp["avg_sales"].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1)
    axes[0].set_xlabel("Avg demand (masked)")
    axes[0].set_ylabel("Avg sales (raw)")

    axes[1].hist(comp["bias"].clip(-0.5, 0.5), bins=40, color="#D65F5F", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Relative bias")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    save(fig, "04_censoring")


def plot_segmentation(df):
    df = df.copy()
    df["demand"] = df["sales"].where(df["in_stock"], other=np.nan)

    stats = df.groupby("sku_id")["demand"].agg(
        mean="mean", std="std", zero_rate=lambda x: (x == 0).mean()
    ).dropna()
    stats["cv"] = stats["std"] / stats["mean"].clip(lower=0.01)

    def label(row):
        if row["mean"] < 0.5: return "Very sparse"
        if row["zero_rate"] > 0.5: return "Intermittent"
        if row["cv"] > 1.0: return "Erratic"
        return "Smooth"

    stats["seg"] = stats.apply(label, axis=1)
    counts = stats["seg"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Demand Segmentation", fontweight="bold")

    colors = {"Smooth": "#6ACC65", "Intermittent": "#4878CF", "Erratic": "#D65F5F", "Very sparse": "#B8B8B8"}
    axes[0].pie(counts.values, labels=[f"{k}\n({v})" for k, v in counts.items()],
                colors=[colors[k] for k in counts.index], autopct="%1.0f%%", startangle=90)

    for seg, c in colors.items():
        m = stats["seg"] == seg
        axes[1].scatter(stats.loc[m, "mean"], stats.loc[m, "cv"], alpha=0.5, s=20, color=c, label=seg)
    axes[1].set_xlabel("Mean demand")
    axes[1].set_ylabel("CV")
    axes[1].set_xscale("log")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "05_segmentation")


def plot_inventory_state(inv):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Initial Inventory", fontweight="bold")

    for ax, col, title, c in zip(
        axes, ["inventory_on_hand", "in_transit_1", "in_transit_2"],
        ["On-hand", "Transit W+1", "Transit W+2"],
        ["#4878CF", "#6ACC65", "#D65F5F"],
    ):
        ax.hist(inv[col], bins=30, color=c, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Units")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.axvline(inv[col].mean(), color="red", linestyle="--", label=f"Mean {inv[col].mean():.1f}")
        ax.legend()

    fig.tight_layout()
    save(fig, "06_inventory_state")


def plot_sample_skus(df, n=6):
    df = df.copy()
    df["demand"] = df["sales"].where(df["in_stock"], other=np.nan)
    avg = df.groupby("sku_id")["demand"].mean()
    skus = avg[avg > 0].sample(n, random_state=42).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Sample SKU Histories", fontweight="bold")

    for ax, sku in zip(axes.flat, skus):
        s = df[df["sku_id"] == sku]
        ax.plot(s["week"], s["sales"], color="#B0B0B0", linewidth=0.8, label="Sales")
        ax.plot(s["week"], s["demand"], color="#4878CF", linewidth=1.2, label="Demand")
        oos = s[~s["in_stock"]]
        if not oos.empty:
            ax.scatter(oos["week"], oos["sales"], color="#D65F5F", s=12, zorder=5, label="OOS")
        ax.set_title(sku, fontsize=10)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "07_sample_skus")


def plot_cost_structure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cost Structure", fontweight="bold")

    sl = np.linspace(0.01, 0.99, 200)
    cost = 1.0 * (1 - sl) + 0.2 * sl
    opt = 1.0 / 1.2

    axes[0].plot(sl * 100, cost, color="#4878CF", linewidth=2)
    axes[0].axvline(opt * 100, color="#D65F5F", linestyle="--", label=f"Optimal = {opt:.1%}")
    axes[0].set_xlabel("Service level (%)")
    axes[0].set_ylabel("Expected cost")
    axes[0].legend()

    bias = np.linspace(-5, 15, 300)
    c = np.where(bias < 0, -bias, 0.2 * bias)
    axes[1].plot(bias, c, color="#4878CF", linewidth=2)
    axes[1].fill_between(bias[bias < 0], c[bias < 0], alpha=0.15, color="#D65F5F", label="Stockout (1.0€)")
    axes[1].fill_between(bias[bias >= 0], c[bias >= 0], alpha=0.15, color="#6ACC65", label="Holding (0.2€)")
    axes[1].set_xlabel("Order - demand (units)")
    axes[1].set_ylabel("Cost (€)")
    axes[1].legend()

    fig.tight_layout()
    save(fig, "08_cost_structure")


def main():
    print("Loading data...")
    df = load_training_data()
    inv = load_initial_state()

    print("Generating plots...")
    plot_dataset_overview(df)
    plot_portfolio_trends(df)
    plot_seasonality(df)
    plot_censoring(df)
    plot_segmentation(df)
    plot_inventory_state(inv)
    plot_sample_skus(df)
    plot_cost_structure()
    print(f"\nDone → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
