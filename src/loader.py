from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

SALES_FILE = "Week 0 - 2024-04-08 - Sales.csv"
INSTOCK_FILE = "Week 0 - In Stock.csv"
MASTER_FILE = "Week 0 - Master.csv"
INITIAL_STATE_FILE = "Week 0 - 2024-04-08 - Initial State.csv"
SUBMISSION_TEMPLATE_FILE = "Week 0 - Submission Template.csv"

WEEK_SALES_FILES = {
    1: "Week 1 - 2024-04-15 - Sales.csv",
    2: "Week 2 - 2024-04-22 - Sales.csv",
    3: "Week 3 - 2024-04-29 - Sales.csv",
    4: "Week 4 - 2024-05-06 - Sales.csv",
    5: "Week 5 - 2024-05-13 - Sales.csv",
    6: "Week 6 - 2024-05-20 - Sales.csv",
    7: "Week 7 - 2024-05-27 - Sales.csv",
    8: "Week 8 - 2024-06-03 - Sales.csv",
}


def _melt_wide(path, value_name):
    raw = pd.read_csv(path)
    raw["sku_id"] = raw["Store"].astype(str) + "_" + raw["Product"].astype(str)
    week_cols = [c for c in raw.columns if c not in ("Store", "Product", "sku_id")]
    long = raw.melt(id_vars="sku_id", value_vars=week_cols, var_name="week", value_name=value_name)
    long["week"] = pd.to_datetime(long["week"])
    return long[["sku_id", "week", value_name]]


def _load_master(path):
    df = pd.read_csv(path)
    df["sku_id"] = df["Store"].astype(str) + "_" + df["Product"].astype(str)
    return df[["sku_id", "Store", "Product", "ProductGroup", "Division",
               "Department", "DepartmentGroup", "StoreFormat", "Format"]]


def load_training_data(data_dir=DATA_DIR):
    sales = _melt_wide(data_dir / SALES_FILE, "sales")
    instock = _melt_wide(data_dir / INSTOCK_FILE, "in_stock")
    instock["in_stock"] = instock["in_stock"].fillna(True).astype(bool)
    instock = instock[instock["week"].isin(sales["week"].unique())]

    df = sales.merge(instock, on=["sku_id", "week"], how="left")
    df["in_stock"] = df["in_stock"].fillna(True)
    df = df.merge(_load_master(data_dir / MASTER_FILE), on="sku_id", how="left")
    return df.sort_values(["sku_id", "week"]).reset_index(drop=True)


def load_initial_state(data_dir=DATA_DIR):
    df = pd.read_csv(data_dir / INITIAL_STATE_FILE)
    df["sku_id"] = df["Store"].astype(str) + "_" + df["Product"].astype(str)
    return df[["sku_id", "End Inventory", "In Transit W+1", "In Transit W+2"]].rename(columns={
        "End Inventory": "inventory_on_hand",
        "In Transit W+1": "in_transit_1",
        "In Transit W+2": "in_transit_2",
    })


def load_competition_actuals(data_dir=DATA_DIR):
    frames = []
    for week_num, filename in WEEK_SALES_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw["sku_id"] = raw["Store"].astype(str) + "_" + raw["Product"].astype(str)
        week_col = [c for c in raw.columns if c not in ("Store", "Product", "sku_id")][-1]
        frames.append(pd.DataFrame({
            "sku_id": raw["sku_id"],
            "week": pd.to_datetime(week_col),
            "actual_sales": raw[week_col].fillna(0),
            "competition_round": week_num,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
