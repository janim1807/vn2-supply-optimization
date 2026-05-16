# VN2 Inventory Planning

Solution for the [VN2 Inventory Planning Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge) on DataSource.ai.

## Problem

599 SKUs, 157 weeks of history, 8 competition rounds. Place weekly orders with 2-week lead time. Asymmetric costs: stockout = 1.0€/unit, holding = 0.2€/unit.

## Approach

- **Forecasting**: LightGBM + CatBoost ensemble per horizon (h=1,2,3), optimized blend weights
- **Safety stock**: Locally-weighted conformal prediction with adaptive quantile (ACI)
- **Policy**: Newsvendor with inventory projection through lead time

## Results

| Method | Avg cost/SKU/week |
|--------|-------------------|
| Benchmark | 2.03€ |
| This solution (conformal+ACI) | **0.95€** |

## Usage

```bash
pip install -r requirements.txt
python pipeline.py
```

Data goes in `data/raw/` (download from DataSource.ai).

## Structure

```
src/
  loader.py      - data ingestion (wide→long)
  features.py    - lags, rolling stats, Fourier, scaling
  models.py      - LGB/CB training + blending
  policy.py      - newsvendor order computation
  conformal.py   - adaptive conformal inference
  evaluate.py    - cost metrics
pipeline.py      - main entry point
eda/             - exploratory plots
```
