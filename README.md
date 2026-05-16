# VN2 Inventory Planning

Solution for the [VN2 Inventory Planning Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge) on DataSource.ai.

## Problem

599 SKUs, 157 weeks of history, 8 competition rounds. Place weekly orders with 2-week lead time. Asymmetric costs: stockout = 1.0€/unit, holding = 0.2€/unit (5:1 ratio).

## Results

| Method | Avg cost/SKU/week |
|--------|-------------------|
| Benchmark | 2.03€ |
| This solution | **0.95€** |

## Approach

### Forecasting

LightGBM + CatBoost ensemble, trained per horizon (h=1, h=2, h=3). Blend weights optimized on validation set — CatBoost typically gets 80-95% weight.

**Features:**
- Lag features: 1, 2, 3, 4, 8, 13, 52 weeks
- Rolling means: 4, 8, 13 week windows
- Fourier seasonality (sin/cos, 2 harmonics)
- Intermittency rate (zero-demand frequency over 13 weeks)
- Categoricals: Store, Product, ProductGroup, Department, StoreFormat, Format
- Per-series scaling (53-week rolling mean) and time-decay sample weights

Out-of-stock periods are masked (demand = NaN) so censored zeros don't bias the model downward.

### Safety Stock (Conformal Prediction + ACI)

Instead of assuming a parametric distribution, we use the model's own historical errors to set safety margins per SKU:

1. Compute relative residuals: `(actual - predicted) / max(predicted, 1)`
2. Weight recent residuals higher (exponential decay, halflife = 26 weeks)
3. Take the weighted quantile at level α to get a margin ratio
4. `safety_margin = forecast_h3 × margin_ratio`

α adapts each round via Adaptive Conformal Inference:
- Start at α = 0.65
- If realized coverage < optimal (too many stockouts) → raise α → bigger margins
- If coverage is fine → lower α → reduce holding costs

### Order Policy

```
I(t+1) = max(on_hand + transit_1 - forecast_h1, 0)
I(t+2) = max(I(t+1) + transit_2 - forecast_h2, 0)

target = forecast_h3 + safety_margin
order  = max(target - I(t+2), 0)
```

Projects inventory forward through the lead time using h=1 and h=2 forecasts, then orders enough to hit the target stock level when the order arrives at t+3.

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
