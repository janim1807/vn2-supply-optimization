# VN2 Inventory Planning Challenge: A Comprehensive Technical Research Report

> **Role**: Expert Data Scientist & Supply Chain Strategist
> **Date**: May 2026
> **Sources**: arXiv winner paper, DataSource.ai competition pages, Nicolas Vandeput's retrospective, participant blog posts

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Challenge Mechanics](#2-challenge-mechanics)
3. [The Winner's Pipeline: Two-Stage Predict-Then-Optimize](#3-the-winners-pipeline-two-stage-predict-then-optimize)
4. [Key Technical Innovations](#4-key-technical-innovations)
   - 4.1 Stockout-Aware Feature Engineering (Censored Demand)
   - 4.2 Global vs. Local Models
   - 4.3 Cost-Aware Ordering Policies
5. [Model Comparison: Top 3 Approaches](#5-model-comparison-top-3-approaches)
6. [LightGBM, Nixtla & Ensembling in Top Solutions](#6-lightgbm-nixtla--ensembling-in-top-solutions)
7. [Lessons Learned](#7-lessons-learned)
8. [Practical Takeaways for Real-World Retail](#8-practical-takeaways-for-real-world-retail)
9. [Sources](#9-sources)

---

## 1. Executive Summary

VN2 (October–November 2025) was the **first global inventory planning competition** — a real-world simulation where participants made six consecutive weekly replenishment decisions for 599 grocery store-product combinations. The challenge was designed by supply chain expert Nicolas Vandeput to bridge the gap between forecasting research and operational inventory management.

**Key facts at a glance:**

| Metric | Value |
|---|---|
| Participants | 183 |
| Beat the simple baseline | 25 (≈ 14%) |
| Winner's cost improvement | **–13.2%** vs. benchmark |
| Prize pool | €18,000 |
| Winner | Bartosz Szabłowski (Schneider Electric) |
| Winner's approach | Global CatBoost + Newsvendor ordering policy |

The most striking finding: **86% of participants could not outperform a seasonal 13-week moving average**. This underscores how difficult inventory optimization is in practice — and how high the bar is set by even simple, well-tuned baselines.

---

## 2. Challenge Mechanics

### 2.1 Competition Setup

| Parameter | Detail |
|---|---|
| **Organizer** | Nicolas Vandeput (SupChains) |
| **Platform** | DataSource.ai |
| **Competition window** | September 28 – November 10, 2025 |
| **Products** | 599 store–product pairs (67 stores × 297 products) |
| **Historical data** | ~157 weeks (2021–2024) of weekly sales & stock availability |
| **Ordering rounds** | 6 consecutive weekly decisions (Rounds 3–8) |
| **Participants** | 183 |

### 2.2 Penalties: Holding vs. Stockout

The cost structure creates a deliberate **asymmetry** that drives every strategic decision:

| Cost Type | Symbol | Value | Meaning |
|---|---|---|---|
| Holding cost | `ch` | **0.2 € / unit / week** | Cost of unsold inventory left at end of week |
| Shortage cost | `cs` | **1.0 € / unit** | Cost of each unit of lost sales (no backorders) |
| **Cost ratio** | `cs / ch` | **5×** | Stockouts are 5× more expensive than holding excess stock |

**Total weekly cost** = `cs × lost_sales + ch × ending_inventory`

The 5:1 cost ratio means the optimal strategy is to **bias toward over-ordering** — the theoretically correct service level is 83.3% (derived via the newsvendor critical fractile formula in Section 4.3).

### 2.3 Lead Time & Order Timing

Orders follow a **2-week delivery lead time**:

```
Week t:   Place order Q(t)
Week t+1: Order in transit
Week t+2: Order in transit
Week t+3: Order arrives, available for sale
```

This means every ordering decision must forecast demand **3 weeks ahead** and account for inventory already in transit from weeks t−2 and t−1.

### 2.4 The Benchmark

The official benchmark (Nicolas Vandeput, notebook #32) used:
- **Forecast**: 13-week seasonal moving average
- **Policy**: Order to achieve 4-week forward coverage
- **Cost achieved**: 4.334 € / period (average across 599 SKUs)

This deceptively simple baseline defeated 86% of all participants.

---

## 3. The Winner's Pipeline: Two-Stage Predict-Then-Optimize

Bartosz Szabłowski's winning solution (described in detail in his arXiv paper, *"One Global Model, Many Behaviors"*) is a modular **two-stage predict-then-optimize** pipeline. The two stages are deliberately decoupled — enabling independent validation of each component.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FORECASTING                                           │
│                                                                 │
│  Historical sales (censored-aware) ──► Global CatBoost model   │
│                                         (3 direct sub-models)  │
│                                              │                  │
│                                    D̂(t+1), D̂(t+2), D̂(t+3)     │
└──────────────────────────────────────────────┬──────────────────┘
                                               │
┌──────────────────────────────────────────────▼──────────────────┐
│  STAGE 2: ORDERING POLICY                                       │
│                                                                 │
│  Current inventory I(t) ──► Project forward ──► Ĩ(t+3)         │
│  Forecasts D̂(t+1..t+3) ──► Newsvendor formula ──► B(t+3)       │
│                                                                 │
│  Order: Q(t) = max(B(t+3) − Ĩ(t+3), 0)                        │
└─────────────────────────────────────────────────────────────────┘
```

**Result**: 3.763 € / period — a **13.2% improvement** over the benchmark.

### 3.1 Stage 1: The Global CatBoost Forecasting Model

**Architecture choices:**

| Choice | Detail | Rationale |
|---|---|---|
| Base learner | CatBoost (GBDT) | Native categorical support for store_id, product_id |
| Training scope | Global — all 599 series jointly | Cross-series learning, avoids per-SKU tuning |
| Forecast horizon | Direct multi-horizon: h = 1, 2, 3 | Separate sub-models prevent error accumulation |
| Training loss | RMSE on scaled targets | Robust to scale heterogeneity |
| Hyperparameter tuning | Optuna (100 trials, TPE sampler) | Minimizes validation MAE in original units |
| Early stopping | 500 rounds | Prevents overfitting |
| Validation | Chronological 10% holdout | Respects temporal order |

**Feature importance (approximate):**

| Feature Group | Importance Range |
|---|---|
| Week-of-year (seasonality) | 22–36% |
| Fourier seasonality terms | 4–9% |
| Store / Product identifiers | 2–8% |
| Seasonal lags & momentum | Moderate |
| Intermittency indicators | Moderate |

### 3.2 Stage 2: Cost-Aware Ordering Policy

The ordering policy is grounded in the **Newsvendor model** — the analytically optimal solution for single-period stochastic demand with asymmetric underage/overage costs.

**Step 1: Compute the critical fractile**

```
q* = cs / (cs + ch) = 1.0 / (1.0 + 0.2) = 0.833
```

This means: order enough to satisfy demand with probability **83.3%**. The corresponding z-score:

```
z_q = Φ⁻¹(0.833) ≈ 0.9674
```

**Step 2: Estimate demand uncertainty**

A Poisson approximation (common for retail count data):

```
σ(t+3) = φ × √D̂(t+3)
```

where `φ` is a dispersion scalar calibrated on the validation set.

**Step 3: Set target stock level**

```
B(t+3) = D̂(t+3) + z_q × σ(t+3)
```

**Step 4: Project current inventory forward through lead time**

```
Ĩ(t+3) = simulate inventory through weeks t+1, t+2 using forecasts D̂(t+1), D̂(t+2)
         accounting for already-in-transit orders from weeks t-1, t-2
```

**Step 5: Place order**

```
Q(t) = max(B(t+3) − Ĩ(t+3), 0)
```

> **Key insight**: Pure quantile regression (trying to learn the optimal order quantity end-to-end) ranked as "C-tier" despite its theoretical appeal. The analytical newsvendor formula — combined with a Poisson uncertainty proxy — proved simpler, more interpretable, and more effective.

---

## 4. Key Technical Innovations

### 4.1 Stockout-Aware Feature Engineering (Censored Demand)

**The problem: demand censoring**

When a product is out of stock, observed sales = 0. This zero does *not* represent zero demand — it represents unobservable demand. Using these zeros as training labels introduces systematic **downward bias**, especially for fast-moving items with frequent stockouts.

In the VN2 dataset, approximately **43% of weekly observations are zero sales**, combining genuine zero-demand periods with stockout-censored observations. Distinguishing between them is critical.

**The three-pronged solution:**

| Technique | How It Works | What It Fixes |
|---|---|---|
| **Censoring mask** | Mark `sales = NaN` when `in_stock = False` (using the stock availability flag in the dataset) | Prevents stockout zeros from biasing lag features, rolling means, and training labels |
| **Per-series dynamic scaling** | Divide each observation by the 53-week rolling mean (clipped ≥ 1) before training | Forces the global model to learn demand *patterns* rather than absolute volumes; handles scale from 0.3 to 99 units/week |
| **Time-decayed observation weights** | Recent year: weight 1.0 → prior year: 0.5 → older: 0.25 | Adapts to demand regime shifts (promotions ending, category changes) without discarding historical context |

**Why this matters:** Without censoring awareness, a product that stockouts every other week appears to have much lower demand than it truly does. The model under-forecasts, the ordering policy under-orders, and the cycle repeats — a self-reinforcing bias.

### 4.2 Global vs. Local Models

One of the most debated questions in time series forecasting is whether to train *one model per series* (local) or *one model for all series* (global). VN2 settled it decisively.

| Dimension | Global Model | Local Model |
|---|---|---|
| **Definition** | One estimator trained on all 599 series simultaneously | One separate estimator per SKU (599 models) |
| **Cross-series learning** | ✅ Learns shared patterns (seasonality, intermittency) across products and stores | ❌ Each model is isolated to its ~157 data points |
| **Scale heterogeneity** | Handled via per-series normalization (feature engineering) | Handled implicitly — each model sees only its own scale |
| **Hyperparameter tuning** | One search across the entire portfolio | 599 separate searches (or a single global setting applied blindly) |
| **Sparse/intermittent series** | ✅ Borrows strength from similar, higher-volume series | ❌ High-variance estimates from sparse data |
| **Maintenance overhead** | One model to retrain and monitor | Hundreds of models to manage |
| **VN2 outcome** | **1st place** | Did not reach top 5 |

**Why global models win on intermittent data:** With 43% zero-sales observations, a local model for a sparse product has very few non-zero training examples. A global model trained on hundreds of products — many of which share the same store, category, or seasonality pattern — provides far more signal for learning the underlying demand process.

### 4.3 Cost-Aware Ordering Policies

Forecasting accuracy (MAE, RMSE, MAPE) is a means, not an end. What matters is the **cost incurred by the ordering decisions** derived from those forecasts. This requires translating forecasts into orders in a way that respects the asymmetric cost structure.

**Comparison of ordering policies tried in VN2:**

| Policy | Description | VN2 Tier | Why It Works / Fails |
|---|---|---|---|
| **Coverage policy** (benchmark) | Order to achieve N-week forward coverage | B-tier baseline | Simple but ignores cost ratio — treats all SKUs identically |
| **Newsvendor critical fractile** | Order the q*-quantile of demand, q* = cs/(cs+ch) | **A-tier (won)** | Analytically optimal; directly accounts for cost asymmetry |
| **Quantile regression** | Learn the q*-quantile directly from data | C-tier | Theoretically sound but suffers from distribution shift and data sparsity |
| **Direct RL policy** | Neural network trained to minimize cost via reinforcement | A-tier (2nd) | Optimal in theory; impractical training complexity |
| **Fixed safety stock** | Order mean + k×σ with fixed k | C-tier | Doesn't adapt k to cost structure |
| **ABC/XYZ + DDMRP** | Segment-based replenishment rules | D-tier | No forecasting capability; breaks on demand variability |

**The newsvendor insight:**

Given cs = 1.0 and ch = 0.2, the cost-optimal target service level is:

```
q* = 1.0 / (1.0 + 0.2) = 83.3%
```

This means: accept a 16.7% stockout risk to avoid over-investing in holding costs. The mathematics penalize both extremes symmetrically from this optimum — over-ordering wastes money on holding, under-ordering wastes more money on lost sales.

---

## 5. Model Comparison: Top 3 Approaches

| | **1st: Bartosz Szabłowski** | **2nd: Matias Alvo** | **3rd: Philip Stubbs & Jakub Figura** |
|---|---|---|---|
| **Forecast model** | CatBoost (global GBDT) | LightGBM (demand quantiles) | Refined seasonal moving average + ensemble |
| **Decision model** | Newsvendor formula (analytical) | Deep Reinforcement Learning neural policy | Optimized coverage-based policy |
| **Paradigm** | Predict-then-optimize | End-to-end learned policy | Statistical + simulation-based optimization |
| **Censored demand handling** | NaN masking + per-series scaling + time-decay weights | LightGBM features incorporating stock availability flags | Adjusted historical averages |
| **Global vs. local** | Global (all 599 series jointly) | Local quantile models per series | Hybrid |
| **Key technical innovation** | Stockout-aware feature engineering + Poisson uncertainty proxy for safety stock | RL policy directly optimized against cost signal, bypassing forecast-to-order translation | Rigorous simulation framework enabling systematic policy parameter search |
| **Complexity** | Medium | High | Low–Medium |
| **Practical adoptability** | ✅ High — straightforward to productionize | ⚠️ Niche — requires RL infrastructure and long training time | ✅ High — no ML dependencies required |
| **Cost vs. benchmark** | **–13.2%** (3.763 € vs. 4.334 €) | ~–12% | ~–8% (estimated) |
| **Published write-up** | arXiv paper (2601.18919) | LinkedIn / conference presentation | LinkedIn post |

**What separated 1st from 2nd:**
Matias Alvo's RL approach is theoretically more general — it can learn a policy that accounts for multi-period effects that the single-period newsvendor formula ignores. However, the LightGBM demand model feeding the RL policy introduces forecasting error that propagates through the RL agent. Szabłowski's cleaner modular design (strong forecast + principled policy) proved more robust across all 6 rounds.

**What separated 2nd from 3rd:**
Stubbs & Figura deliberately anchored to the benchmark's structure and improved it via rigorous simulation and parameter optimization. Their approach is the most operationally practical — it requires no machine learning expertise — but it hits a ceiling because the underlying seasonal moving average forecast is not adaptive.

---

## 6. LightGBM, Nixtla & Ensembling in Top Solutions

### LightGBM

LightGBM was the **most widely used ML model** among competitive participants, though it was ultimately beaten by CatBoost in the winning solution.

**Where it appeared:**
- **Matias Alvo (2nd place)**: LightGBM used to predict demand quantiles, which were fed as state features into the RL policy network
- **Standalone forecaster (multiple participants)**: Direct multi-horizon forecasting with engineered lag features, rolling statistics, calendar interactions, and stock availability flags
- **Ensemble component**: `AutoMFLES (Nixtla) + LightGBM` weighted blend, with blend weights optimized by minimizing validation cost using `scipy.optimize`

**Why CatBoost edged LightGBM for the win:**
CatBoost handles high-cardinality categorical features (store_id, product_id, product_category) natively without one-hot encoding or target encoding hacks. In a dataset with 67 stores and 297 products as identifiers, this likely provided a meaningful representational advantage.

### Nixtla (StatsForecast / NeuralForecast)

Nixtla was the **official webinar partner** for VN2, providing participant tutorials and baseline notebooks.

| Role | Detail |
|---|---|
| **Official webinar** | Nixtla ran a live tutorial on using StatsForecast + NeuralForecast for VN2 participants |
| **Demand type classification** | Official notebook (#44) demonstrated using StatsForecast to classify each SKU as intermittent, smooth, or lumpy — guiding model selection |
| **AutoMFLES** | Nixtla's automatic MFLES model appeared as an ensemble component in several solutions |
| **NeuralForecast models** | Used as baseline components in community notebooks |
| **Top 3 presence** | Not directly in the winning solutions, but present as ensemble ingredients |

Nixtla's primary value in VN2 was **demand segmentation** (identifying which SKUs need special intermittency handling) and **rapid experimentation** with many statistical models via `StatsForecast`'s unified API.

### Ensembling Strategies

Nicolas Vandeput's retrospective analysis noted: *"Ensembling consistently delivered gains in the top solutions."*

**Common ensemble patterns observed:**

```
┌──────────────────┐    ┌──────────────────┐
│ Statistical model │    │ ML model         │
│ (AutoMFLES,      │    │ (LightGBM,       │
│  Seasonal MA)    │    │  CatBoost)       │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └──────────┬────────────┘
                    │
              ┌─────▼──────────────────────────┐
              │ Blend: w₁·f₁ + w₂·f₂          │
              │ Weights optimized via           │
              │ scipy.optimize on cost metric   │
              └────────────────────────────────┘
```

**Key ensembling principles from VN2:**
1. **Optimize blend weights on cost, not RMSE** — the cost metric is the true objective
2. **Diversify model types** — statistical models handle structural trends; ML models handle cross-series patterns
3. **Keep it simple** — two-model blends outperformed complex stacking schemes in most cases
4. **Use chronological validation** — shuffled cross-validation is invalid for time series; always validate on a future holdout period

### What Did NOT Work (C-D Tier)

Despite widespread adoption outside competitions, these approaches consistently underperformed:

| Approach | Why It Failed |
|---|---|
| ABC/XYZ segmentation | No forecasting capability — just routes products to fixed rules |
| DDMRP | Designed for manufacturing/MRP contexts; ineffective for demand-driven grocery retail |
| Outlier detection pipelines | Too aggressive — removed legitimate demand signals along with noise |
| Pure quantile regression | Distribution shift between training and test; sparse data for high quantiles |
| ARIMA / SES / Croston (standalone) | Insufficient to beat ML-enhanced seasonal MA on this dataset |

---

## 7. Lessons Learned

### The Meta-Lesson: Most Solutions Failed the Basics

Only 14% of participants beat a 13-week seasonal moving average. Post-competition analysis identified three root causes:

1. **Censoring blindness**: Most participants treated stockout zeros as genuine zero demand, corrupting their training data
2. **Wrong validation metric**: Teams optimized MAE/RMSE but competed on cost — a model can have higher RMSE and lower cost if it's biased in the right direction
3. **Ignoring the cost structure**: Using a symmetric 50th-percentile forecast when the problem demands an 83rd-percentile service level is a fundamental mismatch

### Forecast Accuracy ≠ Inventory Performance

This is the most important practical lesson from VN2. A model with lower MAE can perform *worse* on cost if its errors are systematically in the wrong direction (e.g., always under-forecasting). The winning approach explicitly calibrated forecasts using the cost ratio — not cross-validated RMSE.

**Implication:** In production inventory systems, forecast evaluation should always include cost simulation on a holdout period, not just accuracy metrics.

### The Newsvendor Formula Is Underused

Despite being decades-old operations research, the newsvendor critical fractile formula was the single most impactful technique in VN2. Most industrial inventory systems use heuristic safety stock formulas (e.g., "k weeks of coverage") that ignore the actual cost structure. VN2 demonstrated that even a modest ML forecast + correct newsvendor policy beats an excellent forecast + naive policy.

### Reinforcement Learning: Promising But Premature

The 2nd-place RL solution proved that end-to-end policy learning is competitive — but the training complexity (GPU infrastructure, hyperparameter sensitivity, long convergence time) makes it impractical for most retail operations teams. The predict-then-optimize paradigm remains the practical gold standard.

---

## 8. Practical Takeaways for Real-World Retail

Based on VN2 findings, here are eight high-leverage improvements for production inventory systems:

| # | Takeaway | Implementation |
|---|---|---|
| 1 | **Treat stockout periods as missing data, not zeros** | Add stock availability flag to your data pipeline; mask sales = NaN when stock = 0 before computing any lag features or training ML models |
| 2 | **Train one global model across all SKUs** | Use LightGBM or CatBoost with SKU/store as categorical features; normalize by rolling mean before training |
| 3 | **Use the newsvendor formula for order quantities** | Compute q* = cs / (cs + ch) from your actual cost parameters; use Φ⁻¹(q*) as your safety stock multiplier |
| 4 | **Project inventory through the lead time** | Simulate inventory forward LT weeks using near-term forecasts before computing the order quantity |
| 5 | **Time-weight your training data** | Apply exponential decay: recent year = 1.0, prior year = 0.5, older = 0.25 |
| 6 | **Use direct multi-horizon forecasting** | Train separate models for h=1, h=2, h=3 rather than recursive/ARIMA-style; prevents error accumulation |
| 7 | **Validate on cost, not RMSE** | Build a simulation harness that evaluates any forecast on your actual cost function; this is your true model selection criterion |
| 8 | **Benchmark seriously before adding complexity** | A well-tuned seasonal moving average + newsvendor policy is a formidable baseline; measure improvement over it rigorously before deploying ML |

---

## 9. Sources

| Source | URL |
|---|---|
| Winner arXiv paper: *One Global Model, Many Behaviors* | [arxiv.org/abs/2601.18919](https://arxiv.org/abs/2601.18919) |
| Winner arXiv paper (HTML full text) | [arxiv.org/html/2601.18919](https://arxiv.org/html/2601.18919) |
| DataSource.ai — VN2 competition page | [datasource.ai/.../vn2-inventory-planning-challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge/description) |
| Nicolas Vandeput, Medium — *My Learning Points from VN2* | [nicolas-vandeput.medium.com/.../vn2...](https://nicolas-vandeput.medium.com/my-learning-points-from-vn2-the-first-inventory-competition-a4bffcc92856) |
| Philippe Dagher, Medium — *TimesFM 2.5 Field Report* | [medium.com/dataai/.../timesfm-2-5...](https://medium.com/dataai/forecasting-what-matters-a-field-report-from-the-vn2-inventory-challenge-with-timesfm-2-5-4743e652e48d) |
| Nixtla VN2 webinar announcement | [x.com/nixtlainc/...](https://x.com/nixtlainc/status/1973793186360279166) |
| Nixtla YouTube — *Forecasting models for VN2* | [youtube.com/watch?v=0kGr8twWjag](https://www.youtube.com/watch?v=0kGr8twWjag) |
| YouTube — *VN2 Winners Explain Their Solutions* | [youtube.com/watch?v=pypzcvwmApA](https://www.youtube.com/watch?v=pypzcvwmApA) |
| Bartosz Szabłowski LinkedIn post | [linkedin.com/posts/bartosz-szablowski...](https://www.linkedin.com/posts/bartosz-szablowski_a-few-days-ago-i-won-the-vn2-challenge-activity-7396079061543981056-vTQ_) |

---

*Report compiled from primary sources: competition platform, winner's arXiv paper, and post-competition retrospectives. All cost figures are from official VN2 documentation.*
