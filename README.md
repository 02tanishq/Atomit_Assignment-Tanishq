# Cloud Workload Forecasting Challenge

A production-ready solution for the [Atomity Cloud Workload Forecasting Challenge](https://github.com/atomityhq/forecasting-challenge): forecast fleet-level `active_vm_count` and `aggregate_cpu_utilization` 24 steps (2 hours) ahead at 5-minute sampling cadence.

---

## Multi-Metric Benchmark Results (24-Step Holdout Horizon)

| Model Architecture | MAE (pp) | RMSE (pp) | sMAPE (%) | Normalized MAE | Relative Gain (%) | Peak MAE (Top 20%) | Max Abs Error | Mean Bias Error |
|---|---|---|---|---|---|---|---|---|
| **Champion: EDA-Informed STL Decomposition** | **0.0561** | **0.0720** | **0.60%** | **0.3819** | **+61.81%** | **0.0820** | **0.1700** | **−0.0091** |
| Linear ML: Ridge Regression (6 feats) | 0.1035 | 0.1325 | 1.12% | 0.7053 | +29.47% | 0.1957 | 0.3565 | −0.0078 |
| ML Hybrid: 85% XGB-6 + 15% SNaive-1h | 0.1153 | 0.1527 | 1.24% | 0.7857 | +21.43% | 0.2480 | 0.3707 | −0.0375 |
| ML: XGBoost (6 feats) + Drift | 0.1178 | 0.1559 | 1.27% | 0.8027 | +19.73% | 0.2495 | 0.4020 | −0.0348 |
| ML: Direct Multi-Horizon GBR | 0.1235 | 0.1544 | 1.32% | 0.8413 | +15.87% | 0.1771 | 0.2840 | +0.0820 |
| ML: XGBoost (9 feats) + Drift | 0.1308 | 0.1616 | 1.41% | 0.8911 | +10.89% | 0.2699 | 0.3436 | −0.0602 |
| Baseline: Moving Avg (6 steps / 30 mins) | 0.1336 | 0.1713 | 1.44% | 0.9099 | +9.01% | 0.2926 | 0.4451 | −0.0781 |
| Baseline: Seasonal-Naive 1h (12 steps) | 0.1400 | 0.1590 | 1.51% | 0.9541 | +4.59% | 0.2398 | 0.3114 | −0.0526 |
| Baseline: Seasonal-Naive 2h (24 steps) | 0.1464 | 0.1671 | 1.58% | 0.9974 | +0.26% | 0.2398 | 0.3114 | −0.0195 |
| Baseline: Persistence (official starter) | 0.1468 | 0.1848 | 1.59% | 1.0000 | 0.00% | 0.3189 | 0.4714 | −0.1044 |
| Classical: ARIMA(1,0,0) | 0.1519 | 0.1812 | 1.64% | 1.0351 | −3.51% | 0.0742 | 0.3416 | +0.1201 |

`active_vm_count`: Invariant at **10,000 VMs** across all measurements (zero VM churn) $\rightarrow$ Persistence achieves an **exact MAE = 0.0000**.

---

## Quick Start

### 1. Environment Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Dataset Setup
Ensure raw data is extracted in `Dataset/`:
```
Dataset/
├── vm_cpu_readings.csv.gz    # 1.34M measurement rows (timestamp, vm_id, min_cpu, max_cpu, avg_cpu)
└── vm_metadata.csv.gz        # VM metadata (vm_id, category, deployment_id, etc.)
```

### 3. Generate Submission Artifact
Reproduce `predictions.csv` directly using the production forecasting CLI:
```bash
python -m cwfc.predict --data Dataset --output predictions.csv
```
*Note: The CLI is fully dynamic and smoke-test ready. It supports directories or individual `.csv` / `.csv.gz` files, dynamically infers fleet size and timestamps without hard-coding, and enforces all contract requirements (24 rows, schema, bounds).*

Alternatively, you can run the test suite to verify contract validation:
```bash
pytest tests/
```

### 4. Interactive Notebooks
```bash
jupyter notebook notebooks/01_eda.ipynb       # Exploratory analysis (16 figures, 13 tables)
jupyter notebook notebooks/02_modeling.ipynb  # Feature engineering, XGBoost benchmark & actual plotting code
jupyter notebook notebooks/03_policy.ipynb    # Capacity planning policy & headroom curves
```

---

## Repository Structure

```
.
├── Dataset/                     # Raw traces (vm_cpu_readings.csv.gz, vm_metadata.csv.gz)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis (16 figures, 13 tables)
│   ├── 02_modeling.ipynb        # Feature engineering, XGBoost & multi-model evaluation, native plots
│   └── 03_policy.ipynb          # Headroom curves, burst risk detection, capacity policy
├── reports/
│   ├── final_report.md          # Comprehensive technical report
│   ├── figures/                 # Generated PNG figures (eda_*, mod_*, pol_*)
│   └── tables/                  # Generated CSV scorecards and summaries
├── src/cwfc/                    # Official candidate starter package
│   ├── baselines.py             # Persistence & Seasonal-Naive implementations
│   ├── metrics.py               # MAE, RMSE, sMAPE, Normalized MAE
│   └── validation.py            # Rolling-origin expanding-window split generators
├── predictions.csv              # 24-step submission artifact (dynamic range 9.03% to 9.47%)
└── README.md
```

---

## Architectural Highlights

- **Direct EDA Grounding**: Derives the champion forecaster directly from the 60-minute repeating seasonal cycle and natural mean-reversion identified in `01_eda.ipynb`.
- **Dynamic Swings**: Unlike standard autoregressive models that decay to a flat mean, the Champion Decomposition Model dynamically captures intra-hour drops (down to 9.03%), rebounds (9.35%), and late-horizon peaks (9.47%).
- **Multi-Metric Rigor**: Evaluated across 8 complementary metrics: MAE, RMSE, sMAPE, Normalized MAE, Relative Improvement, Peak Load MAE (Top 20%), Max Absolute Error, and Mean Bias Error.
- **5-Pillar Operational Policy**: Directly fulfills Section G recommendations by combining Safety Margins, Forecast Thresholds, Lead-Time Uncertainty Bands, Peak-Risk Velocity Triggers, and Workload-Segment Rules (partitioning Interactive vs Delay-Insensitive workloads from `vm_metadata.csv.gz`). Achieves a 0.0% SLA breach rate on the holdout, enables a safe **10.55 : 1** datacenter overcommitment ratio, and slashes physical server footprint by **90.5%**.
- **Pure Starter Utilities**: Employs strictly the provided `cwfc.baselines`, `cwfc.metrics`, and `cwfc.validation` utilities with no artificial internal modules.

## Auther
Tanishq Gupta
