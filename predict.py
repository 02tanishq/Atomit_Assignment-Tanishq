"""Production prediction pipeline for Cloud Workload Forecasting Challenge.

Reproduces predictions.csv from raw or smoke-test telemetry data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def resolve_data_file(data_path: Path) -> Path:
    """Find readings CSV or CSV.gz file within a directory or validate file path."""
    if data_path.is_file():
        return data_path

    if data_path.is_dir():
        candidates = [
            data_path / "vm_cpu_readings.csv",
            data_path / "vm_cpu_readings.csv.gz",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Search for any readings file
        matches = list(data_path.glob("*readings*.csv*")) + list(data_path.glob("*.csv*"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not locate readings dataset in: {data_path}")


def load_and_aggregate(file_path: Path) -> pd.DataFrame:
    """Load raw VM telemetry and aggregate into fleet-level time series."""
    df = pd.read_csv(file_path)

    # Disambiguate columns flexibly
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    # Timestamp column
    time_col = next((col_map[c] for c in ["timestamp", "time", "datetime", "ts"] if c in cols), df.columns[0])

    # VM identifier column
    vm_col = next((col_map[c] for c in ["vm_id", "vmid", "vm", "id"] if c in cols), None)

    # CPU column: prefer avg_cpu over min_cpu/max_cpu
    cpu_col = None
    for target in ["avg_cpu", "aggregate_cpu_utilization", "cpu_utilization", "cpu", "avg"]:
        match = next((col_map[c] for c in cols if target in c), None)
        if match:
            cpu_col = match
            break
    if cpu_col is None:
        cpu_col = df.columns[-1]

    # Groupby timestamp
    if vm_col:
        fleet = df.groupby(time_col, as_index=False).agg(
            active_vm_count=(vm_col, "nunique"),
            aggregate_cpu_utilization=(cpu_col, "mean"),
        )
    else:
        fleet = df.groupby(time_col, as_index=False).agg(
            aggregate_cpu_utilization=(cpu_col, "mean"),
        )
        fleet["active_vm_count"] = 10000

    fleet = fleet.rename(columns={time_col: "timestamp"})
    fleet = fleet.sort_values("timestamp").reset_index(drop=True)
    return fleet


def generate_forecast(fleet: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Fit champion model strictly on training data and generate multi-step holdout forecast."""
    n_samples = len(fleet)

    if n_samples > horizon:
        train = fleet.iloc[:-horizon].copy()
        holdout = fleet.iloc[-horizon:].copy()
        holdout_timestamps = holdout["timestamp"].values
    else:
        train = fleet.copy()
        diffs = fleet["timestamp"].diff().dropna()
        step = int(diffs.median()) if len(diffs) > 0 else 300
        last_t = fleet["timestamp"].iloc[-1]
        holdout_timestamps = np.array([last_t + step * (i + 1) for i in range(horizon)])

    y_train = train["aggregate_cpu_utilization"].astype(float).values
    latest_vm_count = int(round(train["active_vm_count"].iloc[-1]))

    # Forecasting aggregate_cpu_utilization via EDA-Informed Decomposition
    try:
        from statsmodels.tsa.seasonal import STL

        period = 12 if len(y_train) >= 24 else max(2, len(y_train) // 2)
        seasonal_len = period + 1 if (period + 1) % 2 != 0 else period + 2
        stl = STL(y_train, period=period, seasonal=seasonal_len, robust=True).fit()

        # 1. Seasonal forecast
        s_fc = np.tile(stl.seasonal[-period:], int(np.ceil(horizon / period)))[:horizon]

        # 2. Damped trend forecast towards historical mean
        mean_y = float(np.mean(y_train))
        last_trend = float(stl.trend[-1])
        t_fc = np.array([last_trend * (1.0 - 0.035 * h) + mean_y * (0.035 * h) for h in range(1, horizon + 1)])

        # 3. Autoregressive residual forecast
        residuals = y_train - (stl.trend + stl.seasonal)
        if len(residuals) > 2:
            phi = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            phi = np.clip(phi, -0.9, 0.9)
        else:
            phi = 0.0
        r_fc = np.array([float(residuals[-1]) * (phi ** h) for h in range(1, horizon + 1)])

        cpu_preds = t_fc + s_fc + r_fc
    except Exception:
        # Resilient fallback: Seasonal-naive + mean-reversion drift
        period = 12 if len(y_train) >= 12 else len(y_train)
        s_fc = np.tile(y_train[-period:], int(np.ceil(horizon / period)))[:horizon]
        cpu_preds = s_fc * 0.7 + np.mean(y_train) * 0.3

    # Bounds & formatting
    cpu_preds = np.clip(cpu_preds, 0.0, 100.0)
    cpu_preds = np.round(cpu_preds, 4)

    # Active VM count is strictly invariant or persists the latest operational count
    pred_vm_count = np.full(horizon, latest_vm_count, dtype=int)

    out = pd.DataFrame({
        "timestamp": holdout_timestamps,
        "active_vm_count": pred_vm_count,
        "aggregate_cpu_utilization": cpu_preds,
    })
    return out


def validate_contract(df: pd.DataFrame, horizon: int = 24) -> None:
    """Validate submission contract compliance."""
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columns mismatch: expected {expected_cols}, got {list(df.columns)}")

    if len(df) != horizon:
        raise ValueError(f"Expected {horizon} rows, got {len(df)}")

    if df.isna().any().any():
        raise ValueError("Predictions contain NaN values")

    if not np.isfinite(df["aggregate_cpu_utilization"]).all():
        raise ValueError("Aggregate CPU utilization contains non-finite values")

    if (df["active_vm_count"] < 0).any():
        raise ValueError("Active VM count cannot be negative")

    if ((df["aggregate_cpu_utilization"] < 0) | (df["aggregate_cpu_utilization"] > 100)).any():
        raise ValueError("Aggregate CPU utilization must be within [0, 100]")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected in output")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically increasing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Workload Forecasting Challenge Prediction CLI")
    parser.add_argument("--data", type=str, default="Dataset", help="Path to raw dataset directory or file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output predictions CSV")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24 steps)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.output)

    print(f"Loading telemetry data from: {data_path}...")
    file_path = resolve_data_file(data_path)
    fleet = load_and_aggregate(file_path)
    print(f"Aggregated {len(fleet)} total timestamps.")

    print(f"Generating {args.horizon}-step holdout forecast...")
    preds = generate_forecast(fleet, horizon=args.horizon)

    validate_contract(preds, horizon=args.horizon)

    preds.to_csv(out_path, index=False)
    print(f"Successfully generated {out_path} ({len(preds)} rows) matching submission contract.")


if __name__ == "__main__":
    main()
