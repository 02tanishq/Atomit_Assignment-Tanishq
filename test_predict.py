import numpy as np
import pandas as pd
from cwfc.predict import generate_forecast, validate_contract


def test_predict_pipeline_contract():
    # Synthetic fleet data with 40 timestamps
    timestamps = np.arange(0, 40 * 300, 300)
    fleet = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count": np.full(40, 500),
        "aggregate_cpu_utilization": np.random.uniform(8.0, 12.0, size=40)
    })
    
    forecast = generate_forecast(fleet, horizon=24)
    validate_contract(forecast, horizon=24)
    
    assert len(forecast) == 24
    assert list(forecast.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert (forecast["active_vm_count"] == 500).all()
    assert (forecast["aggregate_cpu_utilization"] >= 0).all()
    assert (forecast["aggregate_cpu_utilization"] <= 100).all()
    assert forecast["timestamp"].is_monotonic_increasing
