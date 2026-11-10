"""Unit tests for synthetic dataset generator (ebms_sim.datagen)."""

import pytest
import os
import numpy as np
import pandas as pd
from ebms_sim.datagen import (
    add_sensor_noise,
    generate_single_synthetic_trip,
    generate_dataset,
    save_tables,
)


def test_sensor_noise_bounds():
    """Verify sensor noise tolerances: voltage +-0.5%, temp +-1.0C, elevation +-3.0m."""
    rng = np.random.default_rng(42)
    v_true = np.full(500, 48.0)
    t_true = np.full(500, 30.0)
    z_true = np.full(500, 100.0)

    v_noisy, t_noisy, z_noisy = add_sensor_noise(v_true, t_true, z_true, rng)

    # Voltage noise: within +/- 0.5%
    v_diff_pct = np.abs((v_noisy - v_true) / v_true) * 100.0
    assert np.all(v_diff_pct <= 0.5001)

    # Temp noise: within +/- 1.0 C
    t_diff = np.abs(t_noisy - t_true)
    assert np.all(t_diff <= 1.0001)

    # Elevation noise: within +/- 3.0 m
    z_diff = np.abs(z_noisy - z_true)
    assert np.all(z_diff <= 3.0001)


def test_single_trip_generation_schema_and_bounds():
    """Verify parameter randomization ranges and schema of trip & window records."""
    trip_rec, win_recs = generate_single_synthetic_trip(
        trip_id=1,
        seed=123,
        trip_duration_s=120,  # 4 x 30s windows
        n_unique_routes=20,
    )

    # Check bounds
    assert 0 <= trip_rec["passengers"] <= 5
    assert 0.0 <= trip_rec["luggage_kg"] <= 80.0
    assert 0.012 <= trip_rec["crr"] <= 0.025
    assert 5.0 <= trip_rec["ambient_temp_c"] <= 45.0
    assert 0.70 <= trip_rec["soh"] <= 1.00
    assert 0.20 <= trip_rec["start_soc"] <= 1.00
    assert 15.0 <= trip_rec["speed_cap_kmh"] <= 25.0
    assert -3.0 <= trip_rec["headwind_ms"] <= 3.0
    assert 1 <= trip_rec["route_id"] <= 20

    # Targets must exist
    assert "wh_per_km" in trip_rec
    assert "range_remaining_km" in trip_rec
    assert trip_rec["wh_per_km"] > 0.0
    assert trip_rec["range_remaining_km"] >= 0.0

    # Windows check
    assert len(win_recs) == 4
    for w in win_recs:
        assert w["route_id"] == trip_rec["route_id"]
        assert "mean_speed_kmh" in w
        assert "mean_power_w" in w
        assert "noisy_mean_voltage_v" in w
        assert "noisy_mean_temp_c" in w
        assert "noisy_elevation_change_m" in w


def test_batch_generate_and_save(tmp_path):
    """Verify generating multiple trips and saving both tables."""
    n_trips = 4
    df_trips, df_windows = generate_dataset(
        n_trips=n_trips,
        trip_duration_s=60,
        seed=42,
        n_workers=1,  # In-process test
    )

    assert len(df_trips) == n_trips
    assert len(df_windows) == n_trips * 2  # 60s / 30s = 2 windows per trip
    assert "route_id" in df_trips.columns
    assert "route_id" in df_windows.columns

    # Test Parquet / CSV saving
    out_trips = str(tmp_path / "trips.parquet")
    saved_t, saved_w = save_tables(df_trips, df_windows, out_trips)

    assert os.path.exists(saved_t)
    assert os.path.exists(saved_w)

    # Read back and verify row counts
    if saved_t.endswith(".parquet"):
        read_t = pd.read_parquet(saved_t)
        read_w = pd.read_parquet(saved_w)
    else:
        read_t = pd.read_csv(saved_t)
        read_w = pd.read_csv(saved_w)

    assert len(read_t) == n_trips
    assert len(read_w) == n_trips * 2

