"""Unit tests verifying adapters match direct ebms_sim outputs (tests/test_adapters.py)."""

import pytest
import numpy as np
import pandas as pd

from ebms_sim.config import PRESETS, VehicleParams, BatteryParams, ThermalParams
from ebms_sim.trip import simulate_trip
from ebms_sim.profiles import generate_speed_profile, generate_gradient_profile
from app.adapters import (
    cached_simulate_trip,
    calculate_range_confidence_band,
    compare_flat_vs_hilly_routes,
    get_charging_matrix_comparison,
)


def test_adapter_trip_simulation_equals_direct_call_5_scenarios():
    """Verify adapter output exactly equals direct ebms_sim calls across 5 randomized parameter sets."""
    rng = np.random.default_rng(42)

    for i in range(5):
        # Randomize parameters
        pax = int(rng.integers(0, 6))
        luggage = float(rng.uniform(0.0, 50.0))
        payload_mass = pax * 65.0 + luggage
        temp = float(rng.uniform(10.0, 42.0))
        soc = float(rng.uniform(0.35, 0.95))
        soh = float(rng.uniform(0.80, 1.00))
        spd_cap = float(rng.uniform(18.0, 25.0))
        crr = float(rng.uniform(0.014, 0.022))

        preset_name = rng.choice(["lead_acid_48V_100Ah", "lfp_51V_60Ah"])
        preset = PRESETS[preset_name]

        v_params = VehicleParams(
            curb_mass_kg=preset["vehicle"].curb_mass_kg,
            driver_mass_kg=preset["vehicle"].driver_mass_kg,
            crr=crr,
            cda_m2=preset["vehicle"].cda_m2,
            eta_drive=preset["vehicle"].eta_drive,
            eta_regen=0.05,
            aux_w=preset["vehicle"].aux_w,
            max_speed_kmh=spd_cap,
        )
        b_params = preset["battery"]
        t_params = preset["thermal"]

        dur = 150
        v_profile = generate_speed_profile(duration_s=dur, speed_cap_kmh=spd_cap, seed=int(rng.integers(1, 100000)))
        g_profile = generate_gradient_profile(length_steps=dur, route_type="rolling", seed=int(rng.integers(1, 100000)))

        # 1. Direct call to ebms_sim.trip.simulate_trip
        direct_res = simulate_trip(
            vehicle_params=v_params,
            battery_params=b_params,
            thermal_params=t_params,
            speed_profile_ms=v_profile,
            gradient_profile=g_profile,
            payload_mass_kg=payload_mass,
            t_ambient_c=temp,
            headwind_ms=0.0,
            start_soc=soc,
            start_temp_c=temp,
            soh=soh,
        )

        # 2. Call via app.adapters.cached_simulate_trip
        adapter_df, adapter_summary = cached_simulate_trip(
            vehicle_params=v_params,
            battery_params=b_params,
            thermal_params=t_params,
            speed_profile_tuple=tuple(v_profile),
            gradient_profile_tuple=tuple(g_profile),
            payload_mass_kg=payload_mass,
            t_ambient_c=temp,
            headwind_ms=0.0,
            start_soc=soc,
            soh=soh,
        )

        # Assert exact agreement on all summary metrics
        assert pytest.approx(adapter_summary["wh_per_km"], abs=1e-4) == direct_res.summary.wh_per_km
        assert pytest.approx(adapter_summary["total_energy_wh"], abs=1e-4) == direct_res.summary.total_energy_wh
        assert pytest.approx(adapter_summary["distance_km"], abs=1e-4) == direct_res.summary.distance_km
        assert pytest.approx(adapter_summary["final_soc"], abs=1e-4) == direct_res.summary.final_soc
        assert pytest.approx(adapter_summary["range_remaining_km"], abs=1e-4) == direct_res.summary.range_remaining_km
        assert adapter_summary["cutoff_reached"] == direct_res.summary.cutoff_reached

        # Assert time-series values match exactly
        np.testing.assert_allclose(adapter_df["battery_power_w"].values, direct_res.time_series["battery_power_w"].values, rtol=1e-5)
        np.testing.assert_allclose(adapter_df["soc"].values, direct_res.time_series["soc"].values, rtol=1e-5)


def test_adapter_range_band():
    """Verify range confidence band scales properly with extreme temperatures and aging."""
    nom, min_r, max_r, exp = calculate_range_confidence_band(50.0, ambient_temp_c=25.0, soh=1.0)
    assert pytest.approx(nom, abs=1e-3) == 50.0
    # Standard +/- 10%
    assert pytest.approx(min_r, abs=1e-3) == 45.0
    assert pytest.approx(max_r, abs=1e-3) == 55.0

    # Cold + degraded condition: +/- 20%
    nom_cold, min_cold, max_cold, exp_cold = calculate_range_confidence_band(50.0, ambient_temp_c=5.0, soh=0.75)
    assert pytest.approx(min_cold, abs=1e-3) == 40.0
    assert pytest.approx(max_cold, abs=1e-3) == 60.0
    assert "Extreme thermal regime" in exp_cold
    assert "Degraded pack health" in exp_cold


def test_adapter_charging_matrix():
    """Verify charging matrix returns valid rows for 5C, 25C, and 40C."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    matrix_df = get_charging_matrix_comparison(
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        charging_params=preset["charging"],
    )
    assert len(matrix_df) == 3
    assert "Ambient Temp [°C]" in matrix_df.columns
    assert "Time to Full [hours]" in matrix_df.columns

