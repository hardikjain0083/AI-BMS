"""Unit tests for trip simulation engine (ebms_sim.trip)."""

import pytest
import numpy as np
from ebms_sim.config import PRESETS, VehicleParams
from ebms_sim.trip import simulate_trip, estimate_remaining_range_km
from ebms_sim.profiles import generate_speed_profile, generate_gradient_profile


def test_simulate_trip_flat_20kmh_two_passengers():
    """CRITICAL TEST: Total Wh/km for a flat road, 2 passengers, 20 km/h, 25C is between 25 and 70 Wh/km."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    duration = 300  # 5 minutes
    v_target_ms = 20.0 / 3.6
    speed_profile = np.full(duration, v_target_ms)
    gradient_profile = np.zeros(duration)

    result = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=speed_profile,
        gradient_profile=gradient_profile,
        payload_mass_kg=2 * 65.0,  # 2 passengers
        t_ambient_c=25.0,
        headwind_ms=0.0,
        start_soc=0.90,
    )

    wh_km = result.summary.wh_per_km
    assert 25.0 <= wh_km <= 70.0, f"Expected Wh/km between 25 and 70, got {wh_km:.2f}"
    assert result.summary.distance_km > 1.0


def test_consumption_increases_with_passengers_grade_wind():
    """CRITICAL TEST: Consumption increases with passengers, with uphill grade, and with headwind."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    duration = 300
    v_profile = generate_speed_profile(duration_s=duration, speed_cap_kmh=20.0, seed=42)
    flat_grad = np.zeros(duration)
    uphill_grad = np.full(duration, 0.04)  # 4% uphill

    # Base: 0 passengers, flat, 0 headwind
    res_base = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=flat_grad,
        payload_mass_kg=0.0,
        t_ambient_c=25.0,
        headwind_ms=0.0,
    )

    # More passengers: 4 passengers (260 kg)
    res_passengers = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=flat_grad,
        payload_mass_kg=260.0,
        t_ambient_c=25.0,
        headwind_ms=0.0,
    )

    # Uphill: 4% slope
    res_uphill = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=uphill_grad,
        payload_mass_kg=0.0,
        t_ambient_c=25.0,
        headwind_ms=0.0,
    )

    # Headwind: 3 m/s
    res_wind = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=flat_grad,
        payload_mass_kg=0.0,
        t_ambient_c=25.0,
        headwind_ms=3.0,
    )

    assert res_passengers.summary.wh_per_km > res_base.summary.wh_per_km
    assert res_uphill.summary.wh_per_km > res_base.summary.wh_per_km
    assert res_wind.summary.wh_per_km > res_base.summary.wh_per_km


def test_downhill_zero_regen_current():
    """CRITICAL TEST: Downhill with eta_regen=0 gives zero battery current beyond aux load."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    v_params = VehicleParams(
        curb_mass_kg=350.0,
        driver_mass_kg=65.0,
        crr=0.018,
        cda_m2=0.85,
        eta_drive=0.82,
        eta_regen=0.0,  # Zero regen
        aux_w=50.0,
    )

    duration = 100
    v_profile = np.full(duration, 20.0 / 3.6)
    # Steep downhill (-7%)
    steep_downhill = np.full(duration, -0.07)

    res = simulate_trip(
        vehicle_params=v_params,
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=steep_downhill,
        payload_mass_kg=130.0,
    )

    df = res.time_series
    # Battery power must equal exactly aux_w (50W)
    np.testing.assert_allclose(df["battery_power_w"].values, 50.0, rtol=1e-3)
    # Current is simply aux_w / voltage (~1.0 A), zero traction discharge
    assert np.all(df["current_a"].values < 1.2)
    assert np.all(df["current_a"].values > 0.0)


def test_range_cutoff_and_monotonicity():
    """CRITICAL TEST: Range at SOC <= cutoff is 0; range increases monotonically with SOC."""
    cutoff = 0.10
    nom_cap = 100.0
    nom_volt = 48.0
    temp = 25.0
    wh_km = 40.0

    # Test SOC <= cutoff gives 0.0
    assert estimate_remaining_range_km(0.10, cutoff, nom_cap, nom_volt, temp, wh_km) == 0.0
    assert estimate_remaining_range_km(0.08, cutoff, nom_cap, nom_volt, temp, wh_km) == 0.0
    assert estimate_remaining_range_km(0.00, cutoff, nom_cap, nom_volt, temp, wh_km) == 0.0

    # Test monotonic increase
    soc_levels = [0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
    ranges = [
        estimate_remaining_range_km(s, cutoff, nom_cap, nom_volt, temp, wh_km)
        for s in soc_levels
    ]
    for i in range(len(ranges) - 1):
        assert ranges[i] < ranges[i + 1], f"Range must strictly increase: {ranges[i]} vs {ranges[i+1]}"


def test_trip_cutoff_detection():
    """Verify trip correctly flags cutoff_reached when battery drops to cutoff_soc."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    duration = 300
    v_profile = np.full(duration, 20.0 / 3.6)
    grad_profile = np.full(duration, 0.05)  # Uphill load

    # Start near cutoff (0.105) with cutoff at 0.10
    res = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=v_profile,
        gradient_profile=grad_profile,
        start_soc=0.105,
        cutoff_soc=0.10,
    )
    assert res.summary.cutoff_reached is True

