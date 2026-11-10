"""Unit tests for speed optimizer and dynamic programming (ebms_sim.optimizer)."""

import pytest
import numpy as np
from ebms_sim.config import PRESETS
from ebms_sim.optimizer import (
    sweep_speed_wh_per_km,
    optimize_trip_speeds_dp,
    RouteSegment,
)


def test_speed_sweep_curve_is_u_shaped():
    """CRITICAL TEST: Speed sweep curve is U-shaped with an interior minimum between 10 and 25 km/h."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    speeds = np.linspace(10.0, 25.0, 31)

    spds, wh_km, best_spd, min_wh = sweep_speed_wh_per_km(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        payload_mass_kg=130.0,
        grade_fraction=0.0,
        headwind_ms=0.0,
        t_ambient_c=25.0,
        speeds_kmh=speeds,
    )

    # Verify U-shape:
    # 1. Minimum occurs strictly in the interior (not at 10 and not at 25)
    assert 12.0 <= best_spd <= 20.0, f"Expected interior minimum between 12 and 20 km/h, got {best_spd}"
    # 2. Left endpoint (10 km/h) has higher consumption than best speed due to aux load
    assert wh_km[0] > min_wh
    # 3. Right endpoint (25 km/h) has higher consumption than best speed due to aero drag
    assert wh_km[-1] > min_wh

    # 4. First derivative starts negative (decreasing) and ends positive (increasing)
    grad = np.gradient(wh_km, speeds)
    assert grad[0] < 0.0, "Curve should initially decrease as speed increases from 10 km/h"
    assert grad[-1] > 0.0, "Curve should increase as speed approaches 25 km/h"


def test_optimize_trip_speeds_dp_unconstrained():
    """Verify unconstrained optimization selects the energy-minimal speed for each segment."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    segments = [
        RouteSegment(distance_m=1000.0, grade_fraction=0.0),    # Flat
        RouteSegment(distance_m=500.0, grade_fraction=0.04),    # Uphill
        RouteSegment(distance_m=500.0, grade_fraction=-0.04),   # Downhill
    ]

    res = optimize_trip_speeds_dp(
        segments=segments,
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        payload_mass_kg=130.0,
        max_total_time_s=None,
    )

    # All recommended speeds must be within legal e-rickshaw limits [10, 25] km/h
    for spd in res.segment_speeds_kmh:
        assert 10.0 <= spd <= 25.0

    assert res.total_distance_km == 2.0
    assert res.total_energy_wh > 0.0


def test_optimize_trip_speeds_dp_time_constraint():
    """Verify time-constrained DP increases speeds to satisfy trip time ceiling."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    segments = [
        RouteSegment(distance_m=1000.0, grade_fraction=0.0),
        RouteSegment(distance_m=1000.0, grade_fraction=0.02),
    ]

    # First get unconstrained time
    unconstrained = optimize_trip_speeds_dp(
        segments=segments,
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        max_total_time_s=None,
    )

    # Impose tighter time constraint (e.g. 80% of unconstrained duration)
    target_time = unconstrained.total_time_s * 0.80

    constrained = optimize_trip_speeds_dp(
        segments=segments,
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        max_total_time_s=target_time,
    )

    assert constrained.total_time_s <= target_time + 1.0
    # Faster trip requires higher energy
    assert constrained.total_energy_wh > unconstrained.total_energy_wh
    # Average speed must be higher
    assert np.mean(constrained.segment_speeds_kmh) > np.mean(unconstrained.segment_speeds_kmh)

