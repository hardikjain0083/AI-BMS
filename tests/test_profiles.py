"""Unit tests for speed and gradient profile generators (ebms_sim.profiles)."""

import pytest
import numpy as np
from ebms_sim.profiles import (
    generate_speed_profile,
    generate_gradient_profile,
    cumulative_elevation_m,
)


def test_speed_profile_bounds_and_duration():
    """Verify speed profile stays within [0, speed_cap] and matches length."""
    duration = 300
    cap_kmh = 22.0
    cap_ms = cap_kmh / 3.6

    profile = generate_speed_profile(duration_s=duration, speed_cap_kmh=cap_kmh, seed=42)

    assert len(profile) == duration
    assert np.all(profile >= 0.0)
    assert np.all(profile <= cap_ms + 1e-6)
    # Check that it actually reached close to the cap
    assert np.max(profile) > 0.8 * cap_ms


def test_speed_profile_has_stops():
    """Verify realistic stop-go cycle contains zero-speed dwell intervals."""
    profile = generate_speed_profile(
        duration_s=600, speed_cap_kmh=25.0, stop_interval_s=50.0, stop_duration_s=10.0, seed=123
    )
    zero_steps = np.sum(profile == 0.0)
    assert zero_steps > 10, "Expected at least 10 stopped seconds over 600 s trip"


def test_speed_profile_seed_determinism():
    """Verify deterministic repeatability when seed is provided."""
    p1 = generate_speed_profile(duration_s=200, speed_cap_kmh=20.0, seed=999)
    p2 = generate_speed_profile(duration_s=200, speed_cap_kmh=20.0, seed=999)
    p3 = generate_speed_profile(duration_s=200, speed_cap_kmh=20.0, seed=1000)

    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_gradient_profile_clipped_to_12_percent():
    """Verify all route gradient profiles strictly obey +/- 12% limit."""
    routes = ["flat", "rolling", "hill-climb", "hill-descent", "mixed"]
    for r in routes:
        grad = generate_gradient_profile(length_steps=400, route_type=r, max_grade=0.12, seed=42)
        assert len(grad) == 400
        assert np.all(grad >= -0.12 - 1e-6), f"Exceeded -12% on {r}"
        assert np.all(grad <= 0.12 + 1e-6), f"Exceeded +12% on {r}"


def test_hill_climb_and_descent_means():
    """Verify climb is predominantly positive and descent predominantly negative."""
    climb = generate_gradient_profile(length_steps=500, route_type="hill-climb", seed=42)
    descent = generate_gradient_profile(length_steps=500, route_type="hill-descent", seed=42)
    flat = generate_gradient_profile(length_steps=500, route_type="flat", seed=42)

    assert np.mean(climb) > 0.03
    assert np.mean(descent) < -0.03
    assert abs(np.mean(flat)) < 0.005


def test_cumulative_elevation():
    """Verify cumulative elevation matches integral of gradient * distance."""
    gradient = np.array([0.05, 0.05, -0.02, 0.0])  # 5%, 5%, -2%, 0%
    dx = 100.0  # 100 m per step
    elev = cumulative_elevation_m(gradient, dx)

    # Step 1: 5m, Step 2: 10m, Step 3: 8m, Step 4: 8m
    expected = np.array([5.0, 10.0, 8.0, 8.0])
    np.testing.assert_allclose(elev, expected)

