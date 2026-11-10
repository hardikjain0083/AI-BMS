"""Speed profile and route elevation / gradient profile generators.

Generates realistic stop-and-go driving patterns and road gradient traces
representative of Indian e-rickshaw operating environments.
Pure functions with deterministic random seeding.
"""

from __future__ import annotations
import math
from typing import Literal, Tuple, Union
import numpy as np

RouteType = Literal["flat", "rolling", "hill-climb", "hill-descent", "mixed"]


def generate_speed_profile(
    duration_s: int = 600,
    speed_cap_kmh: float = 25.0,
    stop_interval_s: float = 60.0,
    stop_duration_s: float = 8.0,
    aggressiveness: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a realistic 1-second resolution speed profile in m/s.

    Simulates urban e-rickshaw traffic:
    - Periodic stop-and-go passenger pickups/signals.
    - Accelerates up to cruising speed bounded by speed_cap_kmh.
    - Driver aggressiveness modulates acceleration/deceleration rates.

    Args:
        duration_s: Total profile duration in seconds [s].
        speed_cap_kmh: Maximum allowed vehicle speed [km/h] (legal cap 25.0).
        stop_interval_s: Average interval between passenger/traffic stops [s].
        stop_duration_s: Average idle dwell time per stop [s].
        aggressiveness: Driver aggressiveness factor [0.5 = gentle, 1.0 = normal, 1.5 = aggressive].
        seed: Random seed for deterministic generation.

    Returns:
        1D numpy array of vehicle speed at 1 s intervals in m/s [m/s].
    """
    rng = np.random.default_rng(seed)
    max_speed_ms = speed_cap_kmh / 3.6
    v_profile = np.zeros(duration_s, dtype=np.float64)

    # Base acceleration and deceleration limits [m/s^2]
    max_accel_ms2 = 0.65 * max(0.4, min(2.0, aggressiveness))
    max_decel_ms2 = 0.85 * max(0.4, min(2.0, aggressiveness))

    # Generate stop events
    current_speed = 0.0
    t = 0
    next_stop_time = int(rng.uniform(stop_interval_s * 0.7, stop_interval_s * 1.3))
    stop_until_time = -1

    while t < duration_s:
        if t == next_stop_time:
            # Begin stopping
            actual_stop_dur = int(rng.uniform(stop_duration_s * 0.6, stop_duration_s * 1.4))
            stop_until_time = t + actual_stop_dur
            next_stop_time = stop_until_time + int(rng.uniform(stop_interval_s * 0.7, stop_interval_s * 1.3))

        if t < stop_until_time:
            # Decelerate to 0 and dwell
            current_speed = max(0.0, current_speed - max_decel_ms2)
        else:
            # Desired cruising speed with slight road/traffic noise
            noise = rng.normal(0.0, 0.3)
            desired_speed = max_speed_ms * (0.85 + 0.15 * rng.uniform(0.0, 1.0)) + noise
            desired_speed = max(0.0, min(max_speed_ms, desired_speed))

            if current_speed < desired_speed:
                current_speed = min(desired_speed, current_speed + max_accel_ms2)
            elif current_speed > desired_speed:
                current_speed = max(desired_speed, current_speed - max_decel_ms2)

        v_profile[t] = max(0.0, min(max_speed_ms, current_speed))
        t += 1

    return v_profile


def generate_gradient_profile(
    length_steps: int = 600,
    route_type: RouteType = "flat",
    max_grade: float = 0.12,
    seed: int | None = None,
) -> np.ndarray:
    """Generate road gradient array clipped to [-max_grade, +max_grade].

    Args:
        length_steps: Number of 1-second simulation steps.
        route_type: One of ('flat', 'rolling', 'hill-climb', 'hill-descent', 'mixed').
        max_grade: Maximum permissible road grade fraction (default 0.12 = 12%).
        seed: Random seed for repeatability.

    Returns:
        1D numpy array of road gradient (m rise / m run) for each step.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length_steps)
    clipped_limit = min(0.12, max(0.01, max_grade))

    if route_type == "flat":
        # Minor road irregularities within +/- 0.5%
        gradient = rng.normal(0.0, 0.002, size=length_steps)

    elif route_type == "rolling":
        # Undulating rolling hills: sinusoidal waves with periods of ~100-200 s
        wave1 = 0.025 * np.sin(2.0 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
        wave2 = 0.015 * np.sin(2.0 * np.pi * 7.0 * t + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0.0, 0.002, size=length_steps)
        gradient = wave1 + wave2 + noise

    elif route_type == "hill-climb":
        # Predominantly uphill gradient: +3% to +9% with realistic variations
        base_grade = rng.uniform(0.04, 0.08)
        variation = 0.02 * np.sin(2.0 * np.pi * 2.0 * t)
        noise = rng.normal(0.0, 0.003, size=length_steps)
        gradient = base_grade + variation + noise

    elif route_type == "hill-descent":
        # Predominantly downhill gradient: -3% to -9%
        base_grade = -rng.uniform(0.04, 0.08)
        variation = 0.02 * np.sin(2.0 * np.pi * 2.0 * t)
        noise = rng.normal(0.0, 0.003, size=length_steps)
        gradient = base_grade + variation + noise

    elif route_type == "mixed":
        # Combines flat, climb, rolling, and descent segments
        n_segs = 4
        seg_len = length_steps // n_segs
        segments = []
        seg_types: list[RouteType] = ["flat", "hill-climb", "rolling", "hill-descent"]
        rng.shuffle(seg_types)
        for i, st in enumerate(seg_types):
            this_len = seg_len if i < n_segs - 1 else length_steps - (n_segs - 1) * seg_len
            seg_grad = generate_gradient_profile(
                length_steps=this_len,
                route_type=st,
                max_grade=clipped_limit,
                seed=rng.integers(0, 1000000),
            )
            segments.append(seg_grad)
        gradient = np.concatenate(segments)

    else:
        raise ValueError(f"Unknown route_type '{route_type}'. Available: flat, rolling, hill-climb, hill-descent, mixed")

    # Strictly clip to +/- max_grade (+-12%)
    return np.clip(gradient, -clipped_limit, clipped_limit)


def cumulative_elevation_m(
    gradient_profile: np.ndarray,
    step_distances_m: Union[np.ndarray, float],
) -> np.ndarray:
    """Calculate cumulative elevation profile in meters above start.

    dz = gradient * dx
    elevation = cumsum(dz)

    Args:
        gradient_profile: 1D array of road gradients.
        step_distances_m: 1D array or constant distance traveled per step [m].

    Returns:
        1D array of cumulative altitude change [m].
    """
    delta_z = gradient_profile * step_distances_m
    return np.cumsum(delta_z)

