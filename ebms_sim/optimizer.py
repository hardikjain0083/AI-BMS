"""Speed optimization and segment-level dynamic programming for range maximization.

Finds the energy-optimal cruising speed (10-25 km/h) and optimal speed schedule
across multi-segment routes subject to an optional maximum trip time constraint.
Pure functions with no external state.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from ebms_sim.config import VehicleParams, BatteryParams
from ebms_sim.vehicle import air_density_from_temp, step_vehicle_physics
from ebms_sim.battery import (
    ocv_from_soc,
    internal_resistance_ohm,
    solve_current_and_voltage,
    usable_capacity_factor,
)


@dataclass(frozen=True)
class RouteSegment:
    """Properties of an individual road route segment.

    Attributes:
        distance_m: Length of segment in meters [m].
        grade_fraction: Slope gradient (m rise / m run) [-0.12 .. 0.12].
        headwind_ms: Headwind component [m/s].
    """
    distance_m: float
    grade_fraction: float = 0.0
    headwind_ms: float = 0.0


@dataclass(frozen=True)
class SegmentOptimizationResult:
    """Optimal speed and consumption schedule for a multi-segment route.

    Attributes:
        segment_speeds_kmh: Recommended speed for each segment [km/h].
        segment_times_s: Travel duration for each segment [s].
        segment_energies_wh: Energy consumed in each segment [Wh].
        total_time_s: Total trip time [s].
        total_energy_wh: Total energy consumed [Wh].
        total_distance_km: Total route distance [km].
        avg_wh_per_km: Net energy consumption rate [Wh/km].
    """
    segment_speeds_kmh: List[float]
    segment_times_s: List[float]
    segment_energies_wh: List[float]
    total_time_s: float
    total_energy_wh: float
    total_distance_km: float
    avg_wh_per_km: float


def calculate_steady_state_power_w(
    speed_kmh: float,
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    grade_fraction: float = 0.0,
    headwind_ms: float = 0.0,
    payload_mass_kg: float = 130.0,
    t_ambient_c: float = 25.0,
    soc: float = 0.80,
    soh: float = 1.0,
) -> float:
    """Calculate steady-state electrical battery power at constant speed.

    Args:
        speed_kmh: Constant speed [km/h].
        vehicle_params: Vehicle parameters.
        battery_params: Battery parameters.
        grade_fraction: Road gradient.
        headwind_ms: Headwind speed [m/s].
        payload_mass_kg: Total payload mass [kg].
        t_ambient_c: Ambient temperature [deg C].
        soc: Battery state of charge [0..1].
        soh: Battery state of health [0..1].

    Returns:
        Battery power drawn in Watts [W].
    """
    v_ms = speed_kmh / 3.6
    air_density = air_density_from_temp(t_ambient_c)

    phys = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=grade_fraction,
        headwind_ms=headwind_ms,
        payload_mass_kg=payload_mass_kg,
        vehicle_params=vehicle_params,
        air_density_kg_m3=air_density,
    )
    p_batt = phys["battery_power_w"]

    # Account for internal battery I^2*R loss
    r_int = internal_resistance_ohm(
        battery_params.r0_ohm, t_ambient_c, soh, battery_params.temp_coeff_r, battery_params.soh_coeff_r
    )
    ocv = ocv_from_soc(soc, battery_params.chemistry)
    curr, volt = solve_current_and_voltage(p_batt, ocv, r_int)

    # Effective electrochemical power: OCV * I
    return ocv * curr


def sweep_speed_wh_per_km(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    payload_mass_kg: float = 130.0,
    grade_fraction: float = 0.0,
    headwind_ms: float = 0.0,
    t_ambient_c: float = 25.0,
    speeds_kmh: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Sweep vehicle speed from 10 to 25 km/h to compute Wh/km curve.

    The curve is U-shaped:
    - At low speeds (<12 km/h), fixed auxiliary power (P_aux / v) dominates.
    - At high speeds (>20 km/h), aerodynamic drag (v^2) dominates.

    Args:
        vehicle_params: Vehicle parameters.
        battery_params: Battery parameters.
        payload_mass_kg: Payload mass [kg].
        grade_fraction: Road slope [m/m].
        headwind_ms: Headwind [m/s].
        t_ambient_c: Ambient temperature [deg C].
        speeds_kmh: Array of speeds to sweep [km/h] (default 10 to 25 with 0.5 step).

    Returns:
        Tuple of (speeds_kmh_array, wh_per_km_array, best_speed_kmh, min_wh_per_km).
    """
    if speeds_kmh is None:
        speeds_kmh = np.linspace(10.0, 25.0, 31)

    wh_per_km_list = []
    for spd in speeds_kmh:
        if spd <= 0.0:
            wh_km = float("inf")
        else:
            p_elec = calculate_steady_state_power_w(
                speed_kmh=spd,
                vehicle_params=vehicle_params,
                battery_params=battery_params,
                grade_fraction=grade_fraction,
                headwind_ms=headwind_ms,
                payload_mass_kg=payload_mass_kg,
                t_ambient_c=t_ambient_c,
            )
            wh_km = p_elec / spd
        wh_per_km_list.append(wh_km)

    wh_per_km_arr = np.array(wh_per_km_list, dtype=np.float64)
    best_idx = int(np.argmin(wh_per_km_arr))
    best_speed = float(speeds_kmh[best_idx])
    min_wh_km = float(wh_per_km_arr[best_idx])

    return speeds_kmh, wh_per_km_arr, best_speed, min_wh_km


def optimize_trip_speeds_dp(
    segments: List[RouteSegment],
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    payload_mass_kg: float = 130.0,
    t_ambient_c: float = 25.0,
    max_total_time_s: Optional[float] = None,
    candidate_speeds_kmh: Optional[np.ndarray] = None,
) -> SegmentOptimizationResult:
    """Recommend per-segment speeds (10-25 km/h) that maximize range / minimize energy.

    Subject to an optional max_total_time_s constraint.
    Solved via dual Lagrangian decomposition / dynamic programming:
    min sum(E_i) + lambda * sum(T_i).

    Args:
        segments: List of RouteSegment objects along the trip.
        vehicle_params: Vehicle configuration.
        battery_params: Battery pack parameters.
        payload_mass_kg: Total payload [kg].
        t_ambient_c: Ambient temperature [deg C].
        max_total_time_s: Optional total trip time ceiling [seconds].
        candidate_speeds_kmh: Discrete speeds to evaluate in each segment.

    Returns:
        SegmentOptimizationResult containing optimal speeds, times, and energies.
    """
    if candidate_speeds_kmh is None:
        candidate_speeds_kmh = np.linspace(10.0, 25.0, 16)  # 10.0, 11.0, ... 25.0 km/h

    n_segs = len(segments)
    if n_segs == 0:
        raise ValueError("segments list cannot be empty")

    # Precompute time and energy matrices: shape (n_segs, n_candidates)
    n_cands = len(candidate_speeds_kmh)
    times_matrix = np.zeros((n_segs, n_cands), dtype=np.float64)
    energies_matrix = np.zeros((n_segs, n_cands), dtype=np.float64)

    for i, seg in enumerate(segments):
        for j, spd in enumerate(candidate_speeds_kmh):
            v_ms = spd / 3.6
            dur_s = seg.distance_m / v_ms
            times_matrix[i, j] = dur_s

            p_w = calculate_steady_state_power_w(
                speed_kmh=spd,
                vehicle_params=vehicle_params,
                battery_params=battery_params,
                grade_fraction=seg.grade_fraction,
                headwind_ms=seg.headwind_ms,
                payload_mass_kg=payload_mass_kg,
                t_ambient_c=t_ambient_c,
            )
            # Energy in Wh = P (W) * t (s) / 3600
            energies_matrix[i, j] = (p_w * dur_s) / 3600.0

    min_possible_time = np.sum(np.min(times_matrix, axis=1))
    max_possible_time = np.sum(np.max(times_matrix, axis=1))

    # If no time constraint or generous time limit, choose speed that minimizes energy per segment
    if max_total_time_s is None or max_total_time_s >= max_possible_time:
        best_indices = np.argmin(energies_matrix, axis=1)
    elif max_total_time_s <= min_possible_time:
        # Heavily constrained: run at maximum candidate speed
        best_indices = np.argmin(times_matrix, axis=1)
    else:
        # Binary search over Lagrange multiplier lambda
        lambda_low = 0.0
        lambda_high = 50.0  # Wh/s trade-off upper bound
        best_indices = np.argmin(energies_matrix, axis=1)

        for _ in range(40):  # Bisection iterations
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            # Cost = Energy + lambda * Time
            cost_matrix = energies_matrix + lambda_mid * times_matrix
            trial_indices = np.argmin(cost_matrix, axis=1)
            trial_time = np.sum(times_matrix[np.arange(n_segs), trial_indices])

            if trial_time <= max_total_time_s:
                best_indices = trial_indices
                lambda_high = lambda_mid
            else:
                lambda_low = lambda_mid

    opt_speeds = [float(candidate_speeds_kmh[idx]) for idx in best_indices]
    opt_times = [float(times_matrix[i, idx]) for i, idx in enumerate(best_indices)]
    opt_energies = [float(energies_matrix[i, idx]) for i, idx in enumerate(best_indices)]

    tot_dist_km = sum(s.distance_m for s in segments) / 1000.0
    tot_time = sum(opt_times)
    tot_energy = sum(opt_energies)
    avg_wh_km = (tot_energy / tot_dist_km) if tot_dist_km > 0.0 else 0.0

    return SegmentOptimizationResult(
        segment_speeds_kmh=opt_speeds,
        segment_times_s=opt_times,
        segment_energies_wh=opt_energies,
        total_time_s=float(tot_time),
        total_energy_wh=float(tot_energy),
        total_distance_km=float(tot_dist_km),
        avg_wh_per_km=float(avg_wh_km),
    )

