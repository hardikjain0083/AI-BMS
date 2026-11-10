"""Trip simulation engine at 1-second time resolution.

Simulates vehicle dynamics, battery electrochemical state, and pack thermal
evolution over a full driving cycle.
Pure functions returning structured time-series traces and summary statistics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

from ebms_sim.config import VehicleParams, BatteryParams, ThermalParams
from ebms_sim.vehicle import (
    air_density_from_temp,
    step_vehicle_physics,
)
from ebms_sim.battery import (
    ocv_from_soc,
    internal_resistance_ohm,
    solve_current_and_voltage,
    update_soc_coulomb_counting,
    usable_capacity_factor,
)
from ebms_sim.thermal import step_pack_temperature_c
from ebms_sim.profiles import cumulative_elevation_m


@dataclass(frozen=True)
class TripSummary:
    """Summary metrics of a completed or cutoff trip simulation.

    Attributes:
        distance_km: Total distance covered [km].
        duration_s: Total trip duration [s].
        avg_speed_kmh: Average moving speed [km/h].
        total_energy_wh: Net energy consumed from battery [Wh].
        wh_per_km: Specific energy consumption [Wh/km].
        range_remaining_km: Projected remaining driving range until cutoff SOC [km].
        cutoff_reached: True if battery hit low-SOC cutoff during trip.
        final_soc: SOC at the end of trip [0..1].
        final_temp_c: Pack temperature at the end of trip [deg C].
    """
    distance_km: float
    duration_s: float
    avg_speed_kmh: float
    total_energy_wh: float
    wh_per_km: float
    range_remaining_km: float
    cutoff_reached: bool
    final_soc: float
    final_temp_c: float


@dataclass
class TripResult:
    """Complete trip output containing time series DataFrame and summary."""
    time_series: pd.DataFrame
    summary: TripSummary


def estimate_remaining_range_km(
    current_soc: float,
    cutoff_soc: float,
    nominal_capacity_ah: float,
    nominal_voltage_v: float,
    temp_c: float,
    wh_per_km: float,
) -> float:
    """Calculate remaining driving range until battery hits cutoff SOC.

    Pure function ensuring:
    - If current_soc <= cutoff_soc, range is strictly 0.0.
    - As current_soc increases, range increases monotonically.

    Args:
        current_soc: Battery state of charge [0.0 .. 1.0].
        cutoff_soc: Low-SOC cutoff threshold [0.0 .. 1.0].
        nominal_capacity_ah: Nominal battery capacity [Ah].
        nominal_voltage_v: Nominal pack voltage [V].
        temp_c: Pack temperature [deg C].
        wh_per_km: Expected consumption rate [Wh/km].

    Returns:
        Remaining range in kilometers [km].
    """
    if current_soc <= cutoff_soc:
        return 0.0

    usable_soc = current_soc - cutoff_soc
    cap_derating = usable_capacity_factor(temp_c)
    usable_energy_wh = usable_soc * nominal_capacity_ah * nominal_voltage_v * cap_derating

    # Use realistic baseline consumption if wh_per_km is zero/negative (e.g. idle/downhill)
    effective_wh_km = max(20.0, wh_per_km)
    return max(0.0, usable_energy_wh / effective_wh_km)


def simulate_trip(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    speed_profile_ms: Union[np.ndarray, list],
    gradient_profile: Union[np.ndarray, list],
    payload_mass_kg: float = 130.0,
    t_ambient_c: float = 25.0,
    headwind_ms: float = 0.0,
    start_soc: float = 0.90,
    start_temp_c: Optional[float] = None,
    soh: float = 1.0,
    cutoff_soc: Optional[float] = None,
    dt_s: float = 1.0,
) -> TripResult:
    """Run a first-principles 1-second step physics simulation of a vehicle trip.

    Args:
        vehicle_params: Vehicle configuration dataclass.
        battery_params: Battery pack dataclass.
        thermal_params: Lumped thermal parameters.
        speed_profile_ms: Array of speeds in m/s at dt_s intervals.
        gradient_profile: Array of road slopes (m/m) at dt_s intervals.
        payload_mass_kg: Payload mass (passengers + luggage) [kg].
        t_ambient_c: Ambient air temperature [deg C].
        headwind_ms: Headwind velocity component [m/s].
        start_soc: Initial battery SOC [0.0 .. 1.0].
        start_temp_c: Initial battery pack temperature [deg C] (defaults to t_ambient_c).
        soh: Battery State of Health [0.0 .. 1.0].
        cutoff_soc: Depletion cutoff SOC (defaults to battery_params.cutoff_soc).
        dt_s: Time step in seconds [s].

    Returns:
        TripResult with full time_series DataFrame and TripSummary.
    """
    v_arr = np.asarray(speed_profile_ms, dtype=np.float64)
    g_arr = np.asarray(gradient_profile, dtype=np.float64)
    n_steps = len(v_arr)

    if len(g_arr) != n_steps:
        raise ValueError(f"Length mismatch: speed_profile ({n_steps}) vs gradient_profile ({len(g_arr)})")

    active_cutoff_soc = cutoff_soc if cutoff_soc is not None else battery_params.cutoff_soc
    current_temp = start_temp_c if start_temp_c is not None else t_ambient_c
    current_soc = max(0.0, min(1.0, start_soc))

    air_density = air_density_from_temp(t_ambient_c)

    # Preallocate output arrays
    time_s = np.arange(n_steps, dtype=np.float64) * dt_s
    accel_ms2 = np.zeros(n_steps, dtype=np.float64)
    p_wheel_w = np.zeros(n_steps, dtype=np.float64)
    p_batt_w = np.zeros(n_steps, dtype=np.float64)
    current_a = np.zeros(n_steps, dtype=np.float64)
    voltage_v = np.zeros(n_steps, dtype=np.float64)
    soc_arr = np.zeros(n_steps, dtype=np.float64)
    temp_c_arr = np.zeros(n_steps, dtype=np.float64)
    distance_step_m = np.zeros(n_steps, dtype=np.float64)
    energy_step_wh = np.zeros(n_steps, dtype=np.float64)

    cutoff_reached = False
    cutoff_step = n_steps

    for i in range(n_steps):
        v = v_arr[i]
        grad = g_arr[i]

        # Acceleration (forward difference at start, backward difference afterwards)
        if i == 0:
            a = (v_arr[1] - v_arr[0]) / dt_s if n_steps > 1 else 0.0
        else:
            a = (v - v_arr[i - 1]) / dt_s
        accel_ms2[i] = a

        # Distance step
        dx = v * dt_s
        distance_step_m[i] = dx

        # Check cutoff before driving step
        if current_soc <= active_cutoff_soc and not cutoff_reached:
            cutoff_reached = True
            cutoff_step = i

        # Step vehicle dynamics
        phys = step_vehicle_physics(
            v_curr_ms=v,
            acceleration_ms2=a,
            grade_fraction=grad,
            headwind_ms=headwind_ms,
            payload_mass_kg=payload_mass_kg,
            vehicle_params=vehicle_params,
            air_density_kg_m3=air_density,
        )

        pw = phys["wheel_power_w"]
        pb = phys["battery_power_w"]
        p_wheel_w[i] = pw
        p_batt_w[i] = pb

        # Battery internal resistance & OCV
        r_int = internal_resistance_ohm(
            r0_base_ohm=battery_params.r0_ohm,
            temp_c=current_temp,
            soh=soh,
            temp_coeff_r=battery_params.temp_coeff_r,
            soh_coeff_r=battery_params.soh_coeff_r,
        )
        ocv = ocv_from_soc(current_soc, chemistry=battery_params.chemistry)

        # Solve for terminal current and voltage
        curr, volt = solve_current_and_voltage(pb, ocv, r_int)

        # Enforce maximum current clamp if needed
        if curr > battery_params.max_continuous_current_a:
            curr = battery_params.max_continuous_current_a
            volt = ocv - curr * r_int

        current_a[i] = curr
        voltage_v[i] = volt
        soc_arr[i] = current_soc
        temp_c_arr[i] = current_temp

        # Energy consumed during step: P_batt * dt (Wh)
        e_step = (pb * dt_s) / 3600.0
        energy_step_wh[i] = e_step

        # Update SOC for next step
        current_soc = update_soc_coulomb_counting(
            soc=current_soc,
            current_a=curr,
            dt_s=dt_s,
            nominal_capacity_ah=battery_params.nominal_capacity_ah,
            temp_c=current_temp,
            peukert_exp=battery_params.peukert_exp,
            peukert_i_ref_a=battery_params.peukert_i_ref_a,
        )

        # Update pack temperature for next step
        current_temp = step_pack_temperature_c(
            t_pack_c=current_temp,
            current_a=curr,
            r_ohm=r_int,
            t_ambient_c=t_ambient_c,
            c_th_j_per_k=thermal_params.c_th_j_per_k,
            r_th_k_per_w=thermal_params.r_th_k_per_w,
            dt_s=dt_s,
            v_ms=v,
            cooling_factor=thermal_params.cooling_air_velocity_factor,
        )

    # Calculate cumulative distance and elevation
    cum_dist_m = np.cumsum(distance_step_m)
    cum_elev_m = cumulative_elevation_m(g_arr, distance_step_m)
    cum_energy_wh = np.cumsum(energy_step_wh)

    total_dist_km = cum_dist_m[-1] / 1000.0
    total_energy = cum_energy_wh[-1]
    avg_speed = (total_dist_km / (n_steps * dt_s / 3600.0)) if n_steps > 0 else 0.0

    wh_km = (total_energy / total_dist_km) if total_dist_km > 0.05 else 0.0

    # Remaining range calculation
    final_soc = current_soc
    rem_range = estimate_remaining_range_km(
        current_soc=final_soc,
        cutoff_soc=active_cutoff_soc,
        nominal_capacity_ah=battery_params.nominal_capacity_ah,
        nominal_voltage_v=battery_params.nominal_voltage_v,
        temp_c=current_temp,
        wh_per_km=wh_km,
    )

    df_ts = pd.DataFrame({
        "time_s": time_s,
        "speed_ms": v_arr,
        "speed_kmh": v_arr * 3.6,
        "accel_ms2": accel_ms2,
        "gradient": g_arr,
        "elevation_m": cum_elev_m,
        "wheel_power_w": p_wheel_w,
        "battery_power_w": p_batt_w,
        "current_a": current_a,
        "voltage_v": voltage_v,
        "soc": soc_arr,
        "temp_c": temp_c_arr,
        "distance_m": cum_dist_m,
        "energy_wh": cum_energy_wh,
    })

    summary = TripSummary(
        distance_km=float(total_dist_km),
        duration_s=float(n_steps * dt_s),
        avg_speed_kmh=float(avg_speed),
        total_energy_wh=float(total_energy),
        wh_per_km=float(wh_km),
        range_remaining_km=float(rem_range),
        cutoff_reached=bool(cutoff_reached),
        final_soc=float(final_soc),
        final_temp_c=float(current_temp),
    )

    return TripResult(time_series=df_ts, summary=summary)
