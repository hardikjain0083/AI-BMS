"""Sanity-check module for ebms-sim physics and chemistry models.

Pure functions that execute self-contained physical health checks and return
structured test outcomes: (check_name, passed, detail_message).
Used by automated test suites and the interactive web application.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from ebms_sim.config import PRESETS, VehicleParams, BatteryParams, ThermalParams
from ebms_sim.vehicle import step_vehicle_physics, air_density_from_temp
from ebms_sim.trip import simulate_trip, estimate_remaining_range_km
from ebms_sim.charging import max_charge_current_a, temperature_charge_multiplier


def check_wh_per_km_benchmark(
    vehicle_params: Optional[VehicleParams] = None,
    battery_params: Optional[BatteryParams] = None,
    thermal_params: Optional[ThermalParams] = None,
) -> Tuple[str, bool, str]:
    """Check 1: Total Wh/km for a flat road, 2 passengers (130 kg), 20 km/h, 25C is between 25 and 70 Wh/km."""
    v_params = vehicle_params or PRESETS["lead_acid_48V_100Ah"]["vehicle"]
    b_params = battery_params or PRESETS["lead_acid_48V_100Ah"]["battery"]
    t_params = thermal_params or PRESETS["lead_acid_48V_100Ah"]["thermal"]

    v_kmh = 20.0
    v_ms = v_kmh / 3.6
    duration_s = 100
    speed_profile = np.full(duration_s, v_ms)
    gradient_profile = np.zeros(duration_s)

    res = simulate_trip(
        vehicle_params=v_params,
        battery_params=b_params,
        thermal_params=t_params,
        speed_profile_ms=speed_profile,
        gradient_profile=gradient_profile,
        payload_mass_kg=130.0,
        t_ambient_c=25.0,
        headwind_ms=0.0,
        start_soc=0.85,
    )

    wh_km = res.summary.wh_per_km
    passed = 25.0 <= wh_km <= 70.0
    msg = f"Observed {wh_km:.2f} Wh/km at 20 km/h on flat terrain (expected range: 25.0 - 70.0 Wh/km)"
    return "Flat road cruise benchmark (20 km/h, 2 pax, 25°C)", passed, msg


def check_consumption_monotonicity(
    vehicle_params: Optional[VehicleParams] = None,
) -> Tuple[str, bool, str]:
    """Check 2: Consumption increases with passengers, with uphill grade, and with headwind."""
    v_params = vehicle_params or PRESETS["lead_acid_48V_100Ah"]["vehicle"]
    v_ms = 20.0 / 3.6
    rho = air_density_from_temp(25.0)

    # Base case: 0 passengers, flat road, 0 wind
    p_base = step_vehicle_physics(v_ms, 0.0, 0.0, 0.0, 0.0, v_params, rho)["battery_power_w"]

    # More passengers (+260 kg)
    p_pass = step_vehicle_physics(v_ms, 0.0, 0.0, 0.0, 260.0, v_params, rho)["battery_power_w"]

    # Uphill slope (+4%)
    p_grade = step_vehicle_physics(v_ms, 0.0, 0.04, 0.0, 0.0, v_params, rho)["battery_power_w"]

    # Headwind (+3 m/s)
    p_wind = step_vehicle_physics(v_ms, 0.0, 0.0, 3.0, 0.0, v_params, rho)["battery_power_w"]

    pass_pass = p_pass > p_base
    pass_grade = p_grade > p_base
    pass_wind = p_wind > p_base
    passed = pass_pass and pass_grade and pass_wind

    msg = (
        f"Base: {p_base:.1f}W | +Pax: {p_pass:.1f}W (+{(p_pass - p_base):.1f}W) | "
        f"+Grade: {p_grade:.1f}W (+{(p_grade - p_base):.1f}W) | "
        f"+Headwind: {p_wind:.1f}W (+{(p_wind - p_base):.1f}W)"
    )
    return "Consumption monotonicity (passengers, grade, headwind)", passed, msg


def check_downhill_zero_regen_aux_power(
    vehicle_params: Optional[VehicleParams] = None,
) -> Tuple[str, bool, str]:
    """Check 3: Downhill with eta_regen=0 gives battery power == aux power."""
    base_v = vehicle_params or PRESETS["lead_acid_48V_100Ah"]["vehicle"]
    v_zero_regen = VehicleParams(
        curb_mass_kg=base_v.curb_mass_kg,
        driver_mass_kg=base_v.driver_mass_kg,
        crr=base_v.crr,
        cda_m2=base_v.cda_m2,
        eta_drive=base_v.eta_drive,
        eta_regen=0.0,
        aux_w=base_v.aux_w,
        rotational_inertia_factor=base_v.rotational_inertia_factor,
        max_speed_kmh=base_v.max_speed_kmh,
        eta_model=base_v.eta_model,
        loss_p0_w=base_v.loss_p0_w,
        loss_c1=base_v.loss_c1,
        loss_c2=base_v.loss_c2,
    )

    v_ms = 20.0 / 3.6
    # -8% steep downhill where gravity overcomes drag and rolling resistance
    res = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=-0.08,
        headwind_ms=0.0,
        payload_mass_kg=130.0,
        vehicle_params=v_zero_regen,
    )

    wheel_p = res["wheel_power_w"]
    batt_p = res["battery_power_w"]

    is_gravity_propelling = wheel_p < 0.0
    matches_aux = abs(batt_p - v_zero_regen.aux_w) < 1e-4
    passed = is_gravity_propelling and matches_aux

    msg = (
        f"Wheel power: {wheel_p:.1f}W (<0) -> Battery power: {batt_p:.2f}W "
        f"(matches aux power: {v_zero_regen.aux_w:.2f}W)"
    )
    return "Downhill zero-regen floor (battery power == aux power)", passed, msg


def check_range_soc_monotonicity(
    battery_params: Optional[BatteryParams] = None,
) -> Tuple[str, bool, str]:
    """Check 4: Range at SOC <= cutoff is 0; range increases monotonically with SOC."""
    b_params = battery_params or PRESETS["lead_acid_48V_100Ah"]["battery"]
    cutoff = b_params.cutoff_soc
    nom_cap = b_params.nominal_capacity_ah
    nom_v = b_params.nominal_voltage_v

    # Zero at or below cutoff
    r_at_cutoff = estimate_remaining_range_km(cutoff, cutoff, nom_cap, nom_v, 25.0, 40.0)
    r_below_cutoff = estimate_remaining_range_km(cutoff - 0.05, cutoff, nom_cap, nom_v, 25.0, 40.0)
    zero_pass = (r_at_cutoff == 0.0) and (r_below_cutoff == 0.0)

    # Monotonicity test
    soc_steps = [0.20, 0.40, 0.60, 0.80, 1.00]
    ranges = [estimate_remaining_range_km(s, cutoff, nom_cap, nom_v, 25.0, 40.0) for s in soc_steps]
    monotonic = all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1))
    passed = zero_pass and monotonic

    msg = f"Range at cutoff={r_at_cutoff:.1f} km; Range increases from {ranges[0]:.1f} km (20% SOC) to {ranges[-1]:.1f} km (100% SOC)"
    return "Range cutoff zeroing & SOC monotonicity", passed, msg


def check_charging_temperature_window(
    battery_params: Optional[BatteryParams] = None,
) -> Tuple[str, bool, str]:
    """Check 5: Charging current is 0 outside the allowed temperature window."""
    chem = battery_params.chemistry if battery_params else "lfp"

    if chem == "lead_acid":
        # Lead-acid: 0 outside 0 to 50 C
        i_subzero = max_charge_current_a(-5.0, 0.5, chemistry="lead_acid")
        i_overtemp = max_charge_current_a(52.0, 0.5, chemistry="lead_acid")
        i_valid = max_charge_current_a(25.0, 0.5, chemistry="lead_acid")
        passed = (i_subzero == 0.0) and (i_overtemp == 0.0) and (i_valid > 0.0)
        msg = f"Lead-acid: 0 A at -5°C, 0 A at 52°C, {i_valid:.1f} A at 25°C (safe window 0°C to 50°C)"
    else:
        # LFP: 0 outside 0 to 45 C
        i_subzero = max_charge_current_a(-5.0, 0.5, chemistry="lfp")
        i_overtemp = max_charge_current_a(46.0, 0.5, chemistry="lfp")
        i_valid = max_charge_current_a(25.0, 0.5, chemistry="lfp")
        passed = (i_subzero == 0.0) and (i_overtemp == 0.0) and (i_valid > 0.0)
        msg = f"LFP: 0 A at -5°C, 0 A at 46°C, {i_valid:.1f} A at 25°C (safe window 0°C to 45°C)"

    return f"Charging temperature safety window ({chem})", passed, msg


def run_all_sanity_checks(
    vehicle_params: Optional[VehicleParams] = None,
    battery_params: Optional[BatteryParams] = None,
    thermal_params: Optional[ThermalParams] = None,
) -> List[Tuple[str, bool, str]]:
    """Execute all five fundamental physics and chemistry health checks.

    Args:
        vehicle_params: Optional VehicleParams to test (defaults to preset).
        battery_params: Optional BatteryParams to test.
        thermal_params: Optional ThermalParams to test.

    Returns:
        List of tuples: [(name, passed, detail_message), ...]
    """
    return [
        check_wh_per_km_benchmark(vehicle_params, battery_params, thermal_params),
        check_consumption_monotonicity(vehicle_params),
        check_downhill_zero_regen_aux_power(vehicle_params),
        check_range_soc_monotonicity(battery_params),
        check_charging_temperature_window(battery_params),
    ]

