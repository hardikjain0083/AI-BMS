"""Smart charging algorithms with bell-shaped thermal derating and CC-CV taper.

Models temperature-dependent charge acceptance, CC-to-CV current tapering,
time-to-full estimates, and pack thermal rise during charging.
Pure functions with no external state.

NOTE ON APPROXIMATION:
The Constant Voltage (CV) taper phase is implemented as a linear current taper
approximation from taper_soc to 1.0, rather than a true dynamic differential
equation of electrochemical diffusion overpotential.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd

from ebms_sim.config import BatteryParams, ThermalParams, ChargingParams
from ebms_sim.battery import (
    ocv_from_soc,
    internal_resistance_ohm,
    usable_capacity_factor,
)
from ebms_sim.thermal import step_pack_temperature_c


@dataclass(frozen=True)
class ChargeSummary:
    """Summary of a simulated charging session.

    Attributes:
        time_to_full_s: Total seconds required to reach target SOC or termination [s].
        time_to_full_hours: Total charging duration in hours [h].
        energy_charged_wh: Total electrical energy supplied to battery [Wh].
        start_soc: Initial SOC [0..1].
        final_soc: Final reached SOC [0..1].
        peak_temp_c: Maximum pack temperature reached during charge [deg C].
        final_temp_c: Pack temperature at end of charge [deg C].
        aborted_high_temp: True if charging ceased due to pack overheating (>= max_temp).
        aborted_freezing: True if pack was too cold to charge (<= 0C).
    """
    time_to_full_s: float
    time_to_full_hours: float
    energy_charged_wh: float
    start_soc: float
    final_soc: float
    peak_temp_c: float
    final_temp_c: float
    aborted_high_temp: bool
    aborted_freezing: bool


@dataclass
class ChargeResult:
    """Complete charging simulation output."""
    time_series: pd.DataFrame
    summary: ChargeSummary


def temperature_charge_multiplier(
    temp_c: float,
    temp_min_c: float = 0.0,
    temp_ramp_low_c: float = 5.0,
    temp_opt_low_c: float = 15.0,
    temp_opt_high_c: float = 35.0,
    temp_max_c: float = 45.0,
) -> float:
    """Calculate the bell-shaped charge current thermal derating multiplier for lithium [0.0 .. 1.0].

    Specifications for Lithium / LFP:
    - Below temp_min_c (<= 0 deg C): 0.0 (strictly 0 A to prevent lithium plating).
    - 0 to 5 deg C: no fast charge, reduced trickle rate (0.0 to 0.20).
    - 5 to 15 deg C: reduced charging (0.20 ramping up to 1.00).
    - 15 to 35 deg C: optimal full charge window (multiplier = 1.00).
    - 35 to 45 deg C: derated / reduced charging (1.00 tapering down to 0.0 at 45 C).
    - Above temp_max_c (>= 45 deg C): 0.0 (thermal runaway safety shutoff).

    Args:
        temp_c: Cell/pack temperature in Celsius [deg C].
        temp_min_c: Zero-charge lower cutoff [deg C].
        temp_ramp_low_c: Transition threshold for fast charge [deg C].
        temp_opt_low_c: Start of optimal full-rate window [deg C].
        temp_opt_high_c: End of optimal full-rate window [deg C].
        temp_max_c: Zero-charge upper safety cutoff [deg C].

    Returns:
        Charge current scale factor between 0.0 and 1.0 [dimensionless].
    """
    if temp_c <= temp_min_c:
        return 0.0
    elif temp_c < temp_ramp_low_c:
        # Trickle region: 0.0 to 0.20
        ratio = (temp_c - temp_min_c) / (temp_ramp_low_c - temp_min_c)
        return 0.20 * ratio
    elif temp_c < temp_opt_low_c:
        # Ramping region: 0.20 to 1.00
        ratio = (temp_c - temp_ramp_low_c) / (temp_opt_low_c - temp_ramp_low_c)
        return 0.20 + 0.80 * ratio
    elif temp_c <= temp_opt_high_c:
        # Full optimal window
        return 1.00
    elif temp_c < temp_max_c:
        # Thermal derating region: 1.00 down to 0.0
        ratio = (temp_c - temp_opt_high_c) / (temp_max_c - temp_opt_high_c)
        return 1.00 - ratio
    else:
        # High-temperature shutoff
        return 0.0


def lead_acid_cell_charge_voltage_v(
    temp_c: float,
    phase: Literal["absorption", "float"] = "absorption",
    temp_coeff_v_cell: float = -0.003,
    absorption_ref_v_cell: float = 2.40,
    float_ref_v_cell: float = 2.25,
) -> float:
    """Calculate temperature-compensated lead-acid single-cell charge voltage.

    V_cell = V_ref + (-0.003) * (T - 25) per cell.
    - Absorption reference: 2.40 V/cell at 25C.
    - Float reference: 2.25 V/cell at 25C.
    - Negative temperature coefficient: -0.003 V/(deg C * cell) = -3 mV/C/cell.

    Args:
        temp_c: Cell temperature in Celsius [deg C].
        phase: Charging phase ('absorption' or 'float').
        temp_coeff_v_cell: Voltage temperature compensation coefficient [V/(deg C * cell)].

    Returns:
        Compensated cell target voltage [V/cell].
    """
    v_ref = absorption_ref_v_cell if phase == "absorption" else float_ref_v_cell
    return v_ref + temp_coeff_v_cell * (temp_c - 25.0)


def lead_acid_pack_charge_voltage_v(
    temp_c: float,
    phase: Literal["absorption", "float"] = "absorption",
    cells_count: int = 24,
    temp_coeff_v_cell: float = -0.003,
    absorption_ref_v_cell: float = 2.40,
    float_ref_v_cell: float = 2.25,
) -> float:
    """Calculate temperature-compensated 48V lead-acid total pack charge voltage across series cells (24 cells).

    Args:
        temp_c: Pack temperature in Celsius [deg C].
        phase: 'absorption' (2.40 V/cell ref) or 'float' (2.25 V/cell ref).
        cells_count: Number of series cells (24 for 48V nominal).
        temp_coeff_v_cell: Temperature coefficient per cell [-0.003 V/C/cell].

    Returns:
        Pack charging voltage in Volts [V].
    """
    return cells_count * lead_acid_cell_charge_voltage_v(
        temp_c,
        phase,
        temp_coeff_v_cell,
        absorption_ref_v_cell,
        float_ref_v_cell,
    )


def max_charge_current_a(
    temp_c: float,
    soc: float,
    chemistry: str = "lfp",
    nominal_capacity_ah: float = 100.0,
    max_charge_c_rate: float = 0.33,
    taper_soc: float = 0.80,
    min_charge_current_a: float = 1.5,
    lead_acid_temp_min_c: float = 0.0,
    lead_acid_temp_max_c: float = 50.0,
) -> float:
    """Calculate allowable charging current under thermal and CC-CV constraints.

    NOTE ON APPROXIMATION:
    The Constant Voltage (CV) taper phase is modeled as a linear current taper
    approximation from taper_soc to 1.0, rather than a true dynamic differential
    equation of electrochemical diffusion overpotential.

    Chemistry dispatch:
    - LFP: Uses piecewise bell curve (0 below 0C, reduced 5-15C, full 15-35C,
      reduced 35-45C, shutoff at >=45C).
    - Lead-Acid: Current limit of 0.10 - 0.20 C, zero below 0C and cutoff above 50C.
      (The lithium bell curve is NOT used for lead-acid).

    Args:
        temp_c: Pack temperature [deg C].
        soc: Current battery SOC [0.0 .. 1.0].
        chemistry: 'lead_acid' or 'lfp' (default 'lfp').
        nominal_capacity_ah: Nominal battery capacity [Ah].
        max_charge_c_rate: Maximum fast charge C-rate.
        taper_soc: SOC where CC transitions into linear CV taper approximation.
        min_charge_current_a: Minimum charging cutoff current [A].

    Returns:
        Allowable charging current in Amperes [A].
    """
    if soc >= 1.0:
        return 0.0

    if chemistry == "lead_acid":
        # Lead-Acid Charging Model
        # Cutoff above 50C and below 0C
        if temp_c <= lead_acid_temp_min_c or temp_c >= lead_acid_temp_max_c:
            return 0.0

        # Current limit: 0.10 - 0.20 C
        pb_c_rate = min(0.20, max(0.10, max_charge_c_rate if max_charge_c_rate < 0.30 else 0.15))
        base_current = nominal_capacity_ah * pb_c_rate

        # Mild cold reduction between 0 and 10C
        if temp_c < 10.0:
            base_current *= (0.50 + 0.05 * temp_c)

        # CC phase
        if soc <= taper_soc:
            return max(0.0, base_current)

        # Linear CV taper approximation
        soc_ratio = (1.0 - soc) / max(1e-4, 1.0 - taper_soc)
        tapered = min_charge_current_a + (base_current - min_charge_current_a) * soc_ratio
        return max(0.0, tapered)

    else:
        # Lithium (LFP) Charging with Bell-Curve
        t_mult = temperature_charge_multiplier(temp_c)
        if t_mult <= 0.0:
            return 0.0

        base_max_current = nominal_capacity_ah * max_charge_c_rate
        allowed_cc_current = base_max_current * t_mult

        # CC phase
        if soc <= taper_soc:
            return max(0.0, allowed_cc_current)

        # Linear CV taper approximation
        soc_ratio = (1.0 - soc) / max(1e-4, 1.0 - taper_soc)
        tapered_current = min_charge_current_a + (allowed_cc_current - min_charge_current_a) * soc_ratio
        return max(0.0, tapered_current)


def simulate_charge(
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    charging_params: ChargingParams,
    start_soc: float = 0.20,
    target_soc: float = 1.00,
    start_temp_c: Optional[float] = None,
    t_ambient_c: float = 25.0,
    dt_s: float = 30.0,
    max_charge_duration_s: float = 36000.0,  # 10 hours safety cap
) -> ChargeResult:
    """Simulate a complete battery charging session from start_soc to target_soc.

    Uses chemistry-specific charge current limits, temperature compensation,
    and linear CV taper approximation.

    Args:
        battery_params: Battery pack parameters.
        thermal_params: Pack thermal parameters.
        charging_params: Charging limits and temperature profile.
        start_soc: Starting SOC [0.0 .. 1.0].
        target_soc: Desired terminal SOC [0.0 .. 1.0].
        start_temp_c: Starting pack temperature [deg C] (defaults to t_ambient_c).
        t_ambient_c: Ambient air temperature [deg C].
        dt_s: Simulation time step in seconds [s].
        max_charge_duration_s: Timeout duration in seconds.

    Returns:
        ChargeResult with time series DataFrame and ChargeSummary.
    """
    chem = battery_params.chemistry
    current_soc = max(0.0, min(1.0, start_soc))
    target_soc = max(current_soc, min(1.0, target_soc))
    current_temp = start_temp_c if start_temp_c is not None else t_ambient_c

    # Temperature cutoff limits
    temp_max_limit = (
        charging_params.lead_acid_temp_max_c
        if chem == "lead_acid"
        else charging_params.temp_max_c
    )
    temp_min_limit = charging_params.temp_min_c

    times = []
    socs = []
    currents = []
    voltages = []
    temps = []
    energies_wh = []

    elapsed_s = 0.0
    total_energy_wh = 0.0
    aborted_high_temp = False
    aborted_freezing = current_temp <= temp_min_limit
    peak_temp = current_temp

    coulombic_efficiency = 0.98 if chem == "lfp" else 0.90

    while elapsed_s < max_charge_duration_s and current_soc < target_soc:
        # Check temperature safety
        if current_temp >= temp_max_limit:
            aborted_high_temp = True
            break
        if current_temp <= temp_min_limit:
            aborted_freezing = True
            break

        # Calculate allowable charging current dispatched on chemistry
        i_charge = max_charge_current_a(
            temp_c=current_temp,
            soc=current_soc,
            chemistry=chem,
            nominal_capacity_ah=battery_params.nominal_capacity_ah,
            max_charge_c_rate=charging_params.max_charge_c_rate,
            taper_soc=charging_params.taper_soc,
            min_charge_current_a=charging_params.min_charge_current_a,
            lead_acid_temp_min_c=charging_params.temp_min_c,
            lead_acid_temp_max_c=charging_params.lead_acid_temp_max_c,
        )

        if i_charge < charging_params.min_charge_current_a * 0.5:
            # Current tapered below cutoff threshold; charging complete
            break

        # Terminal voltage during charging
        r_int = internal_resistance_ohm(
            battery_params.r0_ohm, current_temp, soh=1.0,
            temp_coeff_r=battery_params.temp_coeff_r,
            soh_coeff_r=battery_params.soh_coeff_r,
        )
        ocv = ocv_from_soc(current_soc, chem)
        v_term = ocv + i_charge * r_int

        # For lead-acid, cap at temperature-compensated absorption voltage
        if chem == "lead_acid":
            v_abs_max = lead_acid_pack_charge_voltage_v(
                temp_c=current_temp,
                phase="absorption",
                cells_count=charging_params.lead_acid_cells_count,
                temp_coeff_v_cell=charging_params.lead_acid_temp_coeff_v_cell,
                absorption_ref_v_cell=charging_params.lead_acid_v_abs_ref_v_cell,
                float_ref_v_cell=charging_params.lead_acid_v_float_ref_v_cell,
            )
            v_term = min(v_term, v_abs_max)

        # Energy delivered in this step: V * I * dt / 3600
        step_wh = (v_term * i_charge * dt_s) / 3600.0
        total_energy_wh += step_wh

        # Record point
        times.append(elapsed_s)
        socs.append(current_soc)
        currents.append(i_charge)
        voltages.append(v_term)
        temps.append(current_temp)
        energies_wh.append(total_energy_wh)

        # Update SOC (Ah delivered * efficiency)
        cap_factor = usable_capacity_factor(current_temp)
        usable_ah = battery_params.nominal_capacity_ah * cap_factor
        delta_ah = (i_charge * coulombic_efficiency * dt_s) / 3600.0
        current_soc = min(target_soc, current_soc + delta_ah / usable_ah)

        # Step thermal model (charging heats pack via I^2 * R)
        current_temp = step_pack_temperature_c(
            t_pack_c=current_temp,
            current_a=-i_charge,  # negative indicates charge
            r_ohm=r_int,
            t_ambient_c=t_ambient_c,
            c_th_j_per_k=thermal_params.c_th_j_per_k,
            r_th_k_per_w=thermal_params.r_th_k_per_w,
            dt_s=dt_s,
            v_ms=0.0,  # vehicle is stationary while charging
        )
        peak_temp = max(peak_temp, current_temp)
        elapsed_s += dt_s

    # Append terminal row if recorded
    if times:
        times.append(elapsed_s)
        socs.append(current_soc)
        currents.append(0.0)
        voltages.append(ocv_from_soc(current_soc, chem))
        temps.append(current_temp)
        energies_wh.append(total_energy_wh)

    df_ts = pd.DataFrame({
        "time_s": times,
        "time_hours": [t / 3600.0 for t in times],
        "soc": socs,
        "charge_current_a": currents,
        "voltage_v": voltages,
        "temp_c": temps,
        "energy_charged_wh": energies_wh,
    })

    summary = ChargeSummary(
        time_to_full_s=float(elapsed_s),
        time_to_full_hours=float(elapsed_s / 3600.0),
        energy_charged_wh=float(total_energy_wh),
        start_soc=float(start_soc),
        final_soc=float(current_soc),
        peak_temp_c=float(peak_temp),
        final_temp_c=float(current_temp),
        aborted_high_temp=bool(aborted_high_temp),
        aborted_freezing=bool(aborted_freezing),
    )

    return ChargeResult(time_series=df_ts, summary=summary)
