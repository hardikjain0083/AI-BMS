"""Battery equivalent circuit model, OCV-SOC, Peukert law, and thermal derating.

Pure mathematical functions for traction battery packs (Lead-Acid & LFP).
Designed for zero-dependency execution and easy porting to embedded C/C++.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict
from ebms_sim.config import BatteryParams

# Tabulated OCV vs SOC points (SOC in [0..1])
# Lead-Acid (nominal 48V, 24 cells, 1.75V - 2.13V per cell)
LEAD_ACID_SOC_TABLE = (
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
)
LEAD_ACID_OCV_TABLE = (
    42.00, 44.40, 45.60, 46.50, 47.20, 47.80, 48.30, 48.80, 49.30, 49.80, 50.40, 51.20
)

# LFP (nominal 51.2V, 16S, 2.60V - 3.40V per cell)
LFP_SOC_TABLE = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00
)
LFP_OCV_TABLE = (
    41.60, 48.00, 50.40, 51.50, 52.00, 52.60, 52.70, 52.80, 52.90, 53.00, 53.10, 53.30, 53.60, 54.40
)


def _interpolate_1d(x: float, xp: tuple, fp: tuple) -> float:
    """Pure 1D linear interpolation with boundary clamping.

    Args:
        x: Query point.
        xp: Monotonically increasing x-coordinates.
        fp: Function values at xp.

    Returns:
        Interpolated value.
    """
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]

    # Binary search or linear search (tables have <= 15 elements)
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            dx = xp[i + 1] - xp[i]
            if dx <= 0.0:
                return fp[i]
            t = (x - xp[i]) / dx
            return fp[i] + t * (fp[i + 1] - fp[i])

    return fp[-1]


def ocv_from_soc(soc: float, chemistry: str = "lead_acid") -> float:
    """Look up open-circuit voltage (OCV) from State of Charge.

    Args:
        soc: State of charge fraction [0.0 .. 1.0].
        chemistry: Battery chemistry ('lead_acid' or 'lfp').

    Returns:
        Pack Open Circuit Voltage in Volts [V].
    """
    clamped_soc = max(0.0, min(1.0, soc))
    if chemistry == "lfp":
        return _interpolate_1d(clamped_soc, LFP_SOC_TABLE, LFP_OCV_TABLE)
    return _interpolate_1d(clamped_soc, LEAD_ACID_SOC_TABLE, LEAD_ACID_OCV_TABLE)


def usable_capacity_factor(temp_c: float) -> float:
    """Calculate temperature-dependent usable capacity derating factor.

    Target behavior:
        At 25 deg C: factor is 1.0.
        At  0 deg C: factor is ~0.80.
        Above 25 C: capped at 1.02.
        Below 0 C: drops smoothly down to ~0.55 at -20 C.

    Args:
        temp_c: Cell/pack temperature in Celsius [deg C].

    Returns:
        Usable capacity fraction relative to nominal capacity at 25C [dimensionless].
    """
    if temp_c >= 25.0:
        return min(1.02, 1.0 + 0.0008 * (temp_c - 25.0))
    elif temp_c >= 0.0:
        # Linear ramp from 0.80 at 0 C to 1.00 at 25 C
        return 0.80 + 0.20 * (temp_c / 25.0)
    else:
        # Sub-zero degradation: 0.80 down to 0.55 at -20 C
        ratio = max(0.0, min(1.0, (temp_c + 20.0) / 20.0))
        return 0.55 + 0.25 * ratio


def internal_resistance_ohm(
    r0_base_ohm: float,
    temp_c: float,
    soh: float,
    temp_coeff_r: float = 0.015,
    soh_coeff_r: float = 0.80,
) -> float:
    """Calculate internal resistance R0 as a function of temperature and SOH.

    R(T, SOH) = R0_base * (1 + temp_coeff * (25 - T)) * (1 + soh_coeff * (1 - SOH))

    Args:
        r0_base_ohm: Base internal resistance at 25C and SOH 1.0 [Ohm].
        temp_c: Cell temperature in Celsius [deg C].
        soh: State of Health fraction [0.0 .. 1.0].
        temp_coeff_r: Temperature coefficient [1/deg C].
        soh_coeff_r: SOH degradation coefficient [dimensionless].

    Returns:
        Internal resistance in Ohms [Ohm].
    """
    temp_factor = max(0.5, 1.0 + temp_coeff_r * (25.0 - temp_c))
    clamped_soh = max(0.2, min(1.0, soh))
    soh_factor = 1.0 + soh_coeff_r * (1.0 - clamped_soh)
    return max(0.005, r0_base_ohm * temp_factor * soh_factor)


def peukert_effective_current_a(
    current_a: float,
    peukert_exp: float,
    peukert_i_ref_a: float,
) -> float:
    """Calculate effective discharge current accounting for Peukert's law.

    For lead-acid:
        I_eff = I * (max(I, I_ref) / I_ref)^(k - 1) when I > 0
    For LFP:
        k = 1.0 => I_eff = I

    Args:
        current_a: Battery discharge current [A] (positive = discharge).
        peukert_exp: Peukert exponent k [dimensionless].
        peukert_i_ref_a: Reference rated discharge current (e.g. C20 rate) [A].

    Returns:
        Effective current in Amperes [A].
    """
    if current_a <= 0.0:
        return current_a
    if abs(peukert_exp - 1.0) < 1e-4:
        return current_a

    ratio = max(1.0, current_a / max(0.1, peukert_i_ref_a))
    return current_a * (ratio ** (peukert_exp - 1.0))


def terminal_voltage_v(ocv_v: float, current_a: float, r_ohm: float) -> float:
    """Calculate terminal voltage using equivalent circuit: V = OCV - I * R.

    Args:
        ocv_v: Open circuit voltage [V].
        current_a: Current drawn from battery [A] (positive for discharge).
        r_ohm: Total pack internal resistance [Ohm].

    Returns:
        Terminal voltage in Volts [V].
    """
    return ocv_v - current_a * r_ohm


def solve_current_and_voltage(
    power_w: float,
    ocv_v: float,
    r_ohm: float,
) -> Tuple[float, float]:
    """Solve for current and voltage given electrical power demand and battery state.

    From P = V * I = (OCV - I * R) * I => R * I^2 - OCV * I + P = 0.
    I = (OCV - sqrt(OCV^2 - 4 * R * P)) / (2 * R).

    Args:
        power_w: Battery electrical power demand [W] (positive = discharge).
        ocv_v: Open circuit voltage [V].
        r_ohm: Internal resistance [Ohm].

    Returns:
        Tuple of (current_a, terminal_voltage_v).
    """
    if abs(power_w) < 1e-5:
        return 0.0, ocv_v

    if r_ohm <= 1e-6:
        curr = power_w / max(1.0, ocv_v)
        return curr, ocv_v

    discriminant = ocv_v * ocv_v - 4.0 * r_ohm * power_w
    if discriminant < 0.0:
        # Power demand exceeds maximum power transfer limit (P_max = OCV^2 / (4*R))
        # Clamp to max power point current
        curr = ocv_v / (2.0 * r_ohm)
        volt = ocv_v - curr * r_ohm
        return curr, volt

    curr = (ocv_v - math.sqrt(discriminant)) / (2.0 * r_ohm)
    volt = ocv_v - curr * r_ohm
    return curr, volt


def update_soc_coulomb_counting(
    soc: float,
    current_a: float,
    dt_s: float,
    nominal_capacity_ah: float,
    temp_c: float,
    peukert_exp: float = 1.0,
    peukert_i_ref_a: float = 5.0,
    charge_coulombic_efficiency: float = 0.98,
) -> float:
    """Step State of Charge (SOC) using temperature-derated coulomb counting and Peukert law.

    Args:
        soc: Current state of charge [0.0 .. 1.0].
        current_a: Current during step [A] (positive = discharge, negative = charge).
        dt_s: Time step duration [s].
        nominal_capacity_ah: Rated capacity at 25C [Ah].
        temp_c: Current cell temperature [deg C].
        peukert_exp: Peukert exponent k [dimensionless].
        peukert_i_ref_a: Reference current for Peukert [A].
        charge_coulombic_efficiency: Coulombic efficiency when charging [0..1].

    Returns:
        Updated state of charge [0.0 .. 1.0].
    """
    cap_factor = usable_capacity_factor(temp_c)
    usable_ah = max(1.0, nominal_capacity_ah * cap_factor)

    if current_a >= 0.0:
        # Discharge with Peukert effect
        i_eff = peukert_effective_current_a(current_a, peukert_exp, peukert_i_ref_a)
        delta_ah = (i_eff * dt_s) / 3600.0
    else:
        # Charging: current is negative, apply charge efficiency
        delta_ah = (current_a * charge_coulombic_efficiency * dt_s) / 3600.0

    new_soc = soc - (delta_ah / usable_ah)
    return max(0.0, min(1.0, new_soc))

