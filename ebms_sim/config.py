"""Configuration and parameter dataclasses for ebms-sim.

All units are explicitly documented in variable names and docstrings.
Pure dataclasses with no side-effects or I/O.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Dict, Any


@dataclass(frozen=True)
class VehicleParams:
    """Physics parameters for an Indian electric rickshaw (L5M category).

    Attributes:
        curb_mass_kg: Unladen mass of vehicle without driver/passengers [kg].
        driver_mass_kg: Mass of driver [kg].
        crr: Rolling resistance coefficient of tires on road [dimensionless].
        cda_m2: Aerodynamic drag area (Cd * frontal_area_m2) [m^2].
        eta_drive: Mechanical + motor + inverter efficiency during propulsion [0..1].
        eta_regen: Efficiency of regenerative braking (default 0.0) [0..1].
        aux_w: Continuous auxiliary electrical power consumption (lights, horn, dash) [W].
        rotational_inertia_factor: Inertia factor for rotating drivetrain/wheels [dimensionless].
        max_speed_kmh: Legal / mechanical speed cap for Indian e-rickshaws [km/h].
    """
    curb_mass_kg: float = 340.0
    driver_mass_kg: float = 65.0
    crr: float = 0.018
    cda_m2: float = 0.85
    eta_drive: float = 0.82
    eta_regen: float = 0.0
    aux_w: float = 50.0
    rotational_inertia_factor: float = 0.05
    max_speed_kmh: float = 25.0
    eta_model: Literal["constant", "loss_based"] = "constant"
    loss_p0_w: float = 45.0
    loss_c1: float = 0.11
    loss_c2: float = 6.5e-5


@dataclass(frozen=True)
class BatteryParams:
    """Parameters for the traction battery pack.

    Attributes:
        chemistry: Battery chemistry type ("lead_acid" or "lfp").
        nominal_voltage_v: Nominal pack voltage [V].
        nominal_capacity_ah: Nominal capacity at C20 rating [Ah].
        nominal_energy_wh: Nominal pack energy capacity [Wh].
        r0_ohm: Base internal resistance at 25 deg C and SOH 1.0 [Ohm].
        peukert_exp: Peukert exponent k (lead-acid ~1.15-1.25, LFP = 1.0) [dimensionless].
        peukert_i_ref_a: Reference discharge current for Peukert law [A].
        temp_coeff_r: Resistance temperature coefficient [1/deg C].
        soh_coeff_r: Resistance degradation coefficient per unit SOH drop [dimensionless].
        cutoff_soc: Low-SOC cutoff threshold below which pack is depleted [0..1].
        max_continuous_current_a: Maximum continuous discharge current limit [A].
        pack_mass_kg: Mass of battery pack [kg].
    """
    chemistry: Literal["lead_acid", "lfp"] = "lead_acid"
    nominal_voltage_v: float = 48.0
    nominal_capacity_ah: float = 100.0
    nominal_energy_wh: float = 4800.0
    r0_ohm: float = 0.060
    peukert_exp: float = 1.18
    peukert_i_ref_a: float = 5.0
    temp_coeff_r: float = 0.015
    soh_coeff_r: float = 0.8
    cutoff_soc: float = 0.10
    max_continuous_current_a: float = 100.0
    pack_mass_kg: float = 120.0


@dataclass(frozen=True)
class ThermalParams:
    """Lumped pack thermal parameters.

    Attributes:
        c_th_j_per_k: Pack thermal capacitance (mass * specific heat) [J/K].
        r_th_k_per_w: Lumped thermal resistance to ambient [K/W].
        cooling_air_velocity_factor: Multiplier for convective heat dissipation per m/s vehicle speed.
    """
    c_th_j_per_k: float = 90000.0  # ~100 kg * ~900 J/(kg*K)
    r_th_k_per_w: float = 0.65     # Natural/forced convection boundary
    cooling_air_velocity_factor: float = 0.02


@dataclass(frozen=True)
class ChargingParams:
    """Smart charging profile and constraints.

    Attributes:
        max_charge_c_rate: Maximum fast charge C-rate (e.g. 0.15C for lead-acid, 0.5C-1.0C for LFP).
        taper_soc: SOC at which CC transitions into linear CV taper approximation [0..1].
        min_charge_current_a: Cutoff current indicating end of CV charge [A].
        temp_min_c: Temperature below which charging current is 0.0 [deg C].
        temp_ramp_low_c: Temperature at which charging begins ramping from low [deg C].
        temp_opt_low_c: Lower bound of optimal full-rate temperature window [deg C].
        temp_opt_high_c: Upper bound of optimal full-rate temperature window [deg C].
        temp_max_c: Temperature above which charging current is 0.0 (shutoff) [deg C].
        lead_acid_v_abs_ref_v_cell: Reference absorption charge voltage at 25C per cell [V/cell].
        lead_acid_v_float_ref_v_cell: Reference float charge voltage at 25C per cell [V/cell].
        lead_acid_temp_coeff_v_cell: Temperature compensation coefficient [V/(deg C * cell)].
        lead_acid_cells_count: Number of series cells in lead-acid pack (24 for 48V).
        lead_acid_temp_max_c: High-temperature cutoff limit for lead-acid charging [deg C].
    """
    max_charge_c_rate: float = 0.33
    taper_soc: float = 0.80
    min_charge_current_a: float = 1.5
    temp_min_c: float = 0.0
    temp_ramp_low_c: float = 5.0
    temp_opt_low_c: float = 15.0
    temp_opt_high_c: float = 35.0
    temp_max_c: float = 45.0
    lead_acid_v_abs_ref_v_cell: float = 2.40
    lead_acid_v_float_ref_v_cell: float = 2.25
    lead_acid_temp_coeff_v_cell: float = -0.003
    lead_acid_cells_count: int = 24
    lead_acid_temp_max_c: float = 50.0


# Standard Presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "lead_acid_48V_100Ah": {
        "vehicle": VehicleParams(
            curb_mass_kg=350.0,
            driver_mass_kg=65.0,
            crr=0.018,
            cda_m2=0.85,
            eta_drive=0.82,
            eta_regen=0.0,
            aux_w=50.0,
            rotational_inertia_factor=0.05,
            max_speed_kmh=25.0,
        ),
        "battery": BatteryParams(
            chemistry="lead_acid",
            nominal_voltage_v=48.0,
            nominal_capacity_ah=100.0,
            nominal_energy_wh=4800.0,
            r0_ohm=0.065,
            peukert_exp=1.20,
            peukert_i_ref_a=5.0,
            temp_coeff_r=0.015,
            soh_coeff_r=0.85,
            cutoff_soc=0.10,
            max_continuous_current_a=100.0,
            pack_mass_kg=130.0,
        ),
        "thermal": ThermalParams(
            c_th_j_per_k=110000.0,
            r_th_k_per_w=0.60,
            cooling_air_velocity_factor=0.02,
        ),
        "charging": ChargingParams(
            max_charge_c_rate=0.25,
            taper_soc=0.80,
            min_charge_current_a=2.0,
            temp_min_c=0.0,
            temp_ramp_low_c=5.0,
            temp_opt_low_c=15.0,
            temp_opt_high_c=35.0,
            temp_max_c=45.0,
        ),
    },
    "lfp_51V_60Ah": {
        "vehicle": VehicleParams(
            curb_mass_kg=280.0,
            driver_mass_kg=65.0,
            crr=0.018,
            cda_m2=0.85,
            eta_drive=0.85,
            eta_regen=0.0,
            aux_w=45.0,
            rotational_inertia_factor=0.05,
            max_speed_kmh=25.0,
        ),
        "battery": BatteryParams(
            chemistry="lfp",
            nominal_voltage_v=51.2,
            nominal_capacity_ah=60.0,
            nominal_energy_wh=3072.0,
            r0_ohm=0.025,
            peukert_exp=1.00,  # Negligible Peukert for LFP
            peukert_i_ref_a=3.0,
            temp_coeff_r=0.012,
            soh_coeff_r=0.70,
            cutoff_soc=0.05,
            max_continuous_current_a=120.0,
            pack_mass_kg=35.0,
        ),
        "thermal": ThermalParams(
            c_th_j_per_k=45000.0,
            r_th_k_per_w=0.75,
            cooling_air_velocity_factor=0.025,
        ),
        "charging": ChargingParams(
            max_charge_c_rate=0.50,
            taper_soc=0.90,
            min_charge_current_a=1.5,
            temp_min_c=0.0,
            temp_ramp_low_c=5.0,
            temp_opt_low_c=15.0,
            temp_opt_high_c=35.0,
            temp_max_c=45.0,
        ),
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Retrieve deep copy or instantiation of predefined preset by name.

    Args:
        name: Either 'lead_acid_48V_100Ah' or 'lfp_51V_60Ah'.

    Returns:
        Dict with keys: 'vehicle', 'battery', 'thermal', 'charging'.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]

