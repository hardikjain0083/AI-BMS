"""Vehicle longitudinal dynamics and tractive power calculations for e-rickshaws.

Pure mathematical functions with no I/O or global state.
All units are explicitly defined in arguments and docstrings.
"""

from __future__ import annotations
import math
from typing import Dict
from ebms_sim.config import VehicleParams

# Standard gravitational acceleration [m/s^2]
GRAVITY_MS2 = 9.80665

# Standard dry air density at 20 deg C, 101.325 kPa [kg/m^3]
DEFAULT_AIR_DENSITY_KG_M3 = 1.2041


def air_density_from_temp(temp_c: float, pressure_pa: float = 101325.0) -> float:
    """Calculate ambient dry air density using the ideal gas law.

    Args:
        temp_c: Ambient temperature in Celsius [deg C].
        pressure_pa: Atmospheric pressure in Pascals [Pa].

    Returns:
        Air density in kg/m^3.
    """
    temp_k = temp_c + 273.15
    gas_constant_air = 287.058  # J/(kg*K)
    return pressure_pa / (gas_constant_air * temp_k)


def rolling_resistance_force_n(
    mass_kg: float,
    crr: float,
    grade_fraction: float = 0.0,
) -> float:
    """Calculate rolling resistance force magnitude.

    F_roll = m * g * Crr * cos(theta), where theta = arctan(grade).

    Args:
        mass_kg: Total vehicle + passenger + payload mass [kg].
        crr: Coefficient of rolling resistance [dimensionless].
        grade_fraction: Road gradient (e.g. 0.05 for 5% slope) [m rise / m run].

    Returns:
        Rolling resistance force in Newtons [N].
    """
    cos_theta = 1.0 / math.sqrt(1.0 + grade_fraction * grade_fraction)
    return mass_kg * GRAVITY_MS2 * crr * cos_theta


def aerodynamic_drag_force_n(
    v_ms: float,
    headwind_ms: float,
    cda_m2: float,
    air_density_kg_m3: float = DEFAULT_AIR_DENSITY_KG_M3,
) -> float:
    """Calculate aerodynamic drag force.

    F_aero = 0.5 * rho * CdA * (v + v_wind)^2 * sgn(v + v_wind)

    Args:
        v_ms: Vehicle forward speed [m/s].
        headwind_ms: Headwind speed component (positive is headwind, negative tailwind) [m/s].
        cda_m2: Aerodynamic drag area (Cd * A) [m^2].
        air_density_kg_m3: Air density [kg/m^3].

    Returns:
        Aerodynamic drag force in Newtons [N].
    """
    v_rel = v_ms + headwind_ms
    direction = 1.0 if v_rel >= 0.0 else -1.0
    return 0.5 * air_density_kg_m3 * cda_m2 * (v_rel * v_rel) * direction


def grade_force_n(
    mass_kg: float,
    grade_fraction: float,
) -> float:
    """Calculate gravitational force due to road slope.

    F_grade = m * g * sin(theta) = m * g * (grade / sqrt(1 + grade^2))

    Args:
        mass_kg: Total vehicle + passenger + payload mass [kg].
        grade_fraction: Road gradient (positive for uphill, negative for downhill).

    Returns:
        Grade resistance force in Newtons [N] (positive = uphill opposes motion).
    """
    sin_theta = grade_fraction / math.sqrt(1.0 + grade_fraction * grade_fraction)
    return mass_kg * GRAVITY_MS2 * sin_theta


def inertial_force_n(
    mass_kg: float,
    acceleration_ms2: float,
    rotational_inertia_factor: float = 0.05,
) -> float:
    """Calculate acceleration inertial force including drivetrain rotational inertia.

    F_inertia = m * (1 + lambda_rot) * a

    Args:
        mass_kg: Total vehicle + passenger + payload mass [kg].
        acceleration_ms2: Longitudinal acceleration [m/s^2].
        rotational_inertia_factor: Rotational inertia mass factor [dimensionless].

    Returns:
        Inertial force in Newtons [N].
    """
    return mass_kg * (1.0 + rotational_inertia_factor) * acceleration_ms2


def tractive_wheel_force_n(
    mass_kg: float,
    v_ms: float,
    acceleration_ms2: float,
    grade_fraction: float,
    headwind_ms: float,
    crr: float,
    cda_m2: float,
    rotational_inertia_factor: float = 0.05,
    air_density_kg_m3: float = DEFAULT_AIR_DENSITY_KG_M3,
) -> float:
    """Calculate net tractive force required at wheels.

    F_wheel = F_roll + F_aero + F_grade + F_inertia

    Args:
        mass_kg: Total vehicle + passenger + payload mass [kg].
        v_ms: Vehicle speed [m/s].
        acceleration_ms2: Vehicle acceleration [m/s^2].
        grade_fraction: Road slope [m/m].
        headwind_ms: Wind speed [m/s].
        crr: Rolling resistance coefficient [dimensionless].
        cda_m2: Aerodynamic drag area [m^2].
        rotational_inertia_factor: Rotational inertia allowance [dimensionless].
        air_density_kg_m3: Air density [kg/m^3].

    Returns:
        Total tractive force in Newtons [N].
    """
    # When stationary, rolling friction does not propel the vehicle backward
    if abs(v_ms) < 1e-5:
        f_roll = 0.0
    else:
        sgn_v = 1.0 if v_ms > 0.0 else -1.0
        f_roll = sgn_v * rolling_resistance_force_n(mass_kg, crr, grade_fraction)

    f_aero = aerodynamic_drag_force_n(v_ms, headwind_ms, cda_m2, air_density_kg_m3)
    f_grade = grade_force_n(mass_kg, grade_fraction)
    f_inertia = inertial_force_n(mass_kg, acceleration_ms2, rotational_inertia_factor)

    return f_roll + f_aero + f_grade + f_inertia


def wheel_power_w(total_force_n: float, v_ms: float) -> float:
    """Calculate mechanical power delivered at the wheels.

    P_wheel = F_wheel * v

    Args:
        total_force_n: Total longitudinal tractive force [N].
        v_ms: Forward speed [m/s].

    Returns:
        Wheel power in Watts [W] (positive = traction, negative = braking/regen).
    """
    return total_force_n * v_ms


def drivetrain_loss_w(
    wheel_pwr_w: float,
    p0_w: float = 45.0,
    c1: float = 0.11,
    c2: float = 6.5e-5,
) -> float:
    """Calculate mechanical + electrical drivetrain loss using quadratic model.

    P_loss = p0_w + c1 * P_wheel + c2 * P_wheel^2

    Args:
        wheel_pwr_w: Positive mechanical wheel power [W].
        p0_w: Fixed mechanical/electrical tare loss [W].
        c1: Linear loss coefficient [dimensionless].
        c2: Quadratic copper/ohmic loss coefficient [1/W].

    Returns:
        Drivetrain internal power loss in Watts [W].
    """
    if wheel_pwr_w <= 0.0:
        return 0.0
    return p0_w + (c1 * wheel_pwr_w) + (c2 * wheel_pwr_w * wheel_pwr_w)


def battery_power_w(
    wheel_pwr_w: float,
    eta_drive: float,
    eta_regen: float,
    aux_w: float,
    eta_model: str = "constant",
    p0_w: float = 45.0,
    c1: float = 0.11,
    c2: float = 6.5e-5,
) -> float:
    """Calculate total electrical power drawn from (or supplied to) battery.

    If P_wheel >= 0 (propulsion):
        If eta_model == 'loss_based':
            P_batt = P_wheel + P_loss(P_wheel) + aux_w
        Else:
            P_batt = P_wheel / eta_drive + aux_w
    If P_wheel < 0 (braking / descent):
        P_batt = P_wheel * eta_regen + aux_w

    Args:
        wheel_pwr_w: Mechanical wheel power in Watts [W].
        eta_drive: Drive efficiency (inverter * motor * gearbox) [0..1].
        eta_regen: Regenerative braking recovery efficiency [0..1]. Default is 0.0.
        aux_w: Auxiliary system continuous load in Watts [W].
        eta_model: Efficiency model ('constant' or 'loss_based'). Default 'constant'.
        p0_w: Fixed loss term for loss_based model [W].
        c1: Linear loss coefficient for loss_based model.
        c2: Quadratic loss coefficient for loss_based model [1/W].

    Returns:
        Total battery electrical power in Watts [W] (positive = discharge, negative = charge).
    """
    if wheel_pwr_w >= 0.0:
        if eta_model == "loss_based":
            loss = drivetrain_loss_w(wheel_pwr_w, p0_w, c1, c2)
            return wheel_pwr_w + loss + aux_w
        return (wheel_pwr_w / eta_drive) + aux_w
    else:
        return (wheel_pwr_w * eta_regen) + aux_w


def step_vehicle_physics(
    v_curr_ms: float,
    acceleration_ms2: float,
    grade_fraction: float,
    headwind_ms: float,
    payload_mass_kg: float,
    vehicle_params: VehicleParams,
    air_density_kg_m3: float = DEFAULT_AIR_DENSITY_KG_M3,
) -> Dict[str, float]:
    """Pure single-step calculation of all vehicle dynamics quantities.

    Args:
        v_curr_ms: Current speed [m/s].
        acceleration_ms2: Acceleration during current step [m/s^2].
        grade_fraction: Slope gradient [-0.12 .. 0.12].
        headwind_ms: Headwind component [m/s].
        payload_mass_kg: Total passengers and luggage mass [kg].
        vehicle_params: Vehicle configuration dataclass.
        air_density_kg_m3: Air density [kg/m^3].

    Returns:
        Dict containing total_mass_kg, f_roll_n, f_aero_n, f_grade_n,
        f_inertia_n, f_total_n, wheel_power_w, and battery_power_w.
    """
    total_mass_kg = vehicle_params.curb_mass_kg + vehicle_params.driver_mass_kg + payload_mass_kg

    f_roll = (
        rolling_resistance_force_n(total_mass_kg, vehicle_params.crr, grade_fraction)
        if v_curr_ms > 1e-5
        else 0.0
    )
    f_aero = aerodynamic_drag_force_n(
        v_curr_ms, headwind_ms, vehicle_params.cda_m2, air_density_kg_m3
    )
    f_grade = grade_force_n(total_mass_kg, grade_fraction)
    f_inertia = inertial_force_n(
        total_mass_kg, acceleration_ms2, vehicle_params.rotational_inertia_factor
    )
    f_total = f_roll + f_aero + f_grade + f_inertia

    p_wheel = wheel_power_w(f_total, v_curr_ms)
    p_batt = battery_power_w(
        p_wheel,
        vehicle_params.eta_drive,
        vehicle_params.eta_regen,
        vehicle_params.aux_w,
        eta_model=vehicle_params.eta_model,
        p0_w=vehicle_params.loss_p0_w,
        c1=vehicle_params.loss_c1,
        c2=vehicle_params.loss_c2,
    )

    return {
        "total_mass_kg": total_mass_kg,
        "f_roll_n": f_roll,
        "f_aero_n": f_aero,
        "f_grade_n": f_grade,
        "f_inertia_n": f_inertia,
        "f_total_n": f_total,
        "wheel_power_w": p_wheel,
        "battery_power_w": p_batt,
    }

