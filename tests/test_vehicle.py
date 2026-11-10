"""Unit tests for vehicle dynamics module (ebms_sim.vehicle)."""

import pytest
from ebms_sim.config import VehicleParams, PRESETS
from ebms_sim.vehicle import (
    rolling_resistance_force_n,
    aerodynamic_drag_force_n,
    grade_force_n,
    inertial_force_n,
    tractive_wheel_force_n,
    wheel_power_w,
    battery_power_w,
    step_vehicle_physics,
    air_density_from_temp,
)


def test_air_density_temperature_variation():
    """Verify ideal gas law air density decreases with temperature."""
    rho_0 = air_density_from_temp(0.0)
    rho_25 = air_density_from_temp(25.0)
    rho_45 = air_density_from_temp(45.0)

    assert 1.25 < rho_0 < 1.35
    assert 1.15 < rho_25 < 1.22
    assert 1.08 < rho_45 < 1.15
    assert rho_0 > rho_25 > rho_45


def test_rolling_resistance_flat_and_sloped():
    """Verify rolling resistance force on flat vs sloped surface."""
    mass = 500.0  # kg
    crr = 0.018
    f_flat = rolling_resistance_force_n(mass, crr, grade_fraction=0.0)
    f_slope = rolling_resistance_force_n(mass, crr, grade_fraction=0.10)

    expected_flat = 500.0 * 9.80665 * 0.018
    assert pytest.approx(f_flat, rel=1e-4) == expected_flat
    # On slope, normal force reduces slightly by cos(theta)
    assert f_slope < f_flat
    assert f_slope > 0.99 * f_flat


def test_aerodynamic_drag_direction():
    """Verify aerodynamic drag opposes motion and increases with headwind."""
    cda = 0.85
    f_still = aerodynamic_drag_force_n(v_ms=5.0, headwind_ms=0.0, cda_m2=cda)
    f_headwind = aerodynamic_drag_force_n(v_ms=5.0, headwind_ms=3.0, cda_m2=cda)
    f_tailwind = aerodynamic_drag_force_n(v_ms=5.0, headwind_ms=-2.0, cda_m2=cda)

    assert f_still > 0.0
    assert f_headwind > f_still
    assert f_tailwind < f_still


def test_grade_force_uphill_downhill():
    """Verify grade force sign: positive for uphill, negative for downhill."""
    mass = 600.0
    f_up = grade_force_n(mass, grade_fraction=0.05)
    f_flat = grade_force_n(mass, grade_fraction=0.0)
    f_down = grade_force_n(mass, grade_fraction=-0.05)

    assert f_up > 0.0
    assert f_flat == 0.0
    assert f_down == -f_up


def test_inertial_force_acceleration():
    """Verify inertial force matches F = m*(1+lambda)*a."""
    mass = 500.0
    f_acc = inertial_force_n(mass, acceleration_ms2=1.0, rotational_inertia_factor=0.05)
    f_dec = inertial_force_n(mass, acceleration_ms2=-1.0, rotational_inertia_factor=0.05)

    assert pytest.approx(f_acc, rel=1e-4) == 500.0 * 1.05 * 1.0
    assert pytest.approx(f_dec, rel=1e-4) == -500.0 * 1.05 * 1.0


def test_flat_road_consumption_range():
    """CRITICAL TEST: Total Wh/km for flat road, 2 passengers, 20 km/h, 25C is between 25 and 70 Wh/km."""
    v_kmh = 20.0
    v_ms = v_kmh / 3.6
    passengers_mass_kg = 2 * 65.0  # 2 passengers @ 65 kg

    v_params = PRESETS["lead_acid_48V_100Ah"]["vehicle"]
    rho_25 = air_density_from_temp(25.0)

    res = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=0.0,
        headwind_ms=0.0,
        payload_mass_kg=passengers_mass_kg,
        vehicle_params=v_params,
        air_density_kg_m3=rho_25,
    )

    power_batt_w = res["battery_power_w"]
    # Wh per km = Power [W] / Speed [km/h]
    wh_per_km = power_batt_w / v_kmh

    assert 25.0 <= wh_per_km <= 70.0, f"Expected 25-70 Wh/km, got {wh_per_km:.2f} Wh/km"


def test_consumption_monotonicity():
    """CRITICAL TEST: Consumption increases with passengers, with uphill grade, and with headwind."""
    v_kmh = 20.0
    v_ms = v_kmh / 3.6
    v_params = PRESETS["lead_acid_48V_100Ah"]["vehicle"]
    rho = air_density_from_temp(25.0)

    # Base case: 0 passengers, flat, no wind
    base = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=0.0,
        headwind_ms=0.0,
        payload_mass_kg=0.0,
        vehicle_params=v_params,
        air_density_kg_m3=rho,
    )["battery_power_w"]

    # More passengers: 4 passengers (260 kg)
    more_passengers = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=0.0,
        headwind_ms=0.0,
        payload_mass_kg=260.0,
        vehicle_params=v_params,
        air_density_kg_m3=rho,
    )["battery_power_w"]

    # Uphill: grade +4%
    uphill = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=0.04,
        headwind_ms=0.0,
        payload_mass_kg=0.0,
        vehicle_params=v_params,
        air_density_kg_m3=rho,
    )["battery_power_w"]

    # Headwind: +3 m/s
    headwind = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=0.0,
        headwind_ms=3.0,
        payload_mass_kg=0.0,
        vehicle_params=v_params,
        air_density_kg_m3=rho,
    )["battery_power_w"]

    assert more_passengers > base, "Consumption must increase with passengers"
    assert uphill > base, "Consumption must increase with uphill grade"
    assert headwind > base, "Consumption must increase with headwind"


def test_downhill_with_zero_regen():
    """CRITICAL TEST: Downhill with eta_regen=0 gives zero battery current beyond aux load."""
    v_kmh = 20.0
    v_ms = v_kmh / 3.6
    v_params = VehicleParams(
        curb_mass_kg=350.0,
        driver_mass_kg=65.0,
        crr=0.018,
        cda_m2=0.85,
        eta_drive=0.82,
        eta_regen=0.0,  # Zero regen
        aux_w=50.0,
    )

    # Steep downhill (-8% grade) where gravity overcomes friction & drag
    res = step_vehicle_physics(
        v_curr_ms=v_ms,
        acceleration_ms2=0.0,
        grade_fraction=-0.08,
        headwind_ms=0.0,
        payload_mass_kg=130.0,
        vehicle_params=v_params,
    )

    # Wheel power must be negative (gravity pulling down)
    assert res["wheel_power_w"] < 0.0
    # Battery power must equal exactly aux_w when eta_regen == 0
    assert pytest.approx(res["battery_power_w"], abs=1e-5) == v_params.aux_w


def test_loss_based_efficiency_model():
    """Verify loss-based drivetrain efficiency model:
    - Agrees within 3% with constant model at 700 W cruise calibration point.
    - Drivetrain efficiency drops significantly below constant model at low power.
    """
    from ebms_sim.vehicle import battery_power_w, drivetrain_loss_w

    p_cruise = 700.0  # W
    aux = 0.0         # Exclude aux to evaluate drive efficiency directly

    # Constant model: P_elec = P_wheel / eta_drive = 700 / 0.82
    p_elec_const = battery_power_w(p_cruise, eta_drive=0.82, eta_regen=0.0, aux_w=aux, eta_model="constant")
    # Loss-based model: P_elec = P_wheel + P_loss(P_wheel)
    p_elec_loss = battery_power_w(p_cruise, eta_drive=0.82, eta_regen=0.0, aux_w=aux, eta_model="loss_based")

    # 1. Must agree within 3% at calibration point (700 W)
    rel_diff = abs(p_elec_loss - p_elec_const) / p_elec_const
    assert rel_diff < 0.03, f"Expected <3% difference at 700W, got {rel_diff * 100:.2f}%"

    # Overall efficiency at cruise: P_wheel / P_elec ~ 0.82
    eta_cruise_loss = p_cruise / p_elec_loss
    assert pytest.approx(eta_cruise_loss, abs=0.02) == 0.82

    # 2. At very low power (50 W), efficiency must fall noticeably below 0.82
    p_low = 50.0  # W
    p_elec_low_loss = battery_power_w(p_low, eta_drive=0.82, eta_regen=0.0, aux_w=aux, eta_model="loss_based")
    eta_low_loss = p_low / p_elec_low_loss

    assert eta_low_loss < 0.60, f"Expected efficiency < 0.60 at 50W, got {eta_low_loss:.3f}"
    assert eta_low_loss < 0.82

