"""Unit tests for pack lumped thermal model (ebms_sim.thermal)."""

import pytest
from ebms_sim.config import ThermalParams
from ebms_sim.thermal import (
    joule_heat_generation_w,
    effective_thermal_resistance,
    heat_loss_to_ambient_w,
    step_pack_temperature_c,
    step_pack_thermal,
)


def test_joule_heat_generation():
    """Verify Joule heating formula Q = I^2 * R."""
    curr = 30.0  # 30 A
    r = 0.050    # 50 mOhm
    q_gen = joule_heat_generation_w(curr, r)
    assert pytest.approx(q_gen, rel=1e-4) == (30.0 ** 2) * 0.050  # 45 W

    # Negative current (charging / regen) also produces positive Joule heating
    q_charge = joule_heat_generation_w(-30.0, r)
    assert pytest.approx(q_charge, rel=1e-4) == 45.0


def test_airflow_cooling_enhancement():
    """Verify forward speed lowers effective thermal resistance."""
    r_base = 0.65
    r_static = effective_thermal_resistance(r_base, v_ms=0.0)
    r_moving = effective_thermal_resistance(r_base, v_ms=6.0, cooling_factor=0.02)

    assert pytest.approx(r_static, rel=1e-5) == r_base
    assert r_moving < r_static
    expected_moving = r_base / (1.0 + 0.02 * 6.0)
    assert pytest.approx(r_moving, rel=1e-4) == expected_moving


def test_thermal_cooldown_to_ambient():
    """Verify pack cools towards ambient when current is zero."""
    t_pack_init = 40.0
    t_ambient = 25.0
    c_th = 50000.0
    r_th = 0.60
    dt = 3600.0  # 1 hour

    t_1h = step_pack_temperature_c(
        t_pack_c=t_pack_init,
        current_a=0.0,
        r_ohm=0.05,
        t_ambient_c=t_ambient,
        c_th_j_per_k=c_th,
        r_th_k_per_w=r_th,
        dt_s=dt,
    )

    assert t_ambient < t_1h < t_pack_init

    # After very long time (e.g. 100 hours = 12 time constants), temperature should equal ambient
    t_long = step_pack_temperature_c(
        t_pack_c=t_pack_init,
        current_a=0.0,
        r_ohm=0.05,
        t_ambient_c=t_ambient,
        c_th_j_per_k=c_th,
        r_th_k_per_w=r_th,
        dt_s=100 * 3600.0,
    )
    assert pytest.approx(t_long, abs=1e-2) == t_ambient


def test_thermal_equilibrium_under_continuous_load():
    """Verify steady-state temperature reaches T_ambient + Q_gen * R_th."""
    curr = 20.0
    r = 0.05
    q_gen = (curr ** 2) * r  # 20 W
    t_amb = 30.0
    r_th = 0.50
    t_eq_expected = t_amb + q_gen * r_th  # 30 + 10 = 40 deg C

    c_th = 20000.0

    # Simulate for 100,000 seconds
    t_final = step_pack_temperature_c(
        t_pack_c=t_amb,
        current_a=curr,
        r_ohm=r,
        t_ambient_c=t_amb,
        c_th_j_per_k=c_th,
        r_th_k_per_w=r_th,
        dt_s=100000.0,
    )
    assert pytest.approx(t_final, abs=1e-2) == t_eq_expected


def test_step_pack_thermal_dictionary_output():
    """Verify step_pack_thermal produces consistent dictionary."""
    params = ThermalParams(c_th_j_per_k=90000.0, r_th_k_per_w=0.65)
    res = step_pack_thermal(
        t_pack_c=25.0,
        current_a=25.0,
        r_ohm=0.06,
        t_ambient_c=25.0,
        thermal_params=params,
        dt_s=1.0,
        v_ms=5.0,
    )
    assert "t_pack_c" in res
    assert "q_gen_w" in res
    assert "q_loss_w" in res
    assert res["q_gen_w"] > 0.0
    assert res["t_pack_c"] >= 25.0
