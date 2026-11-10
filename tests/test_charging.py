"""Unit tests for smart charging and bell-curve thermal derating (ebms_sim.charging)."""

import pytest
import numpy as np
from ebms_sim.config import PRESETS
from ebms_sim.charging import (
    temperature_charge_multiplier,
    max_charge_current_a,
    simulate_charge,
)


def test_charging_temperature_bell_curve():
    """CRITICAL TEST: Charging current is 0 below 0C and above 45C, and maximum between 15-35C."""
    # 1. Strictly 0 below 0C
    assert temperature_charge_multiplier(-10.0) == 0.0
    assert temperature_charge_multiplier(-0.1) == 0.0
    assert temperature_charge_multiplier(0.0) == 0.0

    # 2. Strictly 0 above 45C
    assert temperature_charge_multiplier(45.0) == 0.0
    assert temperature_charge_multiplier(45.5) == 0.0
    assert temperature_charge_multiplier(60.0) == 0.0

    # 3. Maximum (1.00) between 15C and 35C
    temps_optimal = [15.0, 18.0, 20.0, 25.0, 30.0, 35.0]
    for t in temps_optimal:
        assert temperature_charge_multiplier(t) == 1.0, f"Expected 1.0 at {t}C"

    # 4. Reduced region below ~5C and 5-15C
    assert 0.0 < temperature_charge_multiplier(3.0) <= 0.20
    assert 0.20 < temperature_charge_multiplier(10.0) < 1.00

    # 5. Reduced region 35-45C
    assert 0.0 < temperature_charge_multiplier(40.0) < 1.00

    # 6. Monotonicity checks
    assert temperature_charge_multiplier(2.0) < temperature_charge_multiplier(8.0)
    assert temperature_charge_multiplier(8.0) < temperature_charge_multiplier(15.0)
    assert temperature_charge_multiplier(35.0) > temperature_charge_multiplier(40.0)
    assert temperature_charge_multiplier(40.0) > temperature_charge_multiplier(44.0)


def test_max_charge_current_absolute_values():
    """Verify max charge current scales with nominal capacity and respects zero boundaries."""
    nom_cap = 100.0  # Ah
    c_rate = 0.33    # 33 A max
    taper_soc = 0.80

    # Below 0C
    i_subzero = max_charge_current_a(-5.0, soc=0.5, nominal_capacity_ah=nom_cap, max_charge_c_rate=c_rate)
    assert i_subzero == 0.0

    # Above 45C
    i_overtemp = max_charge_current_a(46.0, soc=0.5, nominal_capacity_ah=nom_cap, max_charge_c_rate=c_rate)
    assert i_overtemp == 0.0

    # Inside optimal window (25C)
    i_optimal = max_charge_current_a(25.0, soc=0.5, nominal_capacity_ah=nom_cap, max_charge_c_rate=c_rate)
    assert pytest.approx(i_optimal, rel=1e-3) == 33.0


def test_cc_cv_current_taper():
    """Verify CC phase keeps max current and CV phase tapers down monotonically."""
    nom_cap = 100.0
    c_rate = 0.30  # 30 A
    taper_soc = 0.80
    min_curr = 2.0

    # CC region: 0.2, 0.5, 0.79 SOC should all receive 30 A
    assert max_charge_current_a(25.0, 0.2, taper_soc=taper_soc) == max_charge_current_a(25.0, 0.79, taper_soc=taper_soc)

    # CV region: 0.85, 0.90, 0.95, 0.99
    i_85 = max_charge_current_a(25.0, 0.85, taper_soc=taper_soc, min_charge_current_a=min_curr)
    i_90 = max_charge_current_a(25.0, 0.90, taper_soc=taper_soc, min_charge_current_a=min_curr)
    i_95 = max_charge_current_a(25.0, 0.95, taper_soc=taper_soc, min_charge_current_a=min_curr)

    assert i_85 > i_90 > i_95 >= min_curr
    # At 1.0 SOC, current is 0
    assert max_charge_current_a(25.0, 1.0, taper_soc=taper_soc) == 0.0


def test_simulate_charge_session():
    """Verify simulate_charge reaches target SOC, tracks thermal rise, and reports time."""
    preset = PRESETS["lead_acid_48V_100Ah"]

    res = simulate_charge(
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        charging_params=preset["charging"],
        start_soc=0.30,
        target_soc=0.90,
        t_ambient_c=25.0,
        dt_s=60.0,
    )

    assert res.summary.final_soc >= 0.89
    assert res.summary.time_to_full_hours > 1.0  # At 0.25C rate, takes ~2.5-3.5 hours
    assert res.summary.time_to_full_hours < 8.0
    assert res.summary.peak_temp_c >= 25.0
    assert res.summary.energy_charged_wh > 0.0
    assert not res.summary.aborted_high_temp
    assert not res.summary.aborted_freezing


def test_lead_acid_voltage_temperature_compensation():
    """Verify lead-acid temperature-compensated absorption & float voltage:
    V_cell = V_ref + (-0.003) * (T - 25) per cell, 24 cells for 48V pack.
    """
    from ebms_sim.charging import (
        lead_acid_cell_charge_voltage_v,
        lead_acid_pack_charge_voltage_v,
    )

    # 1. Single cell at 25C
    v_abs_25 = lead_acid_cell_charge_voltage_v(25.0, "absorption")
    v_float_25 = lead_acid_cell_charge_voltage_v(25.0, "float")
    assert pytest.approx(v_abs_25, abs=1e-4) == 2.40
    assert pytest.approx(v_float_25, abs=1e-4) == 2.25

    # 2. Hot temperature (35C): voltage decreases by 30 mV/cell
    v_abs_35 = lead_acid_cell_charge_voltage_v(35.0, "absorption")
    assert pytest.approx(v_abs_35, abs=1e-4) == 2.37

    # 3. Cold temperature (15C): voltage increases by 30 mV/cell
    v_abs_15 = lead_acid_cell_charge_voltage_v(15.0, "absorption")
    assert pytest.approx(v_abs_15, abs=1e-4) == 2.43

    # 4. 24-cell pack voltage
    pack_abs_25 = lead_acid_pack_charge_voltage_v(25.0, "absorption")
    pack_float_25 = lead_acid_pack_charge_voltage_v(25.0, "float")
    assert pytest.approx(pack_abs_25, abs=1e-3) == 57.60  # 24 * 2.40
    assert pytest.approx(pack_float_25, abs=1e-3) == 54.00  # 24 * 2.25


def test_chemistry_dispatch_and_lead_acid_limits():
    """Verify max_charge_current dispatches on chemistry:
    - Lead-acid: current limit 0.10-0.20 C, cutoff above 50C, 0 below 0C.
    - LFP: bell-curve, cutoff above 45C, 0 below 0C.
    """
    nom_cap = 100.0

    # Lead-Acid tests
    # At 25C: 0.10-0.20 C (10-20 A)
    i_pb_25 = max_charge_current_a(25.0, soc=0.5, chemistry="lead_acid", nominal_capacity_ah=nom_cap)
    assert 10.0 <= i_pb_25 <= 20.0

    # Lead-acid at 48C: still within allowed window (< 50C)
    i_pb_48 = max_charge_current_a(48.0, soc=0.5, chemistry="lead_acid", nominal_capacity_ah=nom_cap)
    assert i_pb_48 > 0.0

    # Lead-acid above 50C: strictly 0 A
    assert max_charge_current_a(50.0, soc=0.5, chemistry="lead_acid", nominal_capacity_ah=nom_cap) == 0.0
    assert max_charge_current_a(52.0, soc=0.5, chemistry="lead_acid", nominal_capacity_ah=nom_cap) == 0.0

    # Lead-acid below 0C: strictly 0 A
    assert max_charge_current_a(-5.0, soc=0.5, chemistry="lead_acid", nominal_capacity_ah=nom_cap) == 0.0

    # LFP tests: cutoff at 45C (so at 48C it must be 0 A, unlike lead-acid)
    i_lfp_48 = max_charge_current_a(48.0, soc=0.5, chemistry="lfp", nominal_capacity_ah=nom_cap)
    assert i_lfp_48 == 0.0


