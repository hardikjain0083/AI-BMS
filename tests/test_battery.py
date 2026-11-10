"""Unit tests for battery equivalent circuit model and chemistry logic."""

import pytest
from ebms_sim.battery import (
    ocv_from_soc,
    usable_capacity_factor,
    internal_resistance_ohm,
    peukert_effective_current_a,
    terminal_voltage_v,
    solve_current_and_voltage,
    update_soc_coulomb_counting,
)


def test_ocv_lookup_tables():
    """Verify OCV monotonically increases with SOC for both Lead-Acid and LFP."""
    for chem in ["lead_acid", "lfp"]:
        socs = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
        ocvs = [ocv_from_soc(s, chemistry=chem) for s in socs]
        for i in range(len(ocvs) - 1):
            assert ocvs[i] < ocvs[i + 1], f"{chem} OCV not strictly increasing at SOC {socs[i]}"

    # Check characteristic values
    assert pytest.approx(ocv_from_soc(1.0, "lead_acid"), rel=1e-2) == 51.2
    assert pytest.approx(ocv_from_soc(0.0, "lead_acid"), rel=1e-2) == 42.0

    # LFP has a remarkably flat plateau between 30% and 70%
    lfp_30 = ocv_from_soc(0.3, "lfp")
    lfp_70 = ocv_from_soc(0.7, "lfp")
    assert abs(lfp_70 - lfp_30) < 1.0  # Plateau variation is small (< 1V)


def test_usable_capacity_temperature_derating():
    """Verify usable capacity is ~1.0 at 25C and ~0.8 at 0C."""
    cap_25 = usable_capacity_factor(25.0)
    cap_0 = usable_capacity_factor(0.0)
    cap_minus10 = usable_capacity_factor(-10.0)
    cap_40 = usable_capacity_factor(40.0)

    assert pytest.approx(cap_25, abs=0.01) == 1.0
    assert pytest.approx(cap_0, abs=0.02) == 0.80
    assert cap_minus10 < cap_0
    assert cap_40 >= 1.0


def test_internal_resistance_temperature_and_soh():
    """Verify internal resistance increases with lower temperature and lower SOH."""
    r0_base = 0.050
    # Temperature effect at SOH = 1.0
    r_warm = internal_resistance_ohm(r0_base, temp_c=25.0, soh=1.0)
    r_cold = internal_resistance_ohm(r0_base, temp_c=5.0, soh=1.0)
    assert pytest.approx(r_warm, abs=1e-4) == r0_base
    assert r_cold > r_warm

    # SOH degradation effect at 25C
    r_new = internal_resistance_ohm(r0_base, temp_c=25.0, soh=1.0)
    r_aged = internal_resistance_ohm(r0_base, temp_c=25.0, soh=0.7)
    assert r_aged > r_new


def test_peukert_law():
    """Verify Peukert law amplifies high discharge currents for Lead-Acid but not LFP."""
    i_ref = 5.0
    current = 50.0  # 10x reference

    # LFP: k = 1.0 -> no amplification
    i_eff_lfp = peukert_effective_current_a(current, peukert_exp=1.0, peukert_i_ref_a=i_ref)
    assert pytest.approx(i_eff_lfp, rel=1e-4) == current

    # Lead-Acid: k = 1.20 -> effective current is significantly higher
    i_eff_pb = peukert_effective_current_a(current, peukert_exp=1.20, peukert_i_ref_a=i_ref)
    expected_ratio = (current / i_ref) ** 0.20
    assert pytest.approx(i_eff_pb, rel=1e-3) == current * expected_ratio
    assert i_eff_pb > current * 1.5  # 10^0.2 = ~1.585


def test_terminal_voltage_and_power_solver():
    """Verify V = OCV - I*R and power solver returns exact P = V * I."""
    ocv = 48.0
    r = 0.050
    target_power = 1200.0  # 1.2 kW discharge

    curr, volt = solve_current_and_voltage(target_power, ocv, r)
    assert curr > 0.0
    assert volt < ocv
    assert pytest.approx(volt, rel=1e-4) == terminal_voltage_v(ocv, curr, r)
    assert pytest.approx(curr * volt, rel=1e-4) == target_power

    # Test regenerative power (charging)
    regen_power = -500.0
    curr_regen, volt_regen = solve_current_and_voltage(regen_power, ocv, r)
    assert curr_regen < 0.0
    assert volt_regen > ocv
    assert pytest.approx(curr_regen * volt_regen, rel=1e-4) == regen_power


def test_coulomb_counting_soc_update():
    """Verify SOC decreases with discharge and cold temp accelerates depletion."""
    dt = 3600.0  # 1 hour
    curr = 20.0   # 20 A
    nom_cap = 100.0  # Ah

    # At 25C (factor = 1.0), 20 A for 1 hr drains 20 Ah = 0.20 SOC (for k=1.0)
    soc_start = 0.80
    soc_end_25c = update_soc_coulomb_counting(
        soc_start, curr, dt, nom_cap, temp_c=25.0, peukert_exp=1.0
    )
    assert pytest.approx(soc_end_25c, abs=1e-3) == 0.60

    # At 0C (factor = 0.80), usable cap is 80 Ah, so 20 Ah drains 20/80 = 0.25 SOC!
    soc_end_0c = update_soc_coulomb_counting(
        soc_start, curr, dt, nom_cap, temp_c=0.0, peukert_exp=1.0
    )
    assert pytest.approx(soc_end_0c, abs=1e-3) == 0.55
    assert soc_end_0c < soc_end_25c

