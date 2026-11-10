"""Unit tests for sanity check module (ebms_sim.sanity)."""

import pytest
from ebms_sim.config import PRESETS
from ebms_sim.sanity import (
    run_all_sanity_checks,
    check_wh_per_km_benchmark,
    check_consumption_monotonicity,
    check_downhill_zero_regen_aux_power,
    check_range_soc_monotonicity,
    check_charging_temperature_window,
)


def test_all_sanity_checks_pass_default():
    """Verify all five fundamental physical checks pass with default parameters."""
    results = run_all_sanity_checks()
    assert len(results) == 5

    for name, passed, msg in results:
        assert passed is True, f"Sanity check '{name}' failed: {msg}"


def test_sanity_checks_both_presets():
    """Verify sanity checks pass for both Lead-Acid and LFP presets."""
    for name, preset in PRESETS.items():
        results = run_all_sanity_checks(
            vehicle_params=preset["vehicle"],
            battery_params=preset["battery"],
            thermal_params=preset["thermal"],
        )
        for c_name, passed, msg in results:
            assert passed is True, f"Preset '{name}' failed check '{c_name}': {msg}"

