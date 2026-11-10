"""Smoke test for Streamlit application using AppTest (tests/test_app_smoke.py)."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


def test_app_loads_and_runs_without_exceptions():
    """Verify Streamlit app loads from file and executes cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Verify no unhandled exceptions occurred
    assert not at.exception, f"App threw exception: {at.exception}"

    # Verify key elements rendered
    assert len(at.tabs) == 9
    assert len(at.metric) >= 6


def test_app_change_inputs_smoke():
    """Verify modifying sidebar inputs (passengers and temperature) re-runs without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Find and update sliders
    for s in at.slider:
        label = getattr(s, "label", "")
        if "Passengers" in label:
            s.set_value(4)
        elif "Ambient Temperature" in label:
            s.set_value(38.0)

    at.run()
    assert not at.exception
