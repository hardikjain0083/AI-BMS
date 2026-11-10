"""Sidebar UI widgets for ebms-sim Streamlit application.

Organized into grouped expanders with physical units, tooltips, sensible bounds,
preset synchronization, Scenario freezing, and Guided Demo mode.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np

from ebms_sim.config import (
    VehicleParams,
    BatteryParams,
    ThermalParams,
    ChargingParams,
    PRESETS,
)
from ebms_sim.profiles import RouteType

GUIDED_DEMO_STEPS = [
    {
        "title": "Step 1: Baseline City Cruise",
        "caption": "Flat city road, Lead-Acid 48V pack, 2 passengers, 20 km/h cruising, 25°C ambient.",
        "params": {
            "preset": "lead_acid_48V_100Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 25.0,
            "route_type": "flat",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.0,
            "eta_model": "constant",
        },
    },
    {
        "title": "Step 2: Full Passenger Load",
        "caption": "Adding maximum capacity: 5 passengers + 50 kg luggage drastically increases rolling resistance.",
        "params": {
            "preset": "lead_acid_48V_100Ah",
            "passengers": 5,
            "luggage_kg": 50.0,
            "temp_c": 25.0,
            "route_type": "flat",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.0,
            "eta_model": "constant",
        },
    },
    {
        "title": "Step 3: Flyover / Hill Climb",
        "caption": "Climbing a steep 5% overpass slope causes sharp battery power spikes and accelerates cell depletion.",
        "params": {
            "preset": "lead_acid_48V_100Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 25.0,
            "route_type": "hill-climb",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.0,
            "eta_model": "constant",
        },
    },
    {
        "title": "Step 4: Extreme Cold Weather",
        "caption": "Sub-zero and cold 5°C conditions increase electrolyte internal resistance and derate usable capacity by ~20%.",
        "params": {
            "preset": "lead_acid_48V_100Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 5.0,
            "route_type": "flat",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.0,
            "eta_model": "constant",
        },
    },
    {
        "title": "Step 5: Find Optimal Speed",
        "caption": "Switch to Tab 2 to view the Wh/km vs Speed U-curve and find the energy-minimal cruising speed.",
        "params": {
            "preset": "lead_acid_48V_100Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 25.0,
            "route_type": "flat",
            "speed_cap_kmh": 25.0,
            "regen_eff": 0.0,
            "eta_model": "loss_based",
        },
    },
    {
        "title": "Step 6: Smart Charging Regimes",
        "caption": "Switch to Tab 4 to examine charging current derating at 5°C (trickle) vs 25°C (full) vs 40°C (thermal derating).",
        "params": {
            "preset": "lfp_51V_60Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 25.0,
            "route_type": "flat",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.0,
            "eta_model": "constant",
        },
    },
    {
        "title": "Step 7: AI Surrogate vs Physics",
        "caption": "Switch to Tab 6 to benchmark the XGBoost neural/gradient-boosted surrogate against first-principles physics.",
        "params": {
            "preset": "lfp_51V_60Ah",
            "passengers": 2,
            "luggage_kg": 0.0,
            "temp_c": 25.0,
            "route_type": "flat",
            "speed_cap_kmh": 20.0,
            "regen_eff": 0.15,
            "eta_model": "constant",
        },
    },
]


def render_sidebar_inputs() -> Dict[str, Any]:
    """Render all sidebar inputs and return structured configuration dictionary."""
    st.sidebar.title("⚡ ebms-sim Control Panel")

    # 1. Guided Demo Walkthrough
    st.sidebar.markdown("### 🎓 Guided Professor Demo")
    demo_active = st.sidebar.toggle("Enable Guided Demo Mode", value=False)
    step_info = None

    if demo_active:
        if "demo_step" not in st.session_state:
            st.session_state["demo_step"] = 0

        curr_step = st.session_state["demo_step"]
        step_info = GUIDED_DEMO_STEPS[curr_step]

        st.sidebar.info(f"**{step_info['title']}**\n\n{step_info['caption']}")

        col_b1, col_b2 = st.sidebar.columns(2)
        if col_b1.button("⬅️ Previous") and curr_step > 0:
            st.session_state["demo_step"] -= 1
            st.rerun()
        if col_b2.button("Next ➡️") and curr_step < len(GUIDED_DEMO_STEPS) - 1:
            st.session_state["demo_step"] += 1
            st.rerun()

    demo_overrides = step_info["params"] if (demo_active and step_info) else {}

    # 2. Preset Selection
    preset_choice = demo_overrides.get(
        "preset",
        st.sidebar.selectbox(
            "Pack Chemistry Preset",
            ["lead_acid_48V_100Ah", "lfp_51V_60Ah"],
            index=0,
            help="Select factory standard traction battery pack specification.",
        )
    )
    preset_data = PRESETS[preset_choice]

    # Quick Reset & Freeze Buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    freeze_scenario_a = col_btn2.button("❄️ Freeze Scenario A", help="Save current inputs as baseline Scenario A.")

    # 3. Vehicle Expander
    with st.sidebar.expander("🚗 Vehicle Dynamics", expanded=False):
        curb_mass = st.number_input(
            "Curb Mass [kg]", min_value=200.0, max_value=500.0,
            value=float(preset_data["vehicle"].curb_mass_kg), step=10.0,
            help="Unladen vehicle chassis mass without passengers.",
        )
        passengers = st.slider(
            "Passengers (65 kg each)", min_value=0, max_value=5,
            value=int(demo_overrides.get("passengers", 2)),
            help="Adult passengers in passenger cabin.",
        )
        luggage_kg = st.slider(
            "Luggage Mass [kg]", min_value=0.0, max_value=100.0,
            value=float(demo_overrides.get("luggage_kg", 0.0)), step=5.0,
            help="Cargo and luggage payload.",
        )
        crr = st.slider(
            "Rolling Coeff Crr", min_value=0.010, max_value=0.030,
            value=float(preset_data["vehicle"].crr), step=0.001, format="%.3f",
            help="Tire rolling resistance on asphalt/pavement.",
        )
        cda = st.slider(
            "Aerodynamic Cd*A [m²]", min_value=0.5, max_value=1.5,
            value=float(preset_data["vehicle"].cda_m2), step=0.05,
            help="Frontal area multiplied by drag coefficient.",
        )
        eta_model = st.selectbox(
            "Drivetrain Efficiency Model",
            ["constant", "loss_based"],
            index=0 if demo_overrides.get("eta_model", "constant") == "constant" else 1,
            help="Constant efficiency or load-dependent quadratic loss model.",
        )
        eta_drive = st.slider(
            "Propulsion Efficiency η", min_value=0.60, max_value=0.95,
            value=float(preset_data["vehicle"].eta_drive), step=0.01,
            help="Inverter + BLDC motor + transmission combined efficiency.",
        )
        regen_eff = st.slider(
            "Regen Efficiency η_regen", min_value=0.0, max_value=0.30,
            value=float(demo_overrides.get("regen_eff", 0.0)), step=0.02,
            help="Kinetic energy recovery efficiency during braking (default 0.0).",
        )
        aux_w = st.number_input(
            "Auxiliary Electrical Load [W]", min_value=10.0, max_value=250.0,
            value=float(preset_data["vehicle"].aux_w), step=5.0,
            help="Continuous electrical consumption of lights, cluster, BMS.",
        )

    # 4. Battery Expander
    with st.sidebar.expander("🔋 Battery Pack State", expanded=False):
        start_soc = st.slider(
            "Initial State of Charge (SOC)", min_value=0.10, max_value=1.00,
            value=0.85, step=0.05, format="%.2f",
            help="Pack SOC fraction at the beginning of the trip.",
        )
        soh = st.slider(
            "State of Health (SOH)", min_value=0.50, max_value=1.00,
            value=1.00, step=0.05, format="%.2f",
            help="Capacity and impedance degradation health factor.",
        )
        r0_override = st.number_input(
            "Base Internal Resistance R0 [Ω]", min_value=0.005, max_value=0.200,
            value=float(preset_data["battery"].r0_ohm), step=0.005, format="%.3f",
            help="Pack internal resistance at 25°C, SOH 1.0.",
        )
        cutoff_soc = st.number_input(
            "Depletion Cutoff SOC", min_value=0.02, max_value=0.20,
            value=float(preset_data["battery"].cutoff_soc), step=0.01, format="%.2f",
            help="Threshold below which battery is considered empty.",
        )

    # 5. Environment Expander
    with st.sidebar.expander("🌡️ Ambient Environment", expanded=False):
        ambient_temp_c = st.slider(
            "Ambient Temperature [°C]", min_value=-5.0, max_value=48.0,
            value=float(demo_overrides.get("temp_c", 25.0)), step=1.0,
            help="Outside ambient air temperature.",
        )
        headwind_ms = st.slider(
            "Headwind Velocity [m/s]", min_value=-5.0, max_value=5.0,
            value=0.0, step=0.5,
            help="Wind velocity (+ for headwind, - for tailwind).",
        )

    # 6. Route Expander
    with st.sidebar.expander("🛣️ Route & Topography", expanded=False):
        route_type = st.selectbox(
            "Topography Profile",
            ["flat", "rolling", "hill-climb", "hill-descent", "mixed", "custom"],
            index=["flat", "rolling", "hill-climb", "hill-descent", "mixed", "custom"].index(
                demo_overrides.get("route_type", "flat")
            ),
            help="Select route elevation profile geometry.",
        )
        distance_km = st.slider(
            "Trip Distance [km]", min_value=1.0, max_value=20.0,
            value=3.0, step=0.5,
            help="Total route length to simulate.",
        )

        custom_df = None
        if route_type == "custom":
            st.markdown("**Custom Segment Table**")
            default_segs = pd.DataFrame([
                {"length_m": 500, "grade_pct": 0.0},
                {"length_m": 800, "grade_pct": 3.5},
                {"length_m": 700, "grade_pct": -2.0},
                {"length_m": 1000, "grade_pct": 0.5},
            ])
            custom_df = st.data_editor(default_segs, num_rows="dynamic")

    # 7. Driving Style Expander
    with st.sidebar.expander("⏱️ Driving Behavior & Constraints", expanded=False):
        speed_cap = st.slider(
            "Speed Cap [km/h]", min_value=10.0, max_value=25.0,
            value=float(demo_overrides.get("speed_cap_kmh", 22.0)), step=1.0,
            help="Driver maximum speed ceiling (legal Indian limit: 25 km/h).",
        )
        aggressiveness = st.slider(
            "Driver Aggressiveness", min_value=0.5, max_value=1.5,
            value=1.0, step=0.1,
            help="Acceleration and braking intensity.",
        )
        enable_time_constraint = st.checkbox("Enforce Max Trip Time Constraint", value=False)
        max_time_min = None
        if enable_time_constraint:
            min_possible_min = (distance_km / 25.0) * 60.0
            max_time_min = st.slider(
                "Max Trip Duration [minutes]",
                min_value=float(round(min_possible_min, 1)),
                max_value=float(round(min_possible_min * 2.0, 1)),
                value=float(round(min_possible_min * 1.3, 1)),
                step=0.5,
                help="Trip completion deadline for DP speed scheduling.",
            )

    # Build dataclasses
    vehicle_params = VehicleParams(
        curb_mass_kg=curb_mass,
        driver_mass_kg=float(preset_data["vehicle"].driver_mass_kg),
        crr=crr,
        cda_m2=cda,
        eta_drive=eta_drive,
        eta_regen=regen_eff,
        aux_w=aux_w,
        rotational_inertia_factor=preset_data["vehicle"].rotational_inertia_factor,
        max_speed_kmh=speed_cap,
        eta_model=eta_model,  # type: ignore
        loss_p0_w=preset_data["vehicle"].loss_p0_w,
        loss_c1=preset_data["vehicle"].loss_c1,
        loss_c2=preset_data["vehicle"].loss_c2,
    )
    battery_params = BatteryParams(
        chemistry=preset_data["battery"].chemistry,
        nominal_voltage_v=preset_data["battery"].nominal_voltage_v,
        nominal_capacity_ah=preset_data["battery"].nominal_capacity_ah,
        nominal_energy_wh=preset_data["battery"].nominal_energy_wh,
        r0_ohm=r0_override,
        peukert_exp=preset_data["battery"].peukert_exp,
        peukert_i_ref_a=preset_data["battery"].peukert_i_ref_a,
        temp_coeff_r=preset_data["battery"].temp_coeff_r,
        soh_coeff_r=preset_data["battery"].soh_coeff_r,
        cutoff_soc=cutoff_soc,
        max_continuous_current_a=preset_data["battery"].max_continuous_current_a,
        pack_mass_kg=preset_data["battery"].pack_mass_kg,
    )
    thermal_params = preset_data["thermal"]
    charging_params = preset_data["charging"]

    payload_mass_kg = passengers * 65.0 + luggage_kg

    return {
        "vehicle_params": vehicle_params,
        "battery_params": battery_params,
        "thermal_params": thermal_params,
        "charging_params": charging_params,
        "payload_mass_kg": payload_mass_kg,
        "passengers": passengers,
        "luggage_kg": luggage_kg,
        "ambient_temp_c": ambient_temp_c,
        "headwind_ms": headwind_ms,
        "start_soc": start_soc,
        "soh": soh,
        "route_type": route_type,
        "distance_km": distance_km,
        "custom_df": custom_df,
        "speed_cap_kmh": speed_cap,
        "aggressiveness": aggressiveness,
        "max_time_min": max_time_min,
        "freeze_scenario_a": freeze_scenario_a,
        "preset_choice": preset_choice,
    }

