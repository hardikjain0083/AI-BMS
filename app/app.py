"""ebms-sim: Interactive AI Battery Management System Web Application.

University professor demonstration dashboard implementing first-principles physics,
energy optimization, smart charging, and AI surrogate benchmarking.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is always in sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# When Streamlit executes app/app.py directly, it registers sys.modules['app'] as a single file module.
# Setting __path__ ensures Python recognizes 'app' as a package with submodules (ui_inputs, adapters, ui_plots).
_app_mod = sys.modules.get("app")
if _app_mod is not None and not hasattr(_app_mod, "__path__"):
    _app_mod.__path__ = [str(APP_DIR)]

# Set page config
st.set_page_config(
    page_title="ebms-sim | E-Rickshaw BMS Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui_inputs import render_sidebar_inputs
from app.adapters import (
    REPO_ROOT,
    cached_simulate_trip,
    cached_speed_sweep,
    cached_simulate_charge,
    calculate_range_confidence_band,
    compare_flat_vs_hilly_routes,
    get_charging_matrix_comparison,
    run_ai_vs_physics_comparison,
    load_tuktuk_replay_trips,
)
from app.ui_plots import (
    build_trip_simulation_figure,
    build_speed_sweep_figure,
    build_range_uncertainty_gauge,
    build_hills_comparison_bar,
    build_smart_charging_dashboard,
    build_scenario_comparison_figure,
    build_ai_scatter_parity_figure,
)
from ebms_sim.profiles import generate_speed_profile, generate_gradient_profile
from ebms_sim.optimizer import optimize_trip_speeds_dp, RouteSegment
from ebms_sim.charging import temperature_charge_multiplier
from ebms_sim.sanity import run_all_sanity_checks


def render_plotly(fig, container=None, **kwargs):
    """Render Plotly figure using width='stretch' if supported, else fallback to use_container_width=True."""
    target = container if container is not None else st
    try:
        target.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        target.plotly_chart(fig, use_container_width=True, **kwargs)


def render_dataframe(df, container=None, **kwargs):
    """Render DataFrame using width='stretch' if supported, else fallback to use_container_width=True."""
    target = container if container is not None else st
    try:
        target.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        target.dataframe(df, use_container_width=True, **kwargs)


def binned_series_for_chart(series: pd.Series, bins: int = 20) -> pd.Series:
    """Compute value counts in bins and format interval labels as clean strings to avoid Altair UserWarning."""
    vc = series.value_counts(bins=bins).sort_index()
    vc.index = [f"[{iv.left:.1f}, {iv.right:.1f}]" if hasattr(iv, "left") else str(iv) for iv in vc.index]
    return vc


def main():
    # 1. Permanent Warning Banner
    st.warning("⚠️ **First-principles simulation. NOT validated against a real vehicle.**")

    # Header
    st.title("⚡ ebms-sim: AI Battery Management System Simulator")
    st.caption("Physics-Based Longitudinal Vehicle Dynamics, Electrochemistry, Thermal Model & AI Surrogate for Indian E-Rickshaws")
    with st.expander("📖 Start here: terms used across the dashboard", expanded=False):
        st.markdown("""
        **What this tool does:** estimates an e-rickshaw trip using a mathematical vehicle-and-battery model, then explores range, charging, and model predictions. It is a simulator, not a live vehicle monitor.

        **Battery options:** **Lead-acid** is a common, lower-cost traction chemistry. **LFP** means *lithium iron phosphate*; it is included as a thermally stable, long-life lithium-ion comparison. **NMC** means *nickel manganese cobalt oxide*; it represents a higher energy-density lithium-ion comparison. Pack voltage and amp-hour values are nominal preset ratings, not a claim that every real pack has those exact properties.

        **SOC** = *state of charge*, the estimated fraction of capacity remaining. **SOH** = *state of health*, a simplified capacity/impedance condition factor (1.0 is the model's healthy baseline). **BMS** = *battery management system*. **Wh/km** means watt-hours consumed per kilometre; lower generally means more distance from the same stored energy. **Regen** means regenerative braking, which returns some braking energy to the battery. **Thermal derating** means reducing charging current to protect a battery at an unsuitable temperature.

        **Crr** is the tire rolling-resistance coefficient; **CdA** is drag coefficient times frontal area. **OCV** is open-circuit voltage (battery voltage without load). The **Peukert effect** describes how lead-acid batteries deliver less usable capacity under higher current. **CC-CV** means constant-current/constant-voltage charging. **DP** is dynamic programming, the method used to select speeds under a trip-time constraint. **MAPE** is mean absolute percentage error, an average relative prediction error.

        **AI surrogate** means a machine-learning approximation of the physics model, not an independently validated battery controller. All outputs depend on the displayed assumptions and are not calibrated against instrumented vehicle or laboratory data.
        """)

    # 2. Render Sidebar Inputs
    cfg = render_sidebar_inputs()

    vp = cfg["vehicle_params"]
    bp = cfg["battery_params"]
    tp = cfg["thermal_params"]
    cp = cfg["charging_params"]
    payload_mass_kg = cfg["payload_mass_kg"]
    ambient_temp_c = cfg["ambient_temp_c"]
    headwind_ms = cfg["headwind_ms"]
    start_soc = cfg["start_soc"]
    soh = cfg["soh"]
    route_type = cfg["route_type"]
    distance_km = cfg["distance_km"]
    speed_cap_kmh = cfg["speed_cap_kmh"]
    aggressiveness = cfg["aggressiveness"]
    max_time_min = cfg["max_time_min"]

    # Generate or build driving profiles
    duration_s = max(60, int((distance_km * 1000.0) / (min(speed_cap_kmh, 20.0) / 3.6)))
    speed_profile = generate_speed_profile(
        duration_s=duration_s,
        speed_cap_kmh=speed_cap_kmh,
        aggressiveness=aggressiveness,
        seed=42,
    )

    if route_type == "custom" and cfg["custom_df"] is not None and len(cfg["custom_df"]) > 0:
        c_df = cfg["custom_df"]
        segs = []
        for _, row in c_df.iterrows():
            length = max(10, int(row.get("length_m", 100)))
            grade = float(row.get("grade_pct", 0.0)) / 100.0
            segs.append(np.full(length, grade))
        gradient_profile = np.concatenate(segs)
        if len(gradient_profile) < duration_s:
            gradient_profile = np.pad(gradient_profile, (0, duration_s - len(gradient_profile)), mode="edge")
        else:
            gradient_profile = gradient_profile[:duration_s]
    else:
        gradient_profile = generate_gradient_profile(
            length_steps=duration_s,
            route_type=route_type,
            seed=42,
        )

    # Execute main simulation
    try:
        ts_df, summary = cached_simulate_trip(
            vehicle_params=vp,
            battery_params=bp,
            thermal_params=tp,
            speed_profile_tuple=tuple(speed_profile),
            gradient_profile_tuple=tuple(gradient_profile),
            payload_mass_kg=payload_mass_kg,
            t_ambient_c=ambient_temp_c,
            headwind_ms=headwind_ms,
            start_soc=start_soc,
            soh=soh,
        )
    except Exception as e:
        st.error(f"Simulation Error: {e}")
        return

    # Scenario A Freezing Logic
    if cfg["freeze_scenario_a"]:
        st.session_state["scenario_a"] = {
            "summary": summary,
            "ts_df": ts_df.copy(),
            "preset": cfg["preset_choice"],
            "vp": vp,
            "bp": bp,
        }
        st.toast("Scenario A successfully frozen!", icon="❄️")

    # 3. Define 9 Structured Tabs
    tabs = st.tabs([
        "📊 Trip Simulation",
        "🧭 Range & Speed Advisor",
        "⛰️ Hills & Regen",
        "🔋 Smart Charging",
        "⚖️ Scenario Comparison",
        "🤖 AI vs Physics",
        "📈 Data & Model",
        "📝 Equations & Limits",
        "🩺 Health Checks",
    ])

    # =========================================================================
    # TAB 1: TRIP SIMULATION
    # =========================================================================
    with tabs[0]:
        st.subheader("1-Second Resolution Trip Simulation")
        st.caption("Each chart follows the trip over elapsed time. Hover a trace for its value; use the legend to show or hide a trace.")

        tab1_mode = st.radio(
            "Telemetry Source",
            ["⚡ Live Physics Simulation", "🔄 Replay Dataset Trip (Tuk-Tuk Telemetry)"],
            horizontal=True,
            help="Choose between real-time longitudinal vehicle dynamics or recorded Indian e-rickshaw trips.",
        )

        if tab1_mode == "⚡ Live Physics Simulation":
            # Metric Cards
            m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
            m_c1.metric("Specific Consumption", f"{summary['wh_per_km']:.1f} Wh/km", help="Trip electrical energy divided by distance. Lower Wh/km generally indicates more efficient energy use.")
            m_c2.metric("Distance Covered", f"{summary['distance_km']:.2f} km", help="Distance represented by this simulated trip.")
            m_c3.metric("Total Energy", f"{summary['total_energy_wh']:.1f} Wh", help="Estimated battery energy used during the trip; 1,000 Wh equals 1 kWh.")
            m_c4.metric("Final SOC", f"{summary['final_soc']*100:.1f} %", delta=f"{(summary['final_soc']-start_soc)*100:.1f}%", help="Estimated state of charge at trip end. The delta compares it with the selected starting SOC.")
            m_c5.metric("Peak Current", f"{np.max(ts_df['current_a']):.1f} A", help="Highest modeled battery current during the trip; larger peaks mean a heavier electrical load.")
            m_c6.metric("Peak Pack Temp", f"{np.max(ts_df['temp_c']):.1f} °C", delta=f"{np.max(ts_df['temp_c'])-ambient_temp_c:+.1f}°C", help="Highest modeled battery-pack temperature. Delta is relative to ambient temperature.")

            if summary["cutoff_reached"]:
                st.error("⚠️ Battery reached low-SOC cutoff threshold before trip completion!")

            # Interactive Plotly Figures
            fig_trip = build_trip_simulation_figure(ts_df)
            render_plotly(fig_trip)
            st.caption("**How to read the trip graph:** top: vehicle speed and road elevation; second: mechanical wheel power versus electrical battery power; third: pack voltage and current; bottom: SOC and pack temperature. The two overlaid values in panels 1, 3, and 4 use separate vertical axes.")

        else:
            df_replay = load_tuktuk_replay_trips()
            if df_replay.empty:
                st.info("No recorded trips found. Compact replay dataset will be generated automatically at data/tuktuk_trips_subset.parquet.")
            else:
                trip_ids = sorted(df_replay["trip_id"].unique())
                col_sel1, col_sel2, col_sel3 = st.columns([3, 2, 2])

                def trip_label(tid):
                    sub = df_replay[df_replay["trip_id"] == tid].iloc[0]
                    return f"Trip #{tid} - {sub['road_regime']} ({sub['battery_chemistry']} {sub['battery_capacity_kwh']} kWh, {sub['rickshaw_class']})"

                sel_trip_id = col_sel1.selectbox(
                    "Select Representative Recorded Trip",
                    trip_ids,
                    format_func=trip_label,
                    help="Choose one recorded example trip from the bundled telemetry subset. The selection changes the replayed traces and summary metrics.",
                )
                approx_wheel = col_sel2.checkbox("Approximate Wheel Power", value=True, help="Estimates Wheel Power (W) as pack_power_kw * 1000 * 0.84 for discharge.")
                hide_wheel = col_sel3.checkbox("Hide Wheel Power Trace", value=False, help="Hides the Wheel Power trace on Chart 2.")

                trip_data = df_replay[df_replay["trip_id"] == sel_trip_id].copy().reset_index(drop=True)

                # Metric Cards for replayed trip (aligning units: pack_power_kw -> W)
                p_kw = trip_data["pack_power_kw"]
                tot_energy_wh = float(np.sum(p_kw * 1000.0 * (1.0 / 3600.0)))
                dist_covered_km = float(trip_data["distance_km"].iloc[-1] - trip_data["distance_km"].iloc[0])
                avg_rate_wh_km = float(tot_energy_wh / max(0.01, dist_covered_km))

                m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
                m_c1.metric("Specific Consumption", f"{avg_rate_wh_km:.1f} Wh/km", help="Recorded energy per distance. Lower Wh/km generally indicates more efficient energy use.")
                m_c2.metric("Distance Covered", f"{dist_covered_km:.2f} km", help="Recorded distance covered by the selected trip.")
                m_c3.metric("Total Energy", f"{tot_energy_wh:.1f} Wh", help="Estimated energy represented by the recorded battery-power trace.")
                m_c4.metric("Final SOC", f"{trip_data['soc_pct'].iloc[-1]:.1f} %", delta=f"{trip_data['soc_pct'].iloc[-1] - trip_data['soc_pct'].iloc[0]:.1f}%", help="Recorded trip's final state of charge; delta is the change from its starting SOC.")
                m_c5.metric("Peak Current", f"{np.max(trip_data['pack_current_a']):.1f} A", help="Highest recorded battery current in this trip.")
                m_c6.metric("Peak Pack Temp", f"{np.max(trip_data['battery_temp_c']):.1f} °C", delta=f"{trip_data['battery_temp_c'].iloc[-1] - trip_data['ambient_temp_c'].iloc[0]:+.1f}°C", help="Highest recorded pack temperature; delta compares final pack temperature with trip-start ambient.")

                fig_replay = build_trip_simulation_figure(
                    trip_data,
                    title_suffix=f" (Replay Trip #{sel_trip_id})",
                    show_wheel_power=not hide_wheel,
                    approximate_wheel_power=approx_wheel,
                )
                render_plotly(fig_replay)
                st.caption("**How to read the replay graph:** speed/elevation, wheel/battery power, voltage/current, then SOC/temperature are plotted against trip time. Wheel power may be estimated from battery power when the data has no wheel-power measurement; the controls above let you show or hide that estimate.")

    # =========================================================================
    # TAB 2: RANGE AND SPEED ADVISOR
    # =========================================================================
    with tabs[1]:
        st.subheader("Range Projection & Optimal Cruising Speed")
        st.caption("Range is an estimate based on remaining modeled energy and a temperature/SOH uncertainty rule. The speed curve shows energy used per kilometre on a flat-road sweep; its lowest point is the most energy-efficient modeled cruising speed.")

        nom_range, min_r, max_r, band_exp = calculate_range_confidence_band(
            nominal_range_km=summary["range_remaining_km"],
            ambient_temp_c=ambient_temp_c,
            soh=soh,
        )

        fig_gauge = build_range_uncertainty_gauge(nom_range, min_r, max_r)
        render_plotly(fig_gauge)
        st.caption("**Range graph:** the green horizontal span is the estimated low-to-high range interval; the vertical marker is the nominal estimate. It is a modeled uncertainty band, not a statistical guarantee.")
        st.caption(f"ℹ️ **Uncertainty Rule**: {band_exp}")

        st.divider()

        # Wh/km vs Speed Sweep
        spds, wh_arr, best_spd, min_wh = cached_speed_sweep(
            vehicle_params=vp,
            battery_params=bp,
            payload_mass_kg=payload_mass_kg,
            grade_fraction=0.0,
            headwind_ms=headwind_ms,
            t_ambient_c=ambient_temp_c,
        )

        constrained_spd = None
        if max_time_min is not None:
            max_s = max_time_min * 60.0
            r_segs = [RouteSegment(distance_m=distance_km * 1000.0, grade_fraction=0.0, headwind_ms=headwind_ms)]
            dp_res = optimize_trip_speeds_dp(
                segments=r_segs,
                vehicle_params=vp,
                battery_params=bp,
                payload_mass_kg=payload_mass_kg,
                t_ambient_c=ambient_temp_c,
                max_total_time_s=max_s,
            )
            constrained_spd = dp_res.segment_speeds_kmh[0]

        fig_sweep = build_speed_sweep_figure(spds, wh_arr, best_spd, min_wh, constrained_spd)
        render_plotly(fig_sweep)
        st.caption("**Speed graph:** each point/curve value is modeled Wh/km at a steady speed. The star marks minimum consumption; when a trip-time limit is active, a diamond marks the speed selected to meet that deadline.")

        # Driver Message Callout
        st.info(
            f"💡 **Driver Advice**: Hold **{best_spd-1.0:.0f}–{best_spd+1.0:.0f} km/h** for maximum range. "
            f"Cruising consumption at this speed is **{min_wh:.1f} Wh/km** on flat asphalt."
            + (f" Time-constrained target speed is **{constrained_spd:.1f} km/h** to meet the {max_time_min:.1f} min deadline." if constrained_spd else "")
        )

    # =========================================================================
    # TAB 3: HILLS & REGENERATION
    # =========================================================================
    with tabs[2]:
        st.subheader("Topography Consumption Comparison & Regenerative Braking")
        if route_type == "custom":
            st.caption("This comparison holds the selected vehicle and battery assumptions constant while changing the road profile and whether regenerative braking is enabled. The trip simulation uses your custom segments; this comparison uses a representative mixed-road profile because it compares standard generated profiles.")
        else:
            st.caption("This comparison holds the selected vehicle and battery assumptions constant while changing the road profile and whether regenerative braking is enabled.")

        hills_res = compare_flat_vs_hilly_routes(
            vehicle_params=vp,
            battery_params=bp,
            thermal_params=tp,
            distance_km=distance_km,
            route_type="mixed" if route_type == "custom" else "hill-climb" if route_type == "flat" else route_type,
            payload_mass_kg=payload_mass_kg,
            ambient_temp_c=ambient_temp_c,
            regen_efficiency=vp.eta_regen,
        )

        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        h_c1.metric("Flat Terrain", f"{hills_res['flat_wh_km']:.1f} Wh/km", help="Baseline energy per kilometre on a flat route.")
        h_c2.metric("Hilly (No Regen)", f"{hills_res['hilly_no_regen_wh_km']:.1f} Wh/km", help="Hilly-route energy per kilometre when braking energy is not recovered.")
        h_c3.metric("Hilly (With Regen)", f"{hills_res['hilly_regen_wh_km']:.1f} Wh/km", help="Hilly-route energy per kilometre with the selected regenerative-braking efficiency.")
        h_c4.metric("Regen Savings", f"{hills_res['regen_savings_wh']:.1f} Wh", delta=f"+{hills_res['regen_range_gain_km']:.2f} km range", help="Estimated energy recovered versus the no-regeneration hilly case, with its equivalent estimated range gain.")

        fig_hills = build_hills_comparison_bar(
            hills_res["flat_wh_km"],
            hills_res["hilly_no_regen_wh_km"],
            hills_res["hilly_regen_wh_km"],
            hills_res["regen_savings_wh"],
        )
        render_plotly(fig_hills)
        st.caption("**Hill graph:** bars compare flat-road consumption with hilly consumption without and with regeneration. A shorter bar means less battery energy per kilometre; recovered energy is summarized separately.")

    # =========================================================================
    # TAB 4: SMART CHARGING
    # =========================================================================
    with tabs[3]:
        st.subheader(f"Smart Charging & Thermal Derating: {bp.chemistry.upper()}")

        c_slider_temp = st.slider(
            "Inspect Charge Acceptance at Temperature [°C]",
            -5.0,
            50.0,
            float(ambient_temp_c),
            1.0,
            help="Set the ambient temperature for the example charging session. The model adjusts allowable charging current according to chemistry and temperature.",
        )

        # Generate thermal profile data
        t_range = np.linspace(-5.0, 50.0, 111)
        if bp.chemistry == "lead_acid":
            mults = [1.0 if (0.0 < t < 50.0) else 0.0 for t in t_range]
        else:
            mults = [temperature_charge_multiplier(t) for t in t_range]
        df_t_mult = pd.DataFrame({"temp_c": t_range, "multiplier": mults})

        # Simulate charge session
        ch_df, ch_summary = cached_simulate_charge(
            battery_params=bp,
            thermal_params=tp,
            charging_params=cp,
            start_soc=0.20,
            target_soc=0.95,
            t_ambient_c=c_slider_temp,
        )

        fig_charge = build_smart_charging_dashboard(df_t_mult, ch_df, c_slider_temp, bp.chemistry)
        render_plotly(fig_charge)
        st.caption("**Charging graphs:** left shows the allowed-current multiplier versus pack temperature (1.0 means no temperature reduction; 0 means charging is stopped), with the selected operating point. Right shows SOC rising and charging current changing during the modeled session; SOC and current use the same panel but different units.")

        st.markdown("### 📋 Multi-Temperature Charging Matrix")
        st.caption("Each row compares a modeled charge session at a different ambient temperature. Charging time and peak current are estimates under this preset, not charger recommendations.")
        render_dataframe(get_charging_matrix_comparison(bp, tp, cp, start_soc=0.20, target_soc=0.95))

        st.caption(
            "ℹ️ **Electrochemical Derating Physics**: At cold temperatures (<5°C), lithium-ion cells suffer from lithium plating on the graphite anode, causing permanent dendrite degradation; charging current is therefore restricted to trickle rates. At high temperatures (>45°C for LFP, >50°C for Lead-Acid), charging ceases to prevent thermal runaway and accelerated electrolyte breakdown."
        )

    # =========================================================================
    # TAB 5: SCENARIO COMPARISON
    # =========================================================================
    with tabs[4]:
        st.subheader("Scenario A vs Scenario B Benchmarking")
        st.caption("Freeze a baseline from the sidebar, then change any inputs. Scenario A is the saved run; Scenario B is the current run.")

        if "scenario_a" not in st.session_state:
            st.info("💡 **Freeze Scenario A**: Click 'Freeze Scenario A' in the sidebar to snapshot the current configuration as baseline.")
        else:
            sc_a = st.session_state["scenario_a"]
            sum_a = sc_a["summary"]
            sum_b = summary

            delta_df = pd.DataFrame([
                {"Metric": "Specific Consumption", "Scenario A": f"{sum_a['wh_per_km']:.1f} Wh/km", "Scenario B (Current)": f"{sum_b['wh_per_km']:.1f} Wh/km", "Delta": f"{sum_b['wh_per_km'] - sum_a['wh_per_km']:+.1f} Wh/km"},
                {"Metric": "Remaining Range", "Scenario A": f"{sum_a['range_remaining_km']:.1f} km", "Scenario B (Current)": f"{sum_b['range_remaining_km']:.1f} km", "Delta": f"{sum_b['range_remaining_km'] - sum_a['range_remaining_km']:+.1f} km"},
                {"Metric": "Total Energy Used", "Scenario A": f"{sum_a['total_energy_wh']:.1f} Wh", "Scenario B (Current)": f"{sum_b['total_energy_wh']:.1f} Wh", "Delta": f"{sum_b['total_energy_wh'] - sum_a['total_energy_wh']:+.1f} Wh"},
                {"Metric": "Final Pack SOC", "Scenario A": f"{sum_a['final_soc']*100:.1f} %", "Scenario B (Current)": f"{sum_b['final_soc']*100:.1f} %", "Delta": f"{(sum_b['final_soc'] - sum_a['final_soc'])*100:+.1f} %"},
            ])
            st.caption("The table's Delta is Scenario B minus Scenario A. For consumption and energy, a negative delta means the current scenario used less; for range and SOC, a positive delta means more remained.")
            render_dataframe(delta_df)

            fig_comp = build_scenario_comparison_figure(sc_a["ts_df"], ts_df, label_a="Scenario A (Frozen)", label_b="Scenario B (Current)")
            render_plotly(fig_comp)
            st.caption("**Comparison graph:** solid blue is the frozen baseline; dashed orange is the current run. The three panels compare speed, battery power, and SOC over elapsed time.")

    # =========================================================================
    # TAB 6: AI VS PHYSICS
    # =========================================================================
    with tabs[5]:
        st.subheader("XGBoost Surrogate vs Physics Engine")
        st.caption("XGBoost is a gradient-boosted decision-tree model (not a neural network here). This panel tests how closely its fast predictions match the simulator's physics-based estimates.")

        ai_res = run_ai_vs_physics_comparison(
            vehicle_params=vp,
            battery_params=bp,
            payload_mass_kg=payload_mass_kg,
            t_ambient_c=ambient_temp_c,
            headwind_ms=headwind_ms,
            soh=soh,
            start_soc=start_soc,
            rain_flaps_or_rack=cfg.get("rain_flaps_or_rack", 0),
        )

        if not ai_res["available"]:
            st.warning(ai_res["message"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Physics Power", f"{ai_res.get('physics_power_w', 0.0):.0f} W", help="Average battery electrical power from numerical physics engine")
            c2.metric("Surrogate Power", f"{ai_res.get('ai_power_w', 0.0):.0f} W", delta=f"{ai_res.get('ai_power_w', 0.0) - ai_res.get('physics_power_w', 0.0):+.1f} W", help="Surrogate prediction (pack_power_kw multiplied by 1000 to yield Watts)")
            c3.metric("Specific Wh/km", f"{ai_res['ai_wh']:.1f} Wh/km", delta=f"{ai_res['ai_wh'] - ai_res['physics_wh']:+.1f} Wh/km", help="Surrogate's estimated energy per kilometre. Delta is surrogate minus physics result; closer to zero means closer agreement.")
            c4.metric("Inference Speed-Up", f"{ai_res['speedup_factor']:.0f}x Faster", help="Microsecond evaluation on MCU")

            fig_parity = build_ai_scatter_parity_figure(ai_res["true_wh_1000"], ai_res["pred_wh_1000"], ai_res["mape_1000"])
            render_plotly(fig_parity)
            st.caption("**Parity graph:** each dot is one test scenario. The horizontal axis is physics-model consumption and the vertical axis is the AI surrogate prediction. Dots close to the dashed y=x line agree; MAPE is mean absolute percentage error.")

            st.caption("ℹ️ **Conceptual Framing**: Machine learning models in BMS serve as **fast numerical approximators** of complex multi-physics differential equations to enable real-time sub-millisecond range and state estimation on low-power 32-bit automotive microcontrollers (ESP32/STM32).")

    # =========================================================================
    # TAB 7: DATA & MODEL
    # =========================================================================
    with tabs[6]:
        st.subheader("Indian Tuk-Tuk Telemetry Dataset & Model Distribution")
        st.caption("The bundled dataset is provided for demonstration and distribution checks; its records are not a live feed from your vehicle.")

        tuktuk_path = str(REPO_ROOT / "data" / "tuktuk_trips_subset.parquet")
        generic_path = str(REPO_ROOT / "data" / "trips.parquet")

        if os.path.exists(tuktuk_path):
            df_trips = pd.read_parquet(tuktuk_path)
            st.success(
                f"Loaded **{len(df_trips):,} representative Indian tuk-tuk rows** across **{df_trips['trip_id'].nunique()} trips** "
                f"({os.path.getsize(tuktuk_path) / (1024*1024):.2f} MB Parquet subset optimized for Streamlit Cloud deployment)."
            )

            st.markdown("#### Operating Regimes & Battery Chemistries")
            st.caption("These tables count distinct trips grouped by road regime or by battery chemistry and rickshaw class.")
            d_c1, d_c2 = st.columns(2)
            d_c1.write("**Trips per Road Regime:**")
            render_dataframe(df_trips.groupby("road_regime")["trip_id"].nunique().rename("Trip Count"), container=d_c1)
            d_c2.write("**Vehicle & Chemistry Breakdown:**")
            render_dataframe(df_trips.groupby(["battery_chemistry", "rickshaw_class"])["trip_id"].nunique().rename("Trip Count"), container=d_c2)

            st.markdown("#### Feature & Telemetry Distributions")
            st.caption("Each histogram groups recorded values into ranges (bins). Taller bars mean more rows fall in that range; these describe the bundled sample, not all Indian e-rickshaw operation.")
            c_hist1, c_hist2, c_hist3 = st.columns(3)
            c_hist1.write("**Pack Power (kW)**")
            c_hist1.bar_chart(binned_series_for_chart(df_trips["pack_power_kw"], bins=20))
            c_hist2.write("**Speed (km/h)**")
            c_hist2.bar_chart(binned_series_for_chart(df_trips["speed_kmh"], bins=20))
            c_hist3.write("**Ambient Temperature (°C)**")
            c_hist3.bar_chart(binned_series_for_chart(df_trips["ambient_temp_c"], bins=15))

        elif os.path.exists(generic_path):
            df_trips = pd.read_parquet(generic_path)
            st.write(f"Loaded **{len(df_trips)} synthetic trip records** from `{generic_path}`.")

            st.markdown("#### Feature Distributions")
            c_hist1, c_hist2, c_hist3 = st.columns(3)
            c_hist1.bar_chart(binned_series_for_chart(df_trips["wh_per_km"], bins=20))
            c_hist2.bar_chart(binned_series_for_chart(df_trips["range_remaining_km"], bins=20))
            c_hist3.bar_chart(binned_series_for_chart(df_trips["ambient_temp_c"], bins=15))
        else:
            st.info("No generated dataset found. Compact parquet dataset will be generated automatically at `data/tuktuk_trips_subset.parquet`.")

    # =========================================================================
    # TAB 8: EQUATIONS & LIMITS
    # =========================================================================
    with tabs[7]:
        st.subheader("Mathematical Formulations & Known Modeling Limitations")

        st.markdown("#### 1. Longitudinal Vehicle Dynamics")
        st.latex(r"F_{\text{wheel}} = F_{\text{roll}} + F_{\text{aero}} + F_{\text{grade}} + F_{\text{inertia}}")
        st.latex(r"F_{\text{roll}} = m \cdot g \cdot C_{rr} \cos(\theta), \quad F_{\text{aero}} = \frac{1}{2} \rho C_d A (v + v_{\text{wind}})^2")
        st.latex(r"P_{\text{wheel}} = F_{\text{wheel}} \cdot v, \quad P_{\text{batt}} = \frac{P_{\text{wheel}}}{\eta_{\text{drive}}} + P_{\text{aux}} \quad (P_{\text{wheel}} \ge 0)")

        st.markdown("#### 2. Drivetrain Quadratic Loss Model")
        st.latex(r"P_{\text{loss}} = p_0 + c_1 \cdot P_{\text{wheel}} + c_2 \cdot P_{\text{wheel}}^2")

        st.markdown("#### 3. Battery Equivalent Circuit & Peukert Law")
        st.latex(r"V = \text{OCV}(\text{SOC}) - I \cdot R(T, \text{SOH}), \quad I_{\text{eff}} = I \cdot \left(\frac{I}{I_{\text{ref}}}\right)^{k-1}")

        st.markdown("#### 4. Lumped Thermal ODE")
        st.latex(r"C_{\text{th}} \frac{dT}{dt} = I^2 R - \frac{T - T_{\text{amb}}}{R_{\text{th\_eff}}}, \quad R_{\text{th\_eff}} = \frac{R_{\text{th}}}{1 + \gamma v}")

        st.markdown("#### 5. Known Assumptions & Limitations")
        st.caption("The equations are compact explanations of the model's main relationships. Read the assumptions before interpreting estimates as real-world performance.")
        st.markdown("""
        - **No Hardware Validation**: Physics equations reflect first-principles engineering models; not yet calibrated against physical e-rickshaw dynamometer or road telemetry.
        - **Linear CC-CV Approximation**: CV charging uses a linear ramp approximation rather than full electrochemical double-layer overpotential ODEs.
        - **Lumped Thermal Approximation**: Pack temperature is modeled as a single uniform thermal node (zero internal temperature gradient).
        - **Pure Coulomb Counting**: Peukert effect applies to lead-acid capacity; internal battery aging degradation uses empirical polynomial coefficients.
        """)

    # =========================================================================
    # TAB 9: HEALTH CHECKS
    # =========================================================================
    with tabs[8]:
        st.subheader("Live Model Physical Sanity Checks (`sanity.py`)")
        st.caption("Checks whether the selected parameters satisfy basic physical relationships, expected trends (monotonicity), and safety boundaries. A pass means the model check passed; it does not certify a real battery or vehicle.")

        checks = run_all_sanity_checks(vp, bp, tp)

        for name, passed, msg in checks:
            if passed:
                st.success(f"✅ **{name}**\n\n{msg}")
            else:
                st.error(f"❌ **{name}**\n\n{msg}")


if __name__ == "__main__":
    main()
