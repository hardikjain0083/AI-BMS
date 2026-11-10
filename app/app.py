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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set page config
st.set_page_config(
    page_title="ebms-sim | E-Rickshaw BMS Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
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
except ModuleNotFoundError:
    from ui_inputs import render_sidebar_inputs
    from adapters import (
        REPO_ROOT,
        cached_simulate_trip,
        cached_speed_sweep,
        cached_simulate_charge,
        calculate_range_confidence_band,
        compare_flat_vs_hilly_routes,
        get_charging_matrix_comparison,
        run_ai_vs_physics_comparison,
    )
    from ui_plots import (
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


def main():
    # 1. Permanent Warning Banner
    st.warning("⚠️ **First-principles simulation. NOT validated against a real vehicle.**")

    # Header
    st.title("⚡ ebms-sim: AI Battery Management System Simulator")
    st.caption("Physics-Based Longitudinal Vehicle Dynamics, Electrochemistry, Thermal Model & AI Surrogate for Indian E-Rickshaws")

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

        # Metric Cards
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        m_c1.metric("Specific Consumption", f"{summary['wh_per_km']:.1f} Wh/km")
        m_c2.metric("Distance Covered", f"{summary['distance_km']:.2f} km")
        m_c3.metric("Total Energy", f"{summary['total_energy_wh']:.1f} Wh")
        m_c4.metric("Final SOC", f"{summary['final_soc']*100:.1f} %", delta=f"{(summary['final_soc']-start_soc)*100:.1f}%")
        m_c5.metric("Peak Current", f"{np.max(ts_df['current_a']):.1f} A")
        m_c6.metric("Peak Pack Temp", f"{np.max(ts_df['temp_c']):.1f} °C", delta=f"{np.max(ts_df['temp_c'])-ambient_temp_c:+.1f}°C")

        if summary["cutoff_reached"]:
            st.error("⚠️ Battery reached low-SOC cutoff threshold before trip completion!")

        # Interactive Plotly Figures
        fig_trip = build_trip_simulation_figure(ts_df)
        st.plotly_chart(fig_trip, use_container_width=True)

    # =========================================================================
    # TAB 2: RANGE AND SPEED ADVISOR
    # =========================================================================
    with tabs[1]:
        st.subheader("Range Projection & Optimal Cruising Speed")

        nom_range, min_r, max_r, band_exp = calculate_range_confidence_band(
            nominal_range_km=summary["range_remaining_km"],
            ambient_temp_c=ambient_temp_c,
            soh=soh,
        )

        fig_gauge = build_range_uncertainty_gauge(nom_range, min_r, max_r)
        st.plotly_chart(fig_gauge, use_container_width=True)
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
        st.plotly_chart(fig_sweep, use_container_width=True)

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

        hills_res = compare_flat_vs_hilly_routes(
            vehicle_params=vp,
            battery_params=bp,
            thermal_params=tp,
            distance_km=distance_km,
            route_type="hill-climb" if route_type == "flat" else route_type,
            payload_mass_kg=payload_mass_kg,
            ambient_temp_c=ambient_temp_c,
            regen_efficiency=vp.eta_regen,
        )

        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        h_c1.metric("Flat Terrain", f"{hills_res['flat_wh_km']:.1f} Wh/km")
        h_c2.metric("Hilly (No Regen)", f"{hills_res['hilly_no_regen_wh_km']:.1f} Wh/km")
        h_c3.metric("Hilly (With Regen)", f"{hills_res['hilly_regen_wh_km']:.1f} Wh/km")
        h_c4.metric("Regen Savings", f"{hills_res['regen_savings_wh']:.1f} Wh", delta=f"+{hills_res['regen_range_gain_km']:.2f} km range")

        fig_hills = build_hills_comparison_bar(
            hills_res["flat_wh_km"],
            hills_res["hilly_no_regen_wh_km"],
            hills_res["hilly_regen_wh_km"],
            hills_res["regen_savings_wh"],
        )
        st.plotly_chart(fig_hills, use_container_width=True)

    # =========================================================================
    # TAB 4: SMART CHARGING
    # =========================================================================
    with tabs[3]:
        st.subheader(f"Smart Charging & Thermal Derating: {bp.chemistry.upper()}")

        c_slider_temp = st.slider("Inspect Charge Acceptance at Temperature [°C]", -5.0, 50.0, float(ambient_temp_c), 1.0)

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
        st.plotly_chart(fig_charge, use_container_width=True)

        st.markdown("### 📋 Multi-Temperature Charging Matrix")
        st.dataframe(get_charging_matrix_comparison(bp, tp, cp, start_soc=0.20, target_soc=0.95), use_container_width=True)

        st.caption(
            "ℹ️ **Electrochemical Derating Physics**: At cold temperatures (<5°C), lithium-ion cells suffer from lithium plating on the graphite anode, causing permanent dendrite degradation; charging current is therefore restricted to trickle rates. At high temperatures (>45°C for LFP, >50°C for Lead-Acid), charging ceases to prevent thermal runaway and accelerated electrolyte breakdown."
        )

    # =========================================================================
    # TAB 5: SCENARIO COMPARISON
    # =========================================================================
    with tabs[4]:
        st.subheader("Scenario A vs Scenario B Benchmarking")

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
            st.dataframe(delta_df, use_container_width=True)

            fig_comp = build_scenario_comparison_figure(sc_a["ts_df"], ts_df, label_a="Scenario A (Frozen)", label_b="Scenario B (Current)")
            st.plotly_chart(fig_comp, use_container_width=True)

    # =========================================================================
    # TAB 6: AI VS PHYSICS
    # =========================================================================
    with tabs[5]:
        st.subheader("XGBoost Neural / Gradient Boosted Surrogate vs Physics Engine")

        ai_res = run_ai_vs_physics_comparison(
            vehicle_params=vp,
            battery_params=bp,
            payload_mass_kg=payload_mass_kg,
            t_ambient_c=ambient_temp_c,
            headwind_ms=headwind_ms,
            soh=soh,
            start_soc=start_soc,
        )

        if not ai_res["available"]:
            st.warning(ai_res["message"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Physics Wh/km", f"{ai_res['physics_wh']:.1f} Wh/km", help="Full numerical integration")
            c2.metric("XGBoost Wh/km", f"{ai_res['ai_wh']:.1f} Wh/km", delta=f"{ai_res['ai_wh'] - ai_res['physics_wh']:+.2f} Wh/km")
            c3.metric("Surrogate MAPE", f"{ai_res['mape_1000']:.2f} %", help="Evaluated over 1,000 fresh random scenarios")
            c4.metric("Inference Speed-Up", f"{ai_res['speedup_factor']:.0f}x Faster", help="Microsecond evaluation on MCU")

            fig_parity = build_ai_scatter_parity_figure(ai_res["true_wh_1000"], ai_res["pred_wh_1000"], ai_res["mape_1000"])
            st.plotly_chart(fig_parity, use_container_width=True)

            st.caption("ℹ️ **Conceptual Framing**: Machine learning models in BMS serve as **fast numerical approximators** of complex multi-physics differential equations to enable real-time sub-millisecond range and state estimation on low-power 32-bit automotive microcontrollers (ESP32/STM32).")

    # =========================================================================
    # TAB 7: DATA & MODEL
    # =========================================================================
    with tabs[6]:
        st.subheader("Synthetic Dataset Distributions & Feature Importance")

        data_path = str(REPO_ROOT / "data" / "trips.parquet")
        if os.path.exists(data_path):
            df_trips = pd.read_parquet(data_path)
            st.write(f"Loaded **{len(df_trips)} synthetic trip records** from `{data_path}`.")

            st.markdown("#### Feature Distributions")
            c_hist1, c_hist2, c_hist3 = st.columns(3)
            c_hist1.bar_chart(df_trips["wh_per_km"].value_counts(bins=20).sort_index())
            c_hist2.bar_chart(df_trips["range_remaining_km"].value_counts(bins=20).sort_index())
            c_hist3.bar_chart(df_trips["ambient_temp_c"].value_counts(bins=15).sort_index())
        else:
            st.info("No generated dataset found at `data/trips.parquet`. Run `python -m ebms_sim.datagen --n 1000` to inspect dataset distributions.")

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
        st.caption("Verifies fundamental thermodynamic laws, monotonicity, and safety boundaries on current parameters.")

        checks = run_all_sanity_checks(vp, bp, tp)

        for name, passed, msg in checks:
            if passed:
                st.success(f"✅ **{name}**\n\n{msg}")
            else:
                st.error(f"❌ **{name}**\n\n{msg}")


if __name__ == "__main__":
    main()

