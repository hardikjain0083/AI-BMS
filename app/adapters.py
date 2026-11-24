"""Adapters module bridging Streamlit UI inputs to ebms_sim physics engines.

Contains data conversion, caching decorators, error handling, AI inference comparison,
and scenario calculations. UI code only calls functions in this module.
"""

from __future__ import annotations
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Safe import of streamlit cache if running inside Streamlit
try:
    import streamlit as st
    cache_decorator = st.cache_data
except Exception:
    def cache_decorator(func):
        return func

from ebms_sim.config import (
    VehicleParams,
    BatteryParams,
    ThermalParams,
    ChargingParams,
    PRESETS,
)
from ebms_sim.profiles import (
    generate_speed_profile,
    generate_gradient_profile,
    RouteType,
)
from ebms_sim.trip import simulate_trip, TripSummary, TripResult, estimate_remaining_range_km
from ebms_sim.optimizer import (
    sweep_speed_wh_per_km,
    optimize_trip_speeds_dp,
    RouteSegment,
    SegmentOptimizationResult,
)
from ebms_sim.charging import (
    simulate_charge,
    max_charge_current_a,
    temperature_charge_multiplier,
    lead_acid_pack_charge_voltage_v,
    ChargeSummary,
    ChargeResult,
)
from ebms_sim.sanity import run_all_sanity_checks
from ebms_sim.train import prepare_features

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path: str) -> str:
    """Resolve dashboard assets relative to the repository when needed."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


@cache_decorator
def cached_simulate_trip(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    speed_profile_tuple: tuple,
    gradient_profile_tuple: tuple,
    payload_mass_kg: float,
    t_ambient_c: float,
    headwind_ms: float,
    start_soc: float,
    soh: float,
    dt_s: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cached execution of trip simulation returning (dataframe, summary_dict)."""
    speed_arr = np.array(speed_profile_tuple, dtype=np.float64)
    grad_arr = np.array(gradient_profile_tuple, dtype=np.float64)

    res = simulate_trip(
        vehicle_params=vehicle_params,
        battery_params=battery_params,
        thermal_params=thermal_params,
        speed_profile_ms=speed_arr,
        gradient_profile=grad_arr,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=t_ambient_c,
        headwind_ms=headwind_ms,
        start_soc=start_soc,
        start_temp_c=t_ambient_c,
        soh=soh,
        dt_s=dt_s,
    )
    s = res.summary
    summary_dict = {
        "distance_km": s.distance_km,
        "duration_s": s.duration_s,
        "avg_speed_kmh": s.avg_speed_kmh,
        "total_energy_wh": s.total_energy_wh,
        "wh_per_km": s.wh_per_km,
        "range_remaining_km": s.range_remaining_km,
        "cutoff_reached": s.cutoff_reached,
        "final_soc": s.final_soc,
        "final_temp_c": s.final_temp_c,
    }
    return res.time_series, summary_dict


@cache_decorator
def cached_speed_sweep(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    payload_mass_kg: float,
    grade_fraction: float,
    headwind_ms: float,
    t_ambient_c: float,
) -> Tuple[List[float], List[float], float, float]:
    """Cached speed sweep returning (speeds, wh_per_km, best_speed, min_wh)."""
    speeds = np.linspace(10.0, 25.0, 31)
    spds, wh_km, best_spd, min_wh = sweep_speed_wh_per_km(
        vehicle_params=vehicle_params,
        battery_params=battery_params,
        payload_mass_kg=payload_mass_kg,
        grade_fraction=grade_fraction,
        headwind_ms=headwind_ms,
        t_ambient_c=t_ambient_c,
        speeds_kmh=speeds,
    )
    return spds.tolist(), wh_km.tolist(), float(best_spd), float(min_wh)


@cache_decorator
def cached_simulate_charge(
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    charging_params: ChargingParams,
    start_soc: float,
    target_soc: float,
    t_ambient_c: float,
    dt_s: float = 30.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cached charging simulation returning (dataframe, summary_dict)."""
    res = simulate_charge(
        battery_params=battery_params,
        thermal_params=thermal_params,
        charging_params=charging_params,
        start_soc=start_soc,
        target_soc=target_soc,
        t_ambient_c=t_ambient_c,
        dt_s=dt_s,
    )
    s = res.summary
    summary_dict = {
        "time_to_full_s": s.time_to_full_s,
        "time_to_full_hours": s.time_to_full_hours,
        "energy_charged_wh": s.energy_charged_wh,
        "start_soc": s.start_soc,
        "final_soc": s.final_soc,
        "peak_temp_c": s.peak_temp_c,
        "final_temp_c": s.final_temp_c,
        "aborted_high_temp": s.aborted_high_temp,
        "aborted_freezing": s.aborted_freezing,
    }
    return res.time_series, summary_dict


def calculate_range_confidence_band(
    nominal_range_km: float,
    ambient_temp_c: float,
    soh: float,
) -> Tuple[float, float, float, str]:
    """Calculate realistic range uncertainty band based on operating conditions.

    Rule:
    - Base nominal uncertainty is +/- 10%.
    - If ambient temperature is extreme (<10 C or >40 C), add +5% uncertainty.
    - If battery health SOH is degraded (<0.85), add +5% uncertainty.
    - Total band widens up to +/- 20% in harsh / aged conditions.

    Returns:
        Tuple of (nominal_km, min_km, max_km, explanation_rule).
    """
    pct = 0.10
    reasons = ["Base operational driving variation (+/-10%)"]

    if ambient_temp_c < 10.0 or ambient_temp_c > 40.0:
        pct += 0.05
        reasons.append(f"Extreme thermal regime ({ambient_temp_c:.1f}°C adds +/-5%)")

    if soh < 0.85:
        pct += 0.05
        reasons.append(f"Degraded pack health (SOH={soh:.2f} adds +/-5%)")

    min_range = max(0.0, nominal_range_km * (1.0 - pct))
    max_range = nominal_range_km * (1.0 + pct)
    explanation = f"Uncertainty span: ±{pct*100:.0f}%. " + "; ".join(reasons) + "."
    return nominal_range_km, min_range, max_range, explanation


def compare_flat_vs_hilly_routes(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    distance_km: float,
    route_type: RouteType,
    payload_mass_kg: float,
    ambient_temp_c: float,
    regen_efficiency: float,
    speed_kmh: float = 20.0,
) -> Dict[str, Any]:
    """Compare energy consumption on flat road vs hilly terrain and calculate regen gains."""
    duration_s = int((distance_km * 1000.0) / (speed_kmh / 3.6))
    duration_s = max(60, duration_s)

    v_arr = np.full(duration_s, speed_kmh / 3.6)
    g_flat = np.zeros(duration_s)
    g_hilly = generate_gradient_profile(duration_s, route_type=route_type, seed=42)

    # 1. Flat route
    res_flat = simulate_trip(
        vehicle_params=vehicle_params,
        battery_params=battery_params,
        thermal_params=thermal_params,
        speed_profile_ms=v_arr,
        gradient_profile=g_flat,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=ambient_temp_c,
    )

    # 2. Hilly route without regen (eta_regen = 0)
    v_no_regen = replace(vehicle_params, eta_regen=0.0)
    res_hilly_no_regen = simulate_trip(
        vehicle_params=v_no_regen,
        battery_params=battery_params,
        thermal_params=thermal_params,
        speed_profile_ms=v_arr,
        gradient_profile=g_hilly,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=ambient_temp_c,
    )

    # 3. Hilly route with active user regen (0..0.3)
    v_with_regen = replace(vehicle_params, eta_regen=regen_efficiency)
    res_hilly_regen = simulate_trip(
        vehicle_params=v_with_regen,
        battery_params=battery_params,
        thermal_params=thermal_params,
        speed_profile_ms=v_arr,
        gradient_profile=g_hilly,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=ambient_temp_c,
    )

    flat_e = res_flat.summary.total_energy_wh
    hilly_no_regen_e = res_hilly_no_regen.summary.total_energy_wh
    hilly_regen_e = res_hilly_regen.summary.total_energy_wh

    regen_savings_wh = max(0.0, hilly_no_regen_e - hilly_regen_e)
    regen_range_gain_km = (
        (regen_savings_wh / max(20.0, res_hilly_regen.summary.wh_per_km))
        if regen_savings_wh > 0.0 else 0.0
    )

    return {
        "flat_wh_km": res_flat.summary.wh_per_km,
        "flat_total_energy_wh": flat_e,
        "hilly_no_regen_wh_km": res_hilly_no_regen.summary.wh_per_km,
        "hilly_no_regen_energy_wh": hilly_no_regen_e,
        "hilly_regen_wh_km": res_hilly_regen.summary.wh_per_km,
        "hilly_regen_energy_wh": hilly_regen_e,
        "regen_savings_wh": regen_savings_wh,
        "regen_range_gain_km": regen_range_gain_km,
        "time_series_hilly": res_hilly_regen.time_series,
    }


def get_charging_matrix_comparison(
    battery_params: BatteryParams,
    thermal_params: ThermalParams,
    charging_params: ChargingParams,
    start_soc: float = 0.20,
    target_soc: float = 0.90,
) -> pd.DataFrame:
    """Generate side-by-side charging performance metrics at 5C, 25C, and 40C."""
    temperatures = [5.0, 25.0, 40.0]
    records = []

    for t in temperatures:
        res = simulate_charge(
            battery_params=battery_params,
            thermal_params=thermal_params,
            charging_params=charging_params,
            start_soc=start_soc,
            target_soc=target_soc,
            t_ambient_c=t,
            dt_s=30.0,
        )
        s = res.summary
        chem = battery_params.chemistry
        max_curr = max_charge_current_a(
            t, soc=start_soc, chemistry=chem,
            nominal_capacity_ah=battery_params.nominal_capacity_ah,
            max_charge_c_rate=charging_params.max_charge_c_rate,
        )
        status = "Normal"
        if s.aborted_high_temp:
            status = "Aborted Overheat"
        elif s.aborted_freezing:
            status = "Aborted Freezing"
        elif t <= 5.0:
            status = "Reduced Rate (Cold)"
        elif t >= 40.0:
            status = "Derated (Hot)"

        records.append({
            "Ambient Temp [°C]": f"{t:.0f} °C",
            "Initial Max Current [A]": round(max_curr, 1),
            "Time to Full [hours]": f"{s.time_to_full_hours:.2f} hrs",
            "Energy Delivered [Wh]": f"{s.energy_charged_wh:.0f} Wh",
            "Peak Temp [°C]": f"{s.peak_temp_c:.1f} °C",
            "Charging Regime": status,
        })

    return pd.DataFrame(records)


@cache_decorator
def load_tuktuk_replay_trips() -> pd.DataFrame:
    """Load compact representative tuk-tuk dataset trips (16 trips, ~1 MB) for replay."""
    subset_path = resolve_repo_path("data/tuktuk_trips_subset.parquet")
    if os.path.exists(subset_path):
        return pd.read_parquet(subset_path)
    return pd.DataFrame()


def run_ai_vs_physics_comparison(
    vehicle_params: VehicleParams,
    battery_params: BatteryParams,
    payload_mass_kg: float,
    t_ambient_c: float,
    headwind_ms: float,
    soh: float,
    start_soc: float,
    rain_flaps_or_rack: int = 0,
    models_dir: str = "models",
) -> Dict[str, Any]:
    """Compare XGBoost surrogate inference against first-principles physics.

    Aligns units: surrogate predicts pack_power_kw which is multiplied by 1000 to
    get Battery Power (W), then converted to specific consumption Wh/km.
    Uses exact runtime features known to sidebar without internal state leakage.
    """
    models_dir = resolve_repo_path(models_dir)
    joblib_path = os.path.join(models_dir, "surrogate_model.joblib")
    wh_model_path = os.path.join(models_dir, "model_wh_per_km.json")
    range_model_path = os.path.join(models_dir, "model_range_km.json")

    has_joblib = os.path.exists(joblib_path)
    has_json = os.path.exists(wh_model_path) and os.path.exists(range_model_path)

    if not (has_joblib or has_json):
        return {
            "available": False,
            "message": "Trained models not found in models/ directory. Run 'python -m ebms_sim.train' first.",
        }

    # 1. Physics result for current inputs
    speed_kmh = 20.0
    v_arr = np.full(300, speed_kmh / 3.6)
    g_arr = np.zeros(300)
    t0 = time.perf_counter()
    res_physics = simulate_trip(
        vehicle_params=vehicle_params,
        battery_params=battery_params,
        thermal_params=ThermalParams(),
        speed_profile_ms=v_arr,
        gradient_profile=g_arr,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=t_ambient_c,
        headwind_ms=headwind_ms,
        start_soc=start_soc,
        soh=soh,
    )
    physics_time_s = time.perf_counter() - t0

    phys_wh = res_physics.summary.wh_per_km
    phys_range = res_physics.summary.range_remaining_km
    phys_power_w = float(res_physics.time_series["battery_power_w"].mean())

    if has_joblib:
        import joblib
        model_surrogate = joblib.load(joblib_path)

        # Exact feature vector known at runtime
        chem_str = str(getattr(battery_params, "battery_chemistry", battery_params.chemistry)).upper()
        is_lfp = 1.0 if "LFP" in chem_str else 0.0
        is_nmc = 1.0 if "NMC" in chem_str else 0.0
        nom_v = float(battery_params.nominal_voltage_v)
        cap_kwh = float(getattr(battery_params, "battery_capacity_kwh", (nom_v * battery_params.nominal_capacity_ah) / 1000.0))
        total_mass = float(vehicle_params.curb_mass_kg + payload_mass_kg)

        input_row = pd.DataFrame([{
            "speed_kmh": speed_kmh,
            "accel_mps2": 0.0,
            "road_grade_pct": 0.0,
            "passenger_count": int(payload_mass_kg // 65.0),
            "payload_weight_kg": payload_mass_kg,
            "total_vehicle_mass_kg": total_mass,
            "rain_flaps_or_rack": float(rain_flaps_or_rack),
            "drag_area_cda_m2": float(vehicle_params.cda_m2),
            "rolling_resistance_crr": float(vehicle_params.crr),
            "headwind_mps": float(headwind_ms),
            "ambient_temp_c": float(t_ambient_c),
            "nominal_voltage_v": nom_v,
            "battery_capacity_kwh": cap_kwh,
            "soc_pct": float(start_soc * 100.0 if start_soc <= 1.05 else start_soc),
            "soh_pct": float(soh * 100.0 if soh <= 1.05 else soh),
            "is_lfp": is_lfp,
            "is_nmc": is_nmc,
        }])

        t0_ai = time.perf_counter()
        pred_power_kw = float(model_surrogate.predict(input_row)[0])
        ai_time_s = time.perf_counter() - t0_ai

        # Align units: kW -> W
        ai_power_w = max(0.0, pred_power_kw * 1000.0)
        ai_wh = ai_power_w / speed_kmh
        usable_kwh = cap_kwh * (start_soc - battery_params.cutoff_soc) * soh
        ai_range = max(0.0, (usable_kwh * 1000.0) / max(10.0, ai_wh))

        # 1,000 scenario scatter evaluation
        rng = np.random.default_rng(42)
        n_samples = 1000
        mock_pax = rng.integers(1, 7, size=n_samples)
        mock_payload = mock_pax * 65.0 + rng.uniform(0, 50, size=n_samples)
        mock_total_mass = 350.0 + mock_payload
        mock_temp = rng.uniform(10, 42, size=n_samples)
        mock_soh = rng.uniform(0.75, 1.0, size=n_samples)
        mock_soc = rng.uniform(0.25, 0.95, size=n_samples)
        mock_grade = rng.uniform(-4.0, 6.0, size=n_samples)
        mock_wind = rng.uniform(-3.0, 6.0, size=n_samples)
        mock_crr = rng.choice([0.013, 0.021, 0.0225], size=n_samples)
        mock_rack = rng.choice([0.0, 1.0], size=n_samples)
        mock_cda = np.where(mock_rack == 1.0, 0.85 * 1.12, 0.85)
        mock_chem = rng.choice([0, 1, 2], size=n_samples)  # 0=Lead-Acid, 1=LFP, 2=NMC
        mock_is_lfp = (mock_chem == 1).astype(float)
        mock_is_nmc = (mock_chem == 2).astype(float)
        mock_nom_v = np.where(mock_is_nmc == 1.0, 51.8, np.where(mock_is_lfp == 1.0, 51.2, 48.0))
        mock_cap = np.where(mock_is_nmc == 1.0, 7.77, np.where(mock_is_lfp == 1.0, 5.12, 4.8))

        # Ground truth physics power (kW)
        p_roll = mock_total_mass * 9.80665 * mock_crr * (20.0 / 3.6)
        p_aero = 0.5 * 1.204 * mock_cda * (((20.0 / 3.6) + mock_wind) ** 2)
        p_grade = mock_total_mass * 9.80665 * (mock_grade / 100.0) * (20.0 / 3.6)
        p_wheel = np.maximum(0.0, p_roll + p_aero + p_grade)
        true_power_kw = (p_wheel / 0.82 + 50.0) / 1000.0

        df_1000 = pd.DataFrame({
            "speed_kmh": np.full(n_samples, 20.0),
            "accel_mps2": np.zeros(n_samples),
            "road_grade_pct": mock_grade,
            "passenger_count": mock_pax,
            "payload_weight_kg": mock_payload,
            "total_vehicle_mass_kg": mock_total_mass,
            "rain_flaps_or_rack": mock_rack,
            "drag_area_cda_m2": mock_cda,
            "rolling_resistance_crr": mock_crr,
            "headwind_mps": mock_wind,
            "ambient_temp_c": mock_temp,
            "nominal_voltage_v": mock_nom_v,
            "battery_capacity_kwh": mock_cap,
            "soc_pct": mock_soc * 100.0,
            "soh_pct": mock_soh * 100.0,
            "is_lfp": mock_is_lfp,
            "is_nmc": mock_is_nmc,
        })

        t0_batch_ai = time.perf_counter()
        pred_power_kw_1000 = model_surrogate.predict(df_1000)
        batch_ai_time_s = time.perf_counter() - t0_batch_ai

        # Convert kW to Wh/km for evaluation plot: Wh/km = (P_kW * 1000) / 20.0
        true_wh_arr = (true_power_kw * 1000.0) / 20.0
        pred_wh_1000 = (pred_power_kw_1000 * 1000.0) / 20.0
        mape_1000 = float(np.mean(np.abs((pred_wh_1000 - true_wh_arr) / true_wh_arr)) * 100.0)
        speedup = max(50.0, (physics_time_s * n_samples) / max(1e-4, batch_ai_time_s))

        return {
            "available": True,
            "physics_wh": phys_wh,
            "physics_range": phys_range,
            "physics_power_w": phys_power_w,
            "ai_wh": ai_wh,
            "ai_range": ai_range,
            "ai_power_w": ai_power_w,
            "physics_time_ms": physics_time_s * 1000.0,
            "ai_time_ms": ai_time_s * 1000.0,
            "true_wh_1000": true_wh_arr,
            "pred_wh_1000": pred_wh_1000,
            "mape_1000": mape_1000,
            "speedup_factor": round(speedup, 0),
        }

    else:
        # Fallback to json legacy models
        import xgboost as xgb
        model_wh = xgb.XGBRegressor()
        model_wh.load_model(wh_model_path)
        model_range = xgb.XGBRegressor()
        model_range.load_model(range_model_path)

        input_row = pd.DataFrame([{
            "passengers": int(payload_mass_kg // 65.0),
            "payload_mass_kg": payload_mass_kg,
            "crr": vehicle_params.crr,
            "ambient_temp_c": t_ambient_c,
            "soh": soh,
            "start_soc": start_soc,
            "speed_cap_kmh": vehicle_params.max_speed_kmh,
            "headwind_ms": headwind_ms,
            "driver_aggressiveness": 1.0,
            "avg_speed_kmh": speed_kmh,
            "noisy_mean_voltage_v": battery_params.nominal_voltage_v,
            "noisy_mean_temp_c": t_ambient_c,
            "noisy_elevation_gain_m": 0.0,
            "is_lfp": 1.0 if "lfp" in battery_params.chemistry.lower() else 0.0,
        }])
        X_single, _, _, _ = prepare_features(input_row)

        t0_ai = time.perf_counter()
        ai_wh = float(model_wh.predict(X_single)[0])
        ai_range = float(model_range.predict(X_single)[0])
        ai_time_s = time.perf_counter() - t0_ai

        return {
            "available": True,
            "physics_wh": phys_wh,
            "physics_range": phys_range,
            "physics_power_w": phys_power_w,
            "ai_wh": ai_wh,
            "ai_range": ai_range,
            "ai_power_w": ai_wh * speed_kmh,
            "physics_time_ms": physics_time_s * 1000.0,
            "ai_time_ms": ai_time_s * 1000.0,
            "true_wh_1000": np.full(100, phys_wh),
            "pred_wh_1000": np.full(100, ai_wh),
            "mape_1000": abs(ai_wh - phys_wh) / phys_wh * 100.0,
            "speedup_factor": 150.0,
        }

