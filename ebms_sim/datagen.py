"""Synthetic dataset generator CLI and batch simulation engine for e-rickshaw BMS.

Generates trip-level and 30-second-window-level datasets with realistic parameter
distributions and sensor noise for training machine learning models.
"""

from __future__ import annotations
import argparse
import os
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from ebms_sim.config import PRESETS, VehicleParams, BatteryParams, ThermalParams
from ebms_sim.profiles import (
    generate_speed_profile,
    generate_gradient_profile,
    RouteType,
)
from ebms_sim.trip import simulate_trip


def add_sensor_noise(
    voltage_series: np.ndarray,
    temp_series: np.ndarray,
    elevation_series: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add realistic physical sensor noise to measurements.

    - Voltage noise: +/- 0.5% (ADC tolerance and harness ripple).
    - Temperature noise: +/- 1.0 deg C (NTC thermistor quantization and tolerance).
    - Altitude noise: +/- 3.0 m (Barometric altimeter / GPS vertical jitter).

    Args:
        voltage_series: True terminal voltage array [V].
        temp_series: True pack temperature array [deg C].
        elevation_series: True cumulative elevation array [m].
        rng: NumPy random generator.

    Returns:
        Tuple of (noisy_voltage, noisy_temp, noisy_elevation).
    """
    # Multiplicative +/- 0.5% uniform noise
    v_noise = rng.uniform(-0.005, 0.005, size=len(voltage_series))
    noisy_voltage = voltage_series * (1.0 + v_noise)

    # Additive +/- 1.0 C uniform noise
    t_noise = rng.uniform(-1.0, 1.0, size=len(temp_series))
    noisy_temp = temp_series + t_noise

    # Additive +/- 3.0 m uniform noise
    z_noise = rng.uniform(-3.0, 3.0, size=len(elevation_series))
    noisy_elevation = elevation_series + z_noise

    return noisy_voltage, noisy_temp, noisy_elevation


def generate_single_synthetic_trip(
    trip_id: int,
    seed: int,
    trip_duration_s: int = 600,
    n_unique_routes: int = 50,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Simulate a single parameterized trip and extract trip-level & window-level records.

    Args:
        trip_id: Unique integer trip index.
        seed: Random seed for this specific trip.
        trip_duration_s: Duration of trip simulation in seconds [s].
        n_unique_routes: Distinct route categories for grouped splitting.

    Returns:
        Tuple of (trip_level_record, list_of_30s_window_records).
    """
    rng = np.random.default_rng(seed)

    # 1. Randomize vehicle & trip parameters
    passengers = int(rng.integers(0, 6))  # 0 to 5 passengers
    luggage_kg = float(rng.uniform(0.0, 80.0))
    payload_mass_kg = passengers * 65.0 + luggage_kg

    crr = float(rng.uniform(0.012, 0.025))
    ambient_c = float(rng.uniform(5.0, 45.0))
    soh = float(rng.uniform(0.70, 1.00))
    start_soc = float(rng.uniform(0.20, 1.00))
    speed_cap_kmh = float(rng.uniform(15.0, 25.0))
    headwind_ms = float(rng.uniform(-3.0, 3.0))
    aggressiveness = float(rng.uniform(0.6, 1.5))

    route_types: list[RouteType] = ["flat", "rolling", "hill-climb", "hill-descent", "mixed"]
    route_type = str(rng.choice(route_types))
    route_id = int(rng.integers(1, n_unique_routes + 1))

    # Chemistry preset
    chemistry = str(rng.choice(["lead_acid_48V_100Ah", "lfp_51V_60Ah"]))
    preset = PRESETS[chemistry]

    # Custom vehicle params with randomized Crr
    v_params = VehicleParams(
        curb_mass_kg=preset["vehicle"].curb_mass_kg,
        driver_mass_kg=preset["vehicle"].driver_mass_kg,
        crr=crr,
        cda_m2=preset["vehicle"].cda_m2,
        eta_drive=preset["vehicle"].eta_drive,
        eta_regen=preset["vehicle"].eta_regen,
        aux_w=preset["vehicle"].aux_w,
        rotational_inertia_factor=preset["vehicle"].rotational_inertia_factor,
        max_speed_kmh=speed_cap_kmh,
    )
    b_params = preset["battery"]
    t_params = preset["thermal"]

    # 2. Generate profiles
    v_profile = generate_speed_profile(
        duration_s=trip_duration_s,
        speed_cap_kmh=speed_cap_kmh,
        stop_interval_s=float(rng.uniform(40.0, 90.0)),
        stop_duration_s=float(rng.uniform(6.0, 15.0)),
        aggressiveness=aggressiveness,
        seed=int(rng.integers(0, 10000000)),
    )
    g_profile = generate_gradient_profile(
        length_steps=trip_duration_s,
        route_type=route_type,  # type: ignore
        seed=int(rng.integers(0, 10000000)),
    )

    # 3. Simulate trip
    sim = simulate_trip(
        vehicle_params=v_params,
        battery_params=b_params,
        thermal_params=t_params,
        speed_profile_ms=v_profile,
        gradient_profile=g_profile,
        payload_mass_kg=payload_mass_kg,
        t_ambient_c=ambient_c,
        headwind_ms=headwind_ms,
        start_soc=start_soc,
        start_temp_c=ambient_c,
        soh=soh,
    )

    ts = sim.time_series
    summary = sim.summary

    # 4. Add sensor noise
    noisy_v, noisy_t, noisy_z = add_sensor_noise(
        voltage_series=ts["voltage_v"].values,
        temp_series=ts["temp_c"].values,
        elevation_series=ts["elevation_m"].values,
        rng=rng,
    )
    ts["noisy_voltage_v"] = noisy_v
    ts["noisy_temp_c"] = noisy_t
    ts["noisy_elevation_m"] = noisy_z

    # 5. Build Trip-level record
    trip_record = {
        "trip_id": trip_id,
        "route_id": route_id,
        "route_type": route_type,
        "chemistry": b_params.chemistry,
        "passengers": passengers,
        "luggage_kg": round(luggage_kg, 1),
        "payload_mass_kg": round(payload_mass_kg, 1),
        "crr": round(crr, 4),
        "ambient_temp_c": round(ambient_c, 1),
        "soh": round(soh, 3),
        "start_soc": round(start_soc, 3),
        "final_soc": round(summary.final_soc, 3),
        "delta_soc": round(start_soc - summary.final_soc, 4),
        "speed_cap_kmh": round(speed_cap_kmh, 1),
        "headwind_ms": round(headwind_ms, 2),
        "driver_aggressiveness": round(aggressiveness, 2),
        "duration_s": summary.duration_s,
        "distance_km": round(summary.distance_km, 3),
        "avg_speed_kmh": round(summary.avg_speed_kmh, 2),
        "elevation_gain_m": round(float(ts["elevation_m"].iloc[-1] - ts["elevation_m"].iloc[0]), 2),
        "mean_voltage_v": round(float(np.mean(ts["voltage_v"])), 2),
        "mean_current_a": round(float(np.mean(ts["current_a"])), 2),
        "mean_temp_c": round(float(np.mean(ts["temp_c"])), 2),
        "noisy_mean_voltage_v": round(float(np.mean(noisy_v)), 2),
        "noisy_mean_temp_c": round(float(np.mean(noisy_t)), 2),
        "noisy_elevation_gain_m": round(float(noisy_z[-1] - noisy_z[0]), 2),
        "total_energy_wh": round(summary.total_energy_wh, 2),
        "wh_per_km": round(summary.wh_per_km, 2),
        "range_remaining_km": round(summary.range_remaining_km, 2),
    }

    # 6. Build 30-second window slices
    window_records = []
    window_size_s = 30
    n_windows = trip_duration_s // window_size_s

    for w in range(n_windows):
        start_idx = w * window_size_s
        end_idx = start_idx + window_size_s
        sub = ts.iloc[start_idx:end_idx]

        w_dist_m = float(sub["distance_m"].iloc[-1] - sub["distance_m"].iloc[0])
        w_energy_wh = float(sub["energy_wh"].iloc[-1] - sub["energy_wh"].iloc[0])
        w_dist_km = w_dist_m / 1000.0
        w_wh_km = (w_energy_wh / w_dist_km) if w_dist_km > 0.01 else 0.0

        w_rec = {
            "trip_id": trip_id,
            "route_id": route_id,
            "window_idx": w,
            "window_start_s": start_idx,
            "window_end_s": end_idx,
            "route_type": route_type,
            "chemistry": b_params.chemistry,
            "mean_speed_kmh": round(float(np.mean(sub["speed_kmh"])), 2),
            "std_speed_kmh": round(float(np.std(sub["speed_kmh"])), 2),
            "mean_accel_ms2": round(float(np.mean(sub["accel_ms2"])), 3),
            "mean_gradient": round(float(np.mean(sub["gradient"])), 4),
            "elevation_change_m": round(float(sub["elevation_m"].iloc[-1] - sub["elevation_m"].iloc[0]), 2),
            "mean_power_w": round(float(np.mean(sub["battery_power_w"])), 1),
            "mean_current_a": round(float(np.mean(sub["current_a"])), 2),
            "mean_voltage_v": round(float(np.mean(sub["voltage_v"])), 2),
            "mean_temp_c": round(float(np.mean(sub["temp_c"])), 2),
            "start_window_soc": round(float(sub["soc"].iloc[0]), 4),
            "end_window_soc": round(float(sub["soc"].iloc[-1]), 4),
            "delta_window_soc": round(float(sub["soc"].iloc[0] - sub["soc"].iloc[-1]), 5),
            "window_energy_wh": round(w_energy_wh, 2),
            "window_distance_m": round(w_dist_m, 1),
            "window_wh_per_km": round(w_wh_km, 2),
            "noisy_mean_voltage_v": round(float(np.mean(sub["noisy_voltage_v"])), 2),
            "noisy_mean_temp_c": round(float(np.mean(sub["noisy_temp_c"])), 2),
            "noisy_elevation_change_m": round(float(sub["noisy_elevation_m"].iloc[-1] - sub["noisy_elevation_m"].iloc[0]), 2),
            "ambient_temp_c": round(ambient_c, 1),
            "soh": round(soh, 3),
            "passengers": passengers,
            "payload_mass_kg": round(payload_mass_kg, 1),
            "crr": round(crr, 4),
            "wh_per_km_target": round(summary.wh_per_km, 2),
            "range_remaining_km_target": round(summary.range_remaining_km, 2),
        }
        window_records.append(w_rec)

    return trip_record, window_records


def _worker_generate_chunk(args: Tuple[List[int], List[int], int, int]) -> Tuple[List[Dict], List[Dict]]:
    """Worker function for multiprocessing."""
    trip_ids, seeds, trip_dur_s, n_routes = args
    chunk_trips = []
    chunk_windows = []
    for tid, s in zip(trip_ids, seeds):
        t_rec, w_recs = generate_single_synthetic_trip(
            trip_id=tid,
            seed=s,
            trip_duration_s=trip_dur_s,
            n_unique_routes=n_routes,
        )
        chunk_trips.append(t_rec)
        chunk_windows.extend(w_recs)
    return chunk_trips, chunk_windows


def generate_dataset(
    n_trips: int = 100,
    trip_duration_s: int = 300,
    seed: int = 42,
    n_workers: Optional[int] = None,
    n_unique_routes: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both trip-level and 30-second-window datasets using multiprocessing.

    Args:
        n_trips: Total number of trips to simulate.
        trip_duration_s: Duration per trip in seconds [s].
        seed: Master random seed.
        n_workers: Number of CPU processes (defaults to available logical CPUs).
        n_unique_routes: Distinct route IDs for group splitting.

    Returns:
        Tuple of (df_trips, df_windows).
    """
    rng = np.random.default_rng(seed)
    trip_ids = list(range(1, n_trips + 1))
    seeds = [int(s) for s in rng.integers(0, 1000000000, size=n_trips)]

    workers = n_workers if n_workers is not None else min(8, max(1, os.cpu_count() or 1))

    if workers <= 1 or n_trips < 4:
        # Sequential execution
        all_trips = []
        all_windows = []
        for tid, s in zip(trip_ids, seeds):
            t_rec, w_recs = generate_single_synthetic_trip(tid, s, trip_duration_s, n_unique_routes)
            all_trips.append(t_rec)
            all_windows.extend(w_recs)
    else:
        # Split into worker chunks
        chunks = []
        chunk_size = (n_trips + workers - 1) // workers
        for i in range(workers):
            start = i * chunk_size
            end = min(n_trips, (i + 1) * chunk_size)
            if start < end:
                chunks.append((trip_ids[start:end], seeds[start:end], trip_duration_s, n_unique_routes))

        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker_generate_chunk, chunks)

        all_trips = []
        all_windows = []
        for t_chunk, w_chunk in results:
            all_trips.extend(t_chunk)
            all_windows.extend(w_chunk)

    df_trips = pd.DataFrame(all_trips)
    df_windows = pd.DataFrame(all_windows)
    return df_trips, df_windows


def save_tables(
    df_trips: pd.DataFrame,
    df_windows: pd.DataFrame,
    out_path: str,
    out_windows_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Save trip and window DataFrames to Parquet or CSV.

    Args:
        df_trips: Trip-level DataFrame.
        df_windows: 30-second window DataFrame.
        out_path: Path for trips table.
        out_windows_path: Optional path for windows table.

    Returns:
        Tuple of (saved_trips_path, saved_windows_path).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if out_windows_path is None:
        base, ext = os.path.splitext(out_path)
        out_windows_path = f"{base}_windows{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(out_windows_path)), exist_ok=True)

    # Choose format by extension
    is_parquet = out_path.endswith(".parquet")
    if is_parquet:
        try:
            df_trips.to_parquet(out_path, index=False)
            df_windows.to_parquet(out_windows_path, index=False)
        except Exception:
            # Fallback to CSV if parquet engine unavailable
            csv_trips = out_path.replace(".parquet", ".csv")
            csv_win = out_windows_path.replace(".parquet", ".csv")
            df_trips.to_csv(csv_trips, index=False)
            df_windows.to_csv(csv_win, index=False)
            return csv_trips, csv_win
    else:
        df_trips.to_csv(out_path, index=False)
        df_windows.to_csv(out_windows_path, index=False)

    return out_path, out_windows_path


def main():
    """CLI Entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic BMS datasets for Indian e-rickshaws.")
    parser.add_argument("--n", type=int, default=100, help="Number of trips to generate (default: 100)")
    parser.add_argument("--out", type=str, default="data/trips.parquet", help="Output path for trip-level table")
    parser.add_argument("--out-windows", type=str, default=None, help="Output path for 30s-window table")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers")
    parser.add_argument("--duration", type=int, default=300, help="Duration per trip in seconds (default: 300)")

    args = parser.parse_args()

    print(f"Generating {args.n} synthetic trips (duration={args.duration}s, seed={args.seed})...")
    df_trips, df_windows = generate_dataset(
        n_trips=args.n,
        trip_duration_s=args.duration,
        seed=args.seed,
        n_workers=args.workers,
    )

    t_out, w_out = save_tables(df_trips, df_windows, args.out, args.out_windows)
    print(f"Successfully generated:")
    print(f"  - Trip-level table:   {t_out} ({len(df_trips)} rows)")
    print(f"  - Window-level table: {w_out} ({len(df_windows)} rows)")


if __name__ == "__main__":
    main()

