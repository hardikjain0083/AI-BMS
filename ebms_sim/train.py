"""Machine learning model training for e-rickshaw energy consumption and range.

Trains XGBoost regression models for Wh/km and range_km using GroupKFold
grouped by route_id to prevent spatial leakage, reports MAE and MAPE,
and evaluates performance on held-out temperature bands.
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb


# Exact feature vector known at runtime by the simulator/sidebar for pack power surrogate
RUNTIME_FEATURE_COLUMNS = [
    "speed_kmh",
    "accel_mps2",
    "road_grade_pct",
    "passenger_count",
    "payload_weight_kg",
    "total_vehicle_mass_kg",
    "rain_flaps_or_rack",
    "drag_area_cda_m2",
    "rolling_resistance_crr",
    "headwind_mps",
    "ambient_temp_c",
    "nominal_voltage_v",
    "battery_capacity_kwh",
    "soc_pct",
    "soh_pct",
    "is_lfp",
    "is_nmc",
]


FEATURE_COLUMNS = [
    "passengers",
    "payload_mass_kg",
    "crr",
    "ambient_temp_c",
    "soh",
    "start_soc",
    "speed_cap_kmh",
    "headwind_ms",
    "driver_aggressiveness",
    "avg_speed_kmh",
    "noisy_mean_voltage_v",
    "noisy_mean_temp_c",
    "noisy_elevation_gain_m",
    "is_lfp",
]


def prepare_runtime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare runtime driving condition feature matrix for pack_power_kw prediction.

    Crucially excludes internal electrical state outputs (pack_current_a, pack_voltage_v,
    internal_resistance_mohm) to prevent feature leakage, matching only what the
    sidebar and vehicle physics actually know at runtime.

    Args:
        df: DataFrame containing trip telemetry or runtime inputs.

    Returns:
        Tuple of (X, y_power_kw, groups_trip_id).
    """
    df_clean = df.copy()

    # Chemistry indicators (1 for LFP, 1 for NMC, 0 for Lead-Acid)
    if "is_lfp" not in df_clean.columns:
        if "battery_chemistry" in df_clean.columns:
            df_clean["is_lfp"] = (df_clean["battery_chemistry"].astype(str).str.upper() == "LFP").astype(float)
        elif "chemistry" in df_clean.columns:
            df_clean["is_lfp"] = (df_clean["chemistry"].astype(str).str.lower() == "lfp").astype(float)
        else:
            df_clean["is_lfp"] = 0.0

    if "is_nmc" not in df_clean.columns:
        if "battery_chemistry" in df_clean.columns:
            df_clean["is_nmc"] = (df_clean["battery_chemistry"].astype(str).str.upper() == "NMC").astype(float)
        elif "chemistry" in df_clean.columns:
            df_clean["is_nmc"] = (df_clean["chemistry"].astype(str).str.lower() == "nmc").astype(float)
        else:
            df_clean["is_nmc"] = 0.0

    # Map aliases across dataset schema and runtime inputs
    if "headwind_mps" not in df_clean.columns and "headwind_ms" in df_clean.columns:
        df_clean["headwind_mps"] = df_clean["headwind_ms"]
    if "rolling_resistance_crr" not in df_clean.columns and "crr" in df_clean.columns:
        df_clean["rolling_resistance_crr"] = df_clean["crr"]
    if "drag_area_cda_m2" not in df_clean.columns and "cda_m2" in df_clean.columns:
        df_clean["drag_area_cda_m2"] = df_clean["cda_m2"]
    if "passenger_count" not in df_clean.columns and "passengers" in df_clean.columns:
        df_clean["passenger_count"] = df_clean["passengers"]
    if "payload_weight_kg" not in df_clean.columns and "payload_mass_kg" in df_clean.columns:
        df_clean["payload_weight_kg"] = df_clean["payload_mass_kg"]
    if "total_vehicle_mass_kg" not in df_clean.columns:
        curb = df_clean.get("curb_weight_kg", df_clean.get("curb_mass_kg", 350.0))
        pld = df_clean.get("payload_weight_kg", df_clean.get("payload_mass_kg", 130.0))
        df_clean["total_vehicle_mass_kg"] = curb + pld
    if "rain_flaps_or_rack" not in df_clean.columns:
        df_clean["rain_flaps_or_rack"] = 0.0
    if "road_grade_pct" not in df_clean.columns:
        df_clean["road_grade_pct"] = df_clean.get("grade_pct", df_clean.get("grade_fraction", 0.0) * 100.0)
    if "accel_mps2" not in df_clean.columns:
        df_clean["accel_mps2"] = 0.0
    if "soc_pct" not in df_clean.columns:
        soc_val = df_clean.get("soc", df_clean.get("start_soc", 0.85))
        df_clean["soc_pct"] = soc_val * 100.0 if (isinstance(soc_val, (int, float)) and soc_val <= 1.05) else soc_val
    if "soh_pct" not in df_clean.columns:
        soh_val = df_clean.get("soh", 1.0)
        df_clean["soh_pct"] = soh_val * 100.0 if (isinstance(soh_val, (int, float)) and soh_val <= 1.05) else soh_val
    if "nominal_voltage_v" not in df_clean.columns:
        df_clean["nominal_voltage_v"] = np.where(df_clean["is_nmc"] == 1.0, 51.8, np.where(df_clean["is_lfp"] == 1.0, 51.2, 48.0))
    if "battery_capacity_kwh" not in df_clean.columns:
        df_clean["battery_capacity_kwh"] = np.where(df_clean["is_nmc"] == 1.0, 7.77, np.where(df_clean["is_lfp"] == 1.0, 5.12, 4.8))

    for col in RUNTIME_FEATURE_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = 0.0

    X = df_clean[RUNTIME_FEATURE_COLUMNS]
    y_power = df_clean["pack_power_kw"] if "pack_power_kw" in df_clean.columns else pd.Series(0.0, index=df_clean.index)
    groups = df_clean["trip_id"] if "trip_id" in df_clean.columns else df_clean.get("route_id", pd.Series(1, index=df_clean.index))

    return X, y_power, groups


def train_surrogate_pack_power(
    df: pd.DataFrame,
    out_path: str = "models/surrogate_model.joblib",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train XGBoost surrogate regressor to predict pack_power_kw using runtime driving conditions.

    Saves the final model directly as a compact .joblib artifact.
    """
    X, y, groups = prepare_runtime_features(df)
    gkf = GroupKFold(n_splits=n_splits)

    fold_maes = []
    fold_r2s = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.10,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=2,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        fold_maes.append(float(mean_absolute_error(y.iloc[val_idx], preds)))
        fold_r2s.append(float(r2_score(y.iloc[val_idx], preds)))

    final_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.10,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=2,
    )
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(final_model, out_path)

    return {
        "model": final_model,
        "cv_mae_kw": float(np.mean(fold_maes)),
        "cv_mae_w": float(np.mean(fold_maes) * 1000.0),
        "cv_r2": float(np.mean(fold_r2s)),
        "saved_path": out_path,
        "feature_names": RUNTIME_FEATURE_COLUMNS,
    }


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Prepare feature matrix and target series from synthetic dataset.

    Args:
        df: Trip-level or window-level DataFrame.

    Returns:
        Tuple of (X, y_wh_km, y_range_km, groups_route_id).
    """
    df_clean = df.copy()

    # Binary chemistry indicator (1 for LFP, 0 for lead-acid)
    if "is_lfp" not in df_clean.columns:
        if "chemistry" in df_clean.columns:
            df_clean["is_lfp"] = (df_clean["chemistry"] == "lfp").astype(float)
        else:
            df_clean["is_lfp"] = 0.0

    # Ensure all required features exist; fall back to uncorrupted equivalents if needed
    for col in FEATURE_COLUMNS:
        if col not in df_clean.columns:
            if col == "noisy_mean_voltage_v" and "mean_voltage_v" in df_clean.columns:
                df_clean[col] = df_clean["mean_voltage_v"]
            elif col == "noisy_mean_temp_c" and "mean_temp_c" in df_clean.columns:
                df_clean[col] = df_clean["mean_temp_c"]
            elif col == "noisy_elevation_gain_m" and "elevation_gain_m" in df_clean.columns:
                df_clean[col] = df_clean["elevation_gain_m"]
            else:
                df_clean[col] = 0.0

    X = df_clean[FEATURE_COLUMNS]
    y_wh_km = df_clean["wh_per_km"] if "wh_per_km" in df_clean.columns else pd.Series(0.0, index=df_clean.index)
    y_range_km = df_clean["range_remaining_km"] if "range_remaining_km" in df_clean.columns else pd.Series(0.0, index=df_clean.index)
    groups = df_clean["route_id"] if "route_id" in df_clean.columns else pd.Series(1, index=df_clean.index)

    return X, y_wh_km, y_range_km, groups


def train_and_evaluate_groupkfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    target_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train XGBoost regressor using GroupKFold grouped by route_id.

    Args:
        X: Feature matrix.
        y: Target series.
        groups: Route identifier series for group splitting.
        target_name: Name of target metric ('wh_per_km' or 'range_remaining_km').
        n_splits: Number of CV folds.
        random_state: Seed for repeatability.

    Returns:
        Dictionary with CV fold metrics, overall mean MAE, MAPE, R2, and final model.
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_maes = []
    fold_mapes = []
    fold_r2s = []
    oof_predictions = np.zeros(len(y))

    # XGBoost regressor parameters tailored for physical telemetry
    params = {
        "n_estimators": 120,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": random_state,
        "n_jobs": 1,
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        oof_predictions[val_idx] = preds

        mae = float(mean_absolute_error(y_val, preds))
        mape = float(mean_absolute_percentage_error(np.maximum(1e-2, y_val), np.maximum(0.0, preds)))
        r2 = float(r2_score(y_val, preds))

        fold_maes.append(mae)
        fold_mapes.append(mape)
        fold_r2s.append(r2)

    # Train final model on full dataset
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y)

    # Feature importance
    feat_imp = dict(zip(X.columns, [float(v) for v in final_model.feature_importances_]))

    return {
        "target_name": target_name,
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_mape_mean_pct": float(np.mean(fold_mapes) * 100.0),
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "fold_maes": fold_maes,
        "fold_mapes": fold_mapes,
        "feature_importances": feat_imp,
        "model": final_model,
        "oof_predictions": oof_predictions,
    }


def evaluate_held_out_temperature_band(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    temp_band_name: str,
) -> Dict[str, float]:
    """Evaluate trained model on an unseen held-out temperature band.

    Args:
        model: Trained XGBoost regressor.
        X_test: Test features in held-out temperature band.
        y_test: True target values.
        temp_band_name: Label describing the temperature band.

    Returns:
        Dict with MAE, MAPE (%), and R2 on the held-out band.
    """
    if len(y_test) == 0:
        return {"mae": 0.0, "mape_pct": 0.0, "r2": 0.0, "count": 0}

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(np.maximum(1e-2, y_test), np.maximum(0.0, preds)))
    r2 = float(r2_score(y_test, preds))

    return {
        "band": temp_band_name,
        "count": len(y_test),
        "mae": round(mae, 3),
        "mape_pct": round(mape * 100.0, 2),
        "r2": round(r2, 4),
    }


def run_full_pipeline(
    df: pd.DataFrame,
    n_splits: int = 5,
    held_out_temp_high: float = 38.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete training pipeline with GroupKFold and held-out temperature evaluation.

    Args:
        df: Synthetic trip dataset DataFrame.
        n_splits: Number of GroupKFold cross-validation folds.
        held_out_temp_high: Temperature threshold for held-out extreme heat band [deg C].
        random_state: Random seed.

    Returns:
        Comprehensive results dictionary containing metrics for both Wh/km and range models.
    """
    # Split into moderate training band and held-out extreme thermal band
    mask_moderate = df["ambient_temp_c"] <= held_out_temp_high
    mask_extreme = df["ambient_temp_c"] > held_out_temp_high

    df_train_pool = df[mask_moderate].reset_index(drop=True)
    df_heldout = df[mask_extreme].reset_index(drop=True)

    X_train_pool, y_wh_train, y_range_train, groups_train = prepare_features(df_train_pool)

    # Train Wh/km model
    results_wh = train_and_evaluate_groupkfold(
        X=X_train_pool,
        y=y_wh_train,
        groups=groups_train,
        target_name="wh_per_km",
        n_splits=n_splits,
        random_state=random_state,
    )

    # Train range_km model
    results_range = train_and_evaluate_groupkfold(
        X=X_train_pool,
        y=y_range_train,
        groups=groups_train,
        target_name="range_remaining_km",
        n_splits=n_splits,
        random_state=random_state,
    )

    # Evaluate on held-out temperature band if available
    held_out_wh = {}
    held_out_range = {}
    if len(df_heldout) > 0:
        X_heldout, y_wh_heldout, y_range_heldout, _ = prepare_features(df_heldout)
        held_out_wh = evaluate_held_out_temperature_band(
            results_wh["model"], X_heldout, y_wh_heldout, f"T > {held_out_temp_high} C"
        )
        held_out_range = evaluate_held_out_temperature_band(
            results_range["model"], X_heldout, y_range_heldout, f"T > {held_out_temp_high} C"
        )

    return {
        "wh_per_km_model": results_wh,
        "range_model": results_range,
        "held_out_wh": held_out_wh,
        "held_out_range": held_out_range,
        "moderate_samples": len(df_train_pool),
        "heldout_samples": len(df_heldout),
    }


def main():
    """CLI Entry point for model training."""
    parser = argparse.ArgumentParser(description="Train XGBoost BMS models on synthetic trip datasets.")
    parser.add_argument("--data", type=str, default="data/trips.parquet", help="Path to input dataset")
    parser.add_argument("--out-dir", type=str, default="models/", help="Output directory for saved models")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits (default: 5)")
    parser.add_argument("--held-out-temp", type=float, default=38.0, help="Held-out temperature threshold in C")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file '{args.data}' not found. Run datagen first.")

    print(f"Loading data from {args.data}...")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    print(f"Loaded {len(df)} records. Training XGBoost models with GroupKFold (k={args.folds})...")
    res = run_full_pipeline(
        df=df,
        n_splits=args.folds,
        held_out_temp_high=args.held_out_temp,
        random_state=args.seed,
    )

    wh_res = res["wh_per_km_model"]
    range_res = res["range_model"]

    print("\n=================== MODEL EVALUATION REPORT ===================")
    print(f"Target: Wh/km")
    print(f"  CV MAE:  {wh_res['cv_mae_mean']:.3f} +/- {wh_res['cv_mae_std']:.3f} Wh/km")
    print(f"  CV MAPE: {wh_res['cv_mape_mean_pct']:.2f}%")
    print(f"  CV R^2:  {wh_res['cv_r2_mean']:.4f}")
    if res["held_out_wh"]:
        print(f"  Held-Out ({res['held_out_wh'].get('band')}): MAE={res['held_out_wh']['mae']} Wh/km, MAPE={res['held_out_wh']['mape_pct']}%")

    print(f"\nTarget: Range (km)")
    print(f"  CV MAE:  {range_res['cv_mae_mean']:.3f} +/- {range_res['cv_mae_std']:.3f} km")
    print(f"  CV MAPE: {range_res['cv_mape_mean_pct']:.2f}%")
    print(f"  CV R^2:  {range_res['cv_r2_mean']:.4f}")
    if res["held_out_range"]:
        print(f"  Held-Out ({res['held_out_range'].get('band')}): MAE={res['held_out_range']['mae']} km, MAPE={res['held_out_range']['mape_pct']}%")

    os.makedirs(args.out_dir, exist_ok=True)
    wh_model_path = os.path.join(args.out_dir, "model_wh_per_km.json")
    range_model_path = os.path.join(args.out_dir, "model_range_km.json")
    wh_res["model"].save_model(wh_model_path)
    range_res["model"].save_model(range_model_path)
    print(f"\nModels successfully saved to {wh_model_path} and {range_model_path}")

    # Also save as .joblib for Streamlit Cloud optimization
    joblib.dump(wh_res["model"], os.path.join(args.out_dir, "model_wh_per_km.joblib"))
    joblib.dump(range_res["model"], os.path.join(args.out_dir, "model_range_km.joblib"))

    # If dataset has pack_power_kw, train the runtime pack_power surrogate
    if "pack_power_kw" in df.columns:
        print("\nTraining pack_power_kw surrogate model on exact runtime features...")
        joblib_path = os.path.join(args.out_dir, "surrogate_model.joblib")
        power_res = train_surrogate_pack_power(df, out_path=joblib_path, n_splits=args.folds, random_state=args.seed)
        print(f"Pack Power Surrogate CV MAE: {power_res['cv_mae_kw']:.4f} kW ({power_res['cv_mae_w']:.1f} W) | R^2: {power_res['cv_r2']:.4f}")
        print(f"Saved surrogate joblib model to {joblib_path}")


if __name__ == "__main__":
    main()

