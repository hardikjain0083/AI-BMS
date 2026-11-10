"""Unit tests for ML training pipeline (ebms_sim.train)."""

import pytest
import numpy as np
import pandas as pd
from ebms_sim.datagen import generate_dataset
from ebms_sim.train import (
    prepare_features,
    train_and_evaluate_groupkfold,
    evaluate_held_out_temperature_band,
    run_full_pipeline,
)


@pytest.fixture(scope="module")
def sample_dataset():
    """Generate a fast reproducible dataset fixture for training tests."""
    df_trips, _ = generate_dataset(
        n_trips=25,
        trip_duration_s=60,
        seed=42,
        n_workers=1,
        n_unique_routes=8,
    )
    return df_trips


def test_prepare_features(sample_dataset):
    """Verify feature matrix is correctly extracted and targets aligned."""
    X, y_wh, y_range, groups = prepare_features(sample_dataset)

    assert len(X) == len(sample_dataset)
    assert len(y_wh) == len(sample_dataset)
    assert len(y_range) == len(sample_dataset)
    assert len(groups) == len(sample_dataset)

    assert "passengers" in X.columns
    assert "noisy_mean_voltage_v" in X.columns
    assert "is_lfp" in X.columns
    assert not X.isnull().values.any()


def test_groupkfold_no_route_leakage(sample_dataset):
    """Verify GroupKFold ensures complete route isolation between train and validation."""
    X, y_wh, _, groups = prepare_features(sample_dataset)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=3)

    for train_idx, val_idx in gkf.split(X, y_wh, groups=groups):
        train_routes = set(groups.iloc[train_idx])
        val_routes = set(groups.iloc[val_idx])
        # Intersection must be empty!
        assert len(train_routes.intersection(val_routes)) == 0


def test_train_and_evaluate_groupkfold(sample_dataset):
    """Verify XGBoost training produces valid MAE, MAPE, R2, and feature importances."""
    X, y_wh, _, groups = prepare_features(sample_dataset)

    results = train_and_evaluate_groupkfold(
        X=X,
        y=y_wh,
        groups=groups,
        target_name="wh_per_km",
        n_splits=3,
        random_state=42,
    )

    assert results["cv_mae_mean"] > 0.0
    assert results["cv_mape_mean_pct"] > 0.0
    assert "model" in results
    assert len(results["feature_importances"]) == len(X.columns)


def test_held_out_temperature_pipeline(sample_dataset):
    """Verify complete pipeline executes GroupKFold and held-out temperature evaluation."""
    # Ensure there is variation in temperature to create a held-out band
    df = sample_dataset.copy()
    # Explicitly set some temperatures above 38 C for the held-out test
    df.loc[:4, "ambient_temp_c"] = 42.0

    pipeline_res = run_full_pipeline(
        df=df,
        n_splits=3,
        held_out_temp_high=38.0,
        random_state=42,
    )

    assert "wh_per_km_model" in pipeline_res
    assert "range_model" in pipeline_res
    assert pipeline_res["wh_per_km_model"]["cv_mae_mean"] > 0.0
    assert pipeline_res["range_model"]["cv_mae_mean"] > 0.0

    assert pipeline_res["held_out_wh"]["count"] > 0
    assert "mae" in pipeline_res["held_out_wh"]
    assert "mape_pct" in pipeline_res["held_out_wh"]

