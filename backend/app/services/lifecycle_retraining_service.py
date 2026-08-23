"""Live retraining adapter for the official PAIMANA monthly lifecycle models.

The website year-range selector must retrain the lifecycle cost, delay and risk
models, not the preserved five-feature completed-project baseline.  This module
keeps that policy in one place and returns an API-friendly training receipt.
"""
from __future__ import annotations

import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import train_window
from backend.app.services import monthly_prediction_service


def _training_data() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    years = data["completion_year"].dropna().astype(int)
    if years.empty:
        raise ValueError("No identity-verified PAIMANA lifecycle outcomes are available for retraining.")
    return data, identity, int(years.min()), int(years.max())


def retrain_lifecycle(start_year: int, end_year: int) -> dict:
    """Retrain the monthly lifecycle cost/delay/risk stack for a selected period.

    Algorithm selection remains internal-temporal: the latest completion year
    actually present inside the selected training range is used to choose the
    cost and delay regressor, then the winning regressors and the Random Forest
    risk classifier are fitted on the full selected training range.  All later
    completion years remain future holdout data.
    """
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("Training start year must be less than or equal to training end year.")

    data, identity, min_year, max_year = _training_data()
    if end_year >= max_year:
        raise ValueError(f"Training must end before {max_year} so an unseen future lifecycle holdout remains.")
    if end_year < min_year or start_year > max_year:
        raise ValueError(f"Training range must overlap identity-verified lifecycle data ({min_year}-{max_year}).")

    selected_training_years = data.loc[data.completion_year.between(start_year, end_year), "completion_year"].dropna()
    if selected_training_years.empty:
        raise ValueError("The selected period has no identity-verified lifecycle training projects.")
    internal_validation_year = int(selected_training_years.max())

    result = train_window(start_year, end_year, max_year, data=data, identity=identity)
    metadata = result["metadata"]
    lifecycle = result["lifecycle"]
    lifecycle_metrics = lifecycle["metrics"]
    baseline_metrics = result["baseline"]["metrics"]
    feature_audit = metadata.get("feature_availability", {})
    selected = metadata.get("selected_algorithms", {})

    # Retraining can overwrite a previously loaded window; force inference to
    # reload the freshly written artifacts on the next forecast request.
    monthly_prediction_service._bundle.cache_clear()

    return {
        "status": "success",
        "model_family": "monthly_lifecycle",
        "model_version": metadata["model_version"],
        "window": f"{start_year}_{end_year}",
        "training_years": f"{start_year}-{end_year}",
        "testing_years": f"{end_year + 1}-{max_year}",
        "training_samples": metadata["training_snapshots"],
        "training_projects": metadata["unique_training_projects"],
        "testing_samples": metadata["test_snapshots"],
        "testing_projects": metadata["unique_test_projects"],
        "features_used": metadata["features_used"],
        "feature_count": len(metadata["features_used"]),
        "selected_algorithms": {
            "cost": selected.get("cost"),
            "delay": selected.get("delay"),
            "risk": "random_forest",
        },
        "internal_validation_year": internal_validation_year,
        "future_holdout_start": end_year + 1,
        "future_holdout_end": max_year,
        "metrics": {
            "cost_model": lifecycle_metrics["cost"],
            "delay_model": lifecycle_metrics["delay"],
            "risk_model": lifecycle_metrics["risk"],
            "metadata": {
                "feature_count": len(metadata["features_used"]),
                "features_used": metadata["features_used"],
                "feature_quality": {
                    "data_quality_score": feature_audit.get("data_quality_score"),
                    "removed_invalid_feature_count": feature_audit.get("removed_invalid_feature_count", len(feature_audit.get("removed_features", []))),
                },
                "leakage_policy": metadata.get("leakage_policy"),
                "snapshot_weighting_policy": metadata.get("snapshot_weighting_policy"),
            },
        },
        "baseline_comparison": {
            "feature_count": 5,
            "cost_mae": baseline_metrics["cost"]["MAE"],
            "delay_mae": baseline_metrics["delay"]["MAE"],
            "risk_macro_f1": baseline_metrics["risk"]["macro_f1"],
            "purpose": "Controlled benchmark only; not the retrained production forecast model.",
        },
        "lifecycle_stages": lifecycle.get("lifecycle_stages", {}),
        "leakage_guard": "The future holdout is excluded from algorithm selection and fitting; project identities may not cross the temporal split.",
    }
