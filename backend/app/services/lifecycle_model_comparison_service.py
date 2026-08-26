"""Generic production-vs-experiment lifecycle comparison orchestration.

This service owns the judge-safe comparison flow, not any experiment. Experiment
PRs register themselves by adding ``backend.app.ml.experiments.adapter_exp*.py``.
The highest-numbered registered adapter becomes the default challenger.
"""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import (
    available_experiments,
    default_experiment_adapter,
    get_experiment_adapter,
)
from backend.app.services import lifecycle_retraining_service as retraining
from backend.app.services import lifecycle_simulation_service as simulation

_COMPARISON_SESSIONS: dict[str, dict] = {}
_MAX_COMPARISON_SESSIONS = 20


def _float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _improvement_percentage(baseline_error: float, candidate_error: float) -> float | None:
    if baseline_error > 0:
        return (baseline_error - candidate_error) / baseline_error * 100.0
    if candidate_error == 0:
        return 0.0
    return None


def experiment_catalog() -> dict:
    items = available_experiments()
    active = default_experiment_adapter()
    return {
        "items": items,
        "count": len(items),
        "active_experiment_id": active.experiment_id if active else None,
        "active_experiment_name": active.name if active else None,
    }


def _open_production_session_from_frozen_data(
    *,
    data: pd.DataFrame,
    start_year: int,
    end_year: int,
    production_bundle: dict,
    production_receipt: dict,
) -> dict:
    frame = data.copy()
    frame["completion_year"] = pd.to_numeric(frame["completion_year"], errors="coerce")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    if "completion_date" in frame:
        frame["completion_date"] = pd.to_datetime(frame["completion_date"], errors="coerce")

    metadata = production_bundle["metadata"]
    features = list(metadata.get("features_used") or production_receipt.get("features_used") or [])
    train_projects = set(frame.loc[frame.completion_year.between(start_year, end_year), "canonical_project_id"].dropna())
    held_all = frame[frame.completion_year.gt(end_year)].copy()
    overlap = train_projects & set(held_all.canonical_project_id.dropna())
    if overlap:
        raise ValueError(f"Project-group leakage across comparison judge split: {len(overlap)} project(s)")
    if held_all.empty:
        raise ValueError("No later lifecycle projects are available for a leakage-free comparison test.")

    held_all = held_all.sort_values(["canonical_project_id", "snapshot_date"])
    held_latest = held_all.drop_duplicates("canonical_project_id", keep="last").reset_index(drop=True)
    held_latest["record_index"] = np.arange(len(held_latest), dtype=int)
    history_counts = held_all.groupby("canonical_project_id").size().to_dict()

    session_id = uuid.uuid4().hex[:16]
    simulation._CUSTOM_SESSIONS[session_id] = {
        "training_start": start_year,
        "training_end": end_year,
        "run_id": production_receipt.get("run_id"),
        "dataset_fingerprint": production_receipt.get("dataset_fingerprint"),
        "features": features,
        "models": production_bundle,
        "held_out": held_latest,
        "history_counts": history_counts,
        "predictions": {},
    }
    while len(simulation._CUSTOM_SESSIONS) > simulation._MAX_CUSTOM_SESSIONS:
        simulation._CUSTOM_SESSIONS.pop(next(iter(simulation._CUSTOM_SESSIONS)), None)

    return {
        "session_id": session_id,
        "run_id": production_receipt.get("run_id"),
        "dataset_fingerprint": production_receipt.get("dataset_fingerprint"),
        "model_version": production_receipt.get("model_version"),
        "training_start": start_year,
        "training_end": end_year,
        "leakage_guard": (
            f"Production and challenger use one frozen PAIMANA frame; only projects completed in "
            f"{start_year}-{end_year} can contribute to fitting and every offered judge project completes after {end_year}."
        ),
        "actual_outcomes_sent_to_browser": False,
    }


def retrain_and_compare(start_year: int, end_year: int, experiment_id: str | None = None) -> dict:
    """Freshly retrain production and one registered experiment on the same evidence."""
    adapter = get_experiment_adapter(experiment_id)
    start_year, end_year = int(start_year), int(end_year)
    data, _identity, _min_year, max_year = retraining._training_data()

    production = retraining.retrain_lifecycle(start_year, end_year)
    production_bundle = simulation._artifact_bundle(start_year, end_year, production.get("run_id"))
    fitted = adapter.module.fit_against_production(
        data=data,
        training_start=start_year,
        training_end=end_year,
        test_end=max_year,
        production_bundle=production_bundle,
        production_receipt=production,
    )
    if not isinstance(fitted, dict):
        raise ValueError(f"Experiment adapter {adapter.experiment_id} returned an invalid fit result.")
    experiment = dict(fitted.get("experiment") or {})
    overall = dict(fitted.get("overall_comparison") or {})
    runtime_state = fitted.get("runtime_state")
    if not experiment.get("run_id") or runtime_state is None:
        raise ValueError(f"Experiment adapter {adapter.experiment_id} did not return run_id/runtime_state.")
    if experiment.get("experiment_id") != adapter.experiment_id:
        raise ValueError("Experiment adapter identity mismatch.")

    production_session = _open_production_session_from_frozen_data(
        data=data,
        start_year=start_year,
        end_year=end_year,
        production_bundle=production_bundle,
        production_receipt=production,
    )
    underlying = simulation._session(production_session["session_id"])
    held = underlying["held_out"].copy()
    comparable = adapter.module.filter_comparable_rows(held, runtime_state)
    if not isinstance(comparable, pd.DataFrame) or comparable.empty:
        raise ValueError(f"No held-out projects satisfy the {adapter.experiment_id} comparison contract.")
    comparable_indices = set(int(value) for value in comparable.record_index.tolist())
    counts = comparable.groupby("completion_year").size().sort_index()

    comparison_session_id = uuid.uuid4().hex[:16]
    _COMPARISON_SESSIONS[comparison_session_id] = {
        "production_session_id": production_session["session_id"],
        "production_run_id": production.get("run_id"),
        "dataset_fingerprint": production.get("dataset_fingerprint"),
        "adapter_id": adapter.experiment_id,
        "experiment_run_id": experiment["run_id"],
        "runtime_state": runtime_state,
        "comparable_indices": comparable_indices,
        "overall": overall,
        "candidate_predictions": {},
    }
    while len(_COMPARISON_SESSIONS) > _MAX_COMPARISON_SESSIONS:
        _COMPARISON_SESSIONS.pop(next(iter(_COMPARISON_SESSIONS)), None)

    session = {
        "session_id": comparison_session_id,
        "comparison_session_id": comparison_session_id,
        "production_session_id": production_session["session_id"],
        "run_id": production.get("run_id"),
        "production_run_id": production.get("run_id"),
        "experiment_id": adapter.experiment_id,
        "experiment_name": adapter.name,
        "experiment_run_id": experiment["run_id"],
        "dataset_fingerprint": production.get("dataset_fingerprint"),
        "training_start": start_year,
        "training_end": end_year,
        "eligible_test_years": [{"year": int(year), "projects": int(count)} for year, count in counts.items()],
        "actual_outcomes_sent_to_browser": False,
        "leakage_guard": production_session.get("leakage_guard"),
    }
    return {
        "status": "success",
        "production": production,
        "experiment": experiment,
        "overall_comparison": overall,
        "session": session,
    }


def _comparison_session(session_id: str) -> dict:
    if session_id not in _COMPARISON_SESSIONS:
        raise KeyError(session_id)
    return _COMPARISON_SESSIONS[session_id]


def comparison_projects(session_id: str, year: int) -> dict:
    session = _comparison_session(session_id)
    response = simulation.custom_projects(session["production_session_id"], int(year))
    response["items"] = [row for row in response["items"] if int(row["record_index"]) in session["comparable_indices"]]
    if not response["items"]:
        raise ValueError(f"No comparable production/challenger projects are available for {year}.")
    response.update({
        "session_id": session_id,
        "comparison_session_id": session_id,
        "experiment_id": session["adapter_id"],
        "experiment_run_id": session["experiment_run_id"],
        "note": "Only projects scoreable by both the fresh production model and active experiment are shown; actual outcomes remain hidden until reveal.",
    })
    return response


def predict_comparison(session_id: str, record_index: int) -> dict:
    session = _comparison_session(session_id)
    record_index = int(record_index)
    if record_index not in session["comparable_indices"]:
        raise ValueError("Selected project is not comparable under the active experiment contract.")
    adapter = get_experiment_adapter(session["adapter_id"])
    production = simulation.predict_custom(session["production_session_id"], record_index)
    underlying = simulation._session(session["production_session_id"])
    row = simulation._session_row(underlying, record_index)
    experiment_payload = adapter.module.predict_project(row, session["runtime_state"])
    if not isinstance(experiment_payload, dict):
        raise ValueError("Experiment adapter returned an invalid project prediction.")
    experiment_payload = dict(experiment_payload)
    experiment_payload.update({
        "experiment_id": adapter.experiment_id,
        "experiment_run_id": session["experiment_run_id"],
        "experiment_name": adapter.name,
        "scope": adapter.scope,
    })

    candidate_cost = _float(experiment_payload.get("predicted_cost_overrun"))
    production_cost = _float(production.get("predicted_cost_overrun"))
    if candidate_cost is None or production_cost is None:
        raise ValueError("Cost-comparison adapters must return predicted_cost_overrun.")

    candidate_delay = _float(experiment_payload.get("predicted_delay_days"))
    production_delay = _float(production.get("predicted_delay_days"))
    session["candidate_predictions"][record_index] = experiment_payload

    comparison = {
        "production": {
            "predicted_cost_overrun": production_cost,
            "predicted_delay_days": production_delay,
            "run_id": session["production_run_id"],
        },
        "experiment": experiment_payload,
        "prediction_difference_pp": round(candidate_cost - production_cost, 4),
        "actual_outcome_sent_to_browser": False,
    }
    if candidate_delay is not None and production_delay is not None:
        comparison["delay_prediction_difference_days"] = round(candidate_delay - production_delay, 4)

    production["comparison"] = comparison
    production["comparison_session_id"] = session_id
    return production


def reveal_comparison(session_id: str, record_index: int) -> dict:
    session = _comparison_session(session_id)
    record_index = int(record_index)
    candidate = session["candidate_predictions"].get(record_index)
    if candidate is None:
        raise ValueError("Generate both predictions before revealing the actual outcome.")

    actual = simulation.reveal_custom(session["production_session_id"], record_index)
    actual_cost = _float(actual.get("actual_cost_overrun"))
    production_cost_error = _float(actual.get("cost_error_absolute_pp"))
    candidate_cost = _float(candidate.get("predicted_cost_overrun"))
    if actual_cost is None or production_cost_error is None or candidate_cost is None:
        raise ValueError("Comparison reveal requires a cost prediction and official actual cost overrun.")

    experiment_cost_error = abs(candidate_cost - actual_cost)
    cost_improvement = _improvement_percentage(production_cost_error, experiment_cost_error)
    payload = {
        "production_cost_error_absolute_pp": round(production_cost_error, 4),
        "experiment_cost_error_absolute_pp": round(experiment_cost_error, 4),
        "individual_error_improvement_percentage": round(cost_improvement, 3) if cost_improvement is not None else None,
        "experiment_better_for_project": experiment_cost_error < production_cost_error,
        "experiment_better_cost_for_project": experiment_cost_error < production_cost_error,
        "production_predicted_cost_overrun": _production_prediction(session, record_index),
        "experiment_predicted_cost_overrun": candidate_cost,
        "experiment_id": session["adapter_id"],
        "experiment_run_id": session["experiment_run_id"],
    }

    actual_delay = _float(actual.get("actual_delay_days"))
    production_delay_error = _float(actual.get("delay_error_absolute_days"))
    candidate_delay = _float(candidate.get("predicted_delay_days"))
    if actual_delay is not None and production_delay_error is not None and candidate_delay is not None:
        experiment_delay_error = abs(candidate_delay - actual_delay)
        delay_improvement = _improvement_percentage(production_delay_error, experiment_delay_error)
        payload.update({
            "production_delay_error_absolute_days": round(production_delay_error, 4),
            "experiment_delay_error_absolute_days": round(experiment_delay_error, 4),
            "individual_delay_error_improvement_percentage": round(delay_improvement, 3) if delay_improvement is not None else None,
            "experiment_better_delay_for_project": experiment_delay_error < production_delay_error,
            "production_predicted_delay_days": _production_delay_prediction(session, record_index),
            "experiment_predicted_delay_days": candidate_delay,
        })

    actual["comparison"] = payload
    actual["comparison_session_id"] = session_id
    return actual


def _production_prediction(session: dict, record_index: int) -> float:
    underlying = simulation._session(session["production_session_id"])
    prediction = underlying["predictions"].get(int(record_index)) or {}
    value = _float(prediction.get("predicted_cost_overrun"))
    if value is None:
        raise ValueError("Production prediction is unavailable for comparison reveal.")
    return value


def _production_delay_prediction(session: dict, record_index: int) -> float:
    underlying = simulation._session(session["production_session_id"])
    prediction = underlying["predictions"].get(int(record_index)) or {}
    value = _float(prediction.get("predicted_delay_days"))
    if value is None:
        raise ValueError("Production delay prediction is unavailable for comparison reveal.")
    return value
