from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison


def test_project_bootstrap_detects_consistently_better_candidate():
    rows = []
    for project in range(20):
        actual = float(project)
        for snapshot in range(3):
            rows.append({
                "canonical_project_id": f"P-{project}",
                "actual": actual,
                "baseline": actual + 10.0,
                "candidate": actual + 2.0,
            })
    result = paired_project_mae_comparison(
        pd.DataFrame(rows),
        actual="actual",
        baseline_prediction="baseline",
        candidate_prediction="candidate",
        bootstrap_samples=200,
        seed=7,
    )
    assert result["evaluation_unit"] == "project"
    assert result["projects"] == 20
    assert result["percentage_mae_improvement"] == 80.0
    assert result["improvement_95pct_ci"][0] > 0
    assert result["probability_candidate_better"] == 1.0
