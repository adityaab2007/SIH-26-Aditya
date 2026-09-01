"""Causal feature builders for experiments 120-124.

All joins are backward/as-of joins. Missing verified external data remains missing;
no zero-valued observations are fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_ROOT = ROOT / "data" / "external"


def _dates(frame: pd.DataFrame, column: str = "snapshot_date") -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def asof_external_join(
    snapshots: pd.DataFrame,
    external: pd.DataFrame,
    *,
    external_date: str,
    by: list[str] | None = None,
    prefix: str,
) -> pd.DataFrame:
    """Backward join external observations without row multiplication."""
    left = snapshots.copy()
    left["snapshot_date"] = _dates(left)
    right = external.copy()
    right[external_date] = pd.to_datetime(right[external_date], errors="coerce")
    by = list(by or [])
    original_rows = len(left)
    left["__row_id"] = np.arange(original_rows)
    sort_left = by + ["snapshot_date"]
    sort_right = by + [external_date]
    left = left.sort_values(sort_left)
    right = right.dropna(subset=[external_date]).sort_values(sort_right)
    if right.duplicated(by + [external_date]).any():
        raise ValueError(f"{prefix}: external join keys are not unique")
    joined = pd.merge_asof(
        left,
        right,
        left_on="snapshot_date",
        right_on=external_date,
        by=by or None,
        direction="backward",
        allow_exact_matches=True,
    )
    if len(joined) != original_rows or joined["__row_id"].nunique() != original_rows:
        raise AssertionError(f"{prefix}: external-data join duplicated/dropped snapshot rows")
    future = joined[external_date].notna() & joined[external_date].gt(joined["snapshot_date"])
    if future.any():
        raise AssertionError(f"{prefix}: future external observation entered feature vector")
    joined = joined.sort_values("__row_id").drop(columns="__row_id")
    joined[f"{prefix}_external_data_available"] = joined[external_date].notna()
    joined[f"{prefix}_external_feature_timestamp"] = joined[external_date]
    return joined


def load_verified_external(experiment: str, filename: str) -> tuple[pd.DataFrame | None, dict]:
    """Load a prepared external table only when its manifest declares verification."""
    folder = EXTERNAL_ROOT / experiment
    manifest_path = folder / "source_manifest.json"
    data_path = folder / filename
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest.get("verified_observations_bundled", False):
        return None, manifest
    if not data_path.exists():
        return None, manifest
    return pd.read_csv(data_path, low_memory=False), manifest


def add_wpi_features(frame: pd.DataFrame, wpi: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "wpi_all", "wpi_steel", "wpi_cement_mineral", "wpi_fuel_power"}
    missing = required - set(wpi.columns)
    if missing:
        raise ValueError(f"Exp120 WPI table missing columns: {sorted(missing)}")
    ext = wpi.copy(); ext["period"] = pd.to_datetime(ext["period"], errors="coerce")
    for col in sorted(required - {"period"}):
        ext[col] = _safe_numeric(ext[col])
        ext[f"{col}_yoy"] = ext[col].pct_change(12) * 100
        ext[f"{col}_mom3"] = ext[col].pct_change(3) * 100
    ext["wpi_all_volatility_6m"] = ext["wpi_all_yoy"].rolling(6, min_periods=3).std()
    ext["wpi_all_acceleration"] = ext["wpi_all_yoy"].diff(3)
    joined = asof_external_join(frame, ext, external_date="period", prefix="exp120")
    joined["exp120_cost_x_inflation"] = _safe_numeric(joined.get("approved_cost_cr", pd.Series(index=joined.index))) * _safe_numeric(joined.get("wpi_all_yoy", pd.Series(index=joined.index)))
    return joined


def add_rainfall_features(frame: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    required = {"period", "geo_key", "rainfall_mm", "rainfall_anomaly_pct"}
    missing = required - set(rainfall.columns)
    if missing:
        raise ValueError(f"Exp121 rainfall table missing columns: {sorted(missing)}")
    data = frame.copy()
    if "weather_geo_key" not in data:
        data["weather_geo_key"] = pd.NA
    ext = rainfall.copy().rename(columns={"geo_key": "weather_geo_key"})
    ext["period"] = pd.to_datetime(ext["period"], errors="coerce")
    ext["rainfall_mm"] = _safe_numeric(ext["rainfall_mm"])
    ext["rainfall_anomaly_pct"] = _safe_numeric(ext["rainfall_anomaly_pct"])
    ext = ext.sort_values(["weather_geo_key", "period"])
    g = ext.groupby("weather_geo_key", dropna=False)
    for days, periods in [(30, 1), (60, 2), (90, 3), (180, 6)]:
        ext[f"rainfall_{days}d"] = g["rainfall_mm"].transform(lambda s, p=periods: s.rolling(p, min_periods=1).sum())
    ext["rainfall_volatility_90d"] = g["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=2).std())
    ext["rainfall_acceleration"] = g["rainfall_mm"].diff()
    joined = asof_external_join(data, ext, external_date="period", by=["weather_geo_key"], prefix="exp121")
    joined["exp121_rain_x_progress_deviation"] = _safe_numeric(joined.get("rainfall_90d", pd.Series(index=joined.index))) * _safe_numeric(joined.get("progress_deviation", pd.Series(index=joined.index)))
    joined["exp121_rain_x_slippage"] = _safe_numeric(joined.get("rainfall_90d", pd.Series(index=joined.index))) * _safe_numeric(joined.get("schedule_slippage_days", pd.Series(index=joined.index)))
    return joined


def normalize_ministry(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def add_budget_features(frame: pd.DataFrame, budget: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    required_budget = {"published_date", "budget_year", "budget_ministry", "capital_expenditure_cr", "total_expenditure_cr"}
    if required_budget - set(budget.columns):
        raise ValueError(f"Exp122 budget table missing columns: {sorted(required_budget - set(budget.columns))}")
    required_mapping = {"normalized_ministry", "budget_ministry", "match_status", "mapping_method"}
    if required_mapping - set(mapping.columns):
        raise ValueError("Exp122 ministry mapping does not satisfy the audited mapping contract")
    data = frame.copy(); data["normalized_ministry"] = data.get("ministry", pd.Series(index=data.index)).map(normalize_ministry)
    safe_map = mapping[mapping["match_status"].eq("verified")].drop_duplicates("normalized_ministry")
    data = data.merge(safe_map, on="normalized_ministry", how="left", validate="m:1")
    ext = budget.copy(); ext["published_date"] = pd.to_datetime(ext["published_date"], errors="coerce")
    ext["capital_expenditure_cr"] = _safe_numeric(ext["capital_expenditure_cr"])
    ext["total_expenditure_cr"] = _safe_numeric(ext["total_expenditure_cr"])
    ext = ext.sort_values(["budget_ministry", "published_date"])
    g = ext.groupby("budget_ministry", dropna=False)
    ext["capital_budget_yoy_growth"] = g["capital_expenditure_cr"].pct_change() * 100
    ext["capital_share"] = ext["capital_expenditure_cr"] / ext["total_expenditure_cr"].replace(0, np.nan)
    joined = asof_external_join(data, ext, external_date="published_date", by=["budget_ministry"], prefix="exp122")
    joined["exp122_project_to_capital_budget"] = _safe_numeric(joined.get("approved_cost_cr", pd.Series(index=joined.index))) / _safe_numeric(joined.get("capital_expenditure_cr", pd.Series(index=joined.index))).replace(0, np.nan)
    joined["exp122_external_match_method"] = joined.get("mapping_method")
    return joined


@dataclass(frozen=True)
class EventThresholds:
    cost_change_pct: float
    schedule_change_days: float
    progress_change_pct: float
    expenditure_change_pct: float


def learn_event_thresholds(train: pd.DataFrame) -> EventThresholds:
    """Learn robust, training-only thresholds from within-project monthly changes."""
    data = train.sort_values(["canonical_project_id", "snapshot_date"]).copy()
    g = data.groupby("canonical_project_id", sort=False)
    cost = g["revised_cost_cr"].pct_change().abs() * 100 if "revised_cost_cr" in data else pd.Series(dtype=float)
    sched = g["schedule_slippage_days"].diff().abs() if "schedule_slippage_days" in data else pd.Series(dtype=float)
    prog = g["physical_progress"].diff().abs() if "physical_progress" in data else pd.Series(dtype=float)
    exp = g["cumulative_expenditure_cr"].pct_change().abs() * 100 if "cumulative_expenditure_cr" in data else pd.Series(dtype=float)
    def q(s: pd.Series, floor: float) -> float:
        clean = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return max(floor, float(clean.quantile(.60))) if not clean.empty else floor
    return EventThresholds(q(cost, .5), q(sched, 15.0), q(prog, 1.0), q(exp, 1.0))


def add_event_sequence_features(frame: pd.DataFrame, thresholds: EventThresholds) -> pd.DataFrame:
    """Build causal event/motif history using only current and previous reports."""
    data = frame.copy(); data["snapshot_date"] = _dates(data)
    data["__order"] = np.arange(len(data))
    data = data.sort_values(["canonical_project_id", "snapshot_date", "__order"])
    out = []
    for _, group in data.groupby("canonical_project_id", sort=False):
        events: list[str] = []
        previous = None
        for idx, row in group.iterrows():
            event = "N0"
            if previous is not None:
                rc, pc = row.get("revised_cost_cr"), previous.get("revised_cost_cr")
                if pd.notna(rc) and pd.notna(pc) and float(pc) != 0:
                    d = (float(rc) - float(pc)) / abs(float(pc)) * 100
                    if d >= thresholds.cost_change_pct: event = "C+"
                    elif d <= -thresholds.cost_change_pct: event = "C-"
                ss, ps = row.get("schedule_slippage_days"), previous.get("schedule_slippage_days")
                if event == "N0" and pd.notna(ss) and pd.notna(ps):
                    d = float(ss) - float(ps)
                    if d >= thresholds.schedule_change_days: event = "S+"
                    elif d <= -thresholds.schedule_change_days: event = "S-"
                pp, pprev = row.get("physical_progress"), previous.get("physical_progress")
                if event == "N0" and pd.notna(pp) and pd.notna(pprev):
                    d = float(pp) - float(pprev)
                    if abs(d) < thresholds.progress_change_pct: event = "P0"
                    elif d >= thresholds.progress_change_pct: event = "P+"
                ee, eprev = row.get("cumulative_expenditure_cr"), previous.get("cumulative_expenditure_cr")
                if event == "N0" and pd.notna(ee) and pd.notna(eprev) and float(eprev) != 0:
                    d = (float(ee) - float(eprev)) / abs(float(eprev)) * 100
                    event = "E+" if d >= thresholds.expenditure_change_pct else "E-" if abs(d) < thresholds.expenditure_change_pct else event
            events.append(event)
            seq = [e for e in events if e != "N0"]
            last2 = ">".join(seq[-2:]) if len(seq) >= 2 else "NONE"
            last3 = ">".join(seq[-3:]) if len(seq) >= 3 else "NONE"
            deterioration = {"C+", "S+", "P0", "E-"}
            streak = 0
            for e in reversed(seq):
                if e in deterioration: streak += 1
                else: break
            counts = pd.Series(seq).value_counts(normalize=True) if seq else pd.Series(dtype=float)
            entropy = float(-(counts * np.log2(counts)).sum()) if len(counts) else 0.0
            out.append((idx, event, last2, last3, len(seq), streak, entropy))
            previous = row
    features = pd.DataFrame(out, columns=["__idx", "exp123_latest_event", "exp123_last2", "exp123_last3", "exp123_event_count", "exp123_deterioration_streak", "exp123_event_entropy"]).set_index("__idx")
    for col in features.columns: data.loc[features.index, col] = features[col]
    return data.sort_values("__order").drop(columns="__order")


def select_stable_motifs(train_features: pd.DataFrame, min_project_support: int = 12, limit: int = 12) -> list[str]:
    """Select motif vocabulary from training support only, never holdout outcomes."""
    candidates: list[tuple[str, int]] = []
    for col in ["exp123_last2", "exp123_last3"]:
        if col not in train_features: continue
        support = train_features[train_features[col].ne("NONE")].groupby(col)["canonical_project_id"].nunique()
        candidates.extend((str(motif), int(n)) for motif, n in support.items() if int(n) >= min_project_support)
    candidates.sort(key=lambda x: (-x[1], x[0]))
    result = []
    for motif, _ in candidates:
        if motif not in result: result.append(motif)
        if len(result) >= limit: break
    return result


def materialize_motif_indicators(frame: pd.DataFrame, motifs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    data = frame.copy(); names = []
    for i, motif in enumerate(motifs):
        name = f"exp123_motif_{i:02d}"; names.append(name)
        data[name] = data.get("exp123_last2", pd.Series(index=data.index)).eq(motif) | data.get("exp123_last3", pd.Series(index=data.index)).eq(motif)
        data[name] = data[name].astype(float)
    return data, names
