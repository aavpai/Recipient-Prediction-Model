from datetime import date, timedelta
from pathlib import Path
import joblib
import pytest
import pandas as pd

from seva_model.calendar import future_drive_meta, infer_core_calendar
from seva_model.config import load_config
from seva_model.features import (FEATURES, _count_recent_visits, advance_recipient_state_snapshot, build_future_scoring_frame, build_recipient_state_snapshot, build_scoring_frame_from_states)
from seva_model.io import collapse_to_visits, load_raw_export


def test_pretrained_model_exists_and_matches_features():
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(root / "models" / "seva_operational_model.joblib")
    assert bundle["features"] == FEATURES
    assert "gradient_boosting_model" in bundle


def test_fast_scoring_pipeline_on_synthetic_data():
    root = Path(__file__).resolve().parents[1]
    raw = load_raw_export(root / "sample_data" / "synthetic_guest_visits.csv")
    visits = collapse_to_visits(raw)
    config = load_config(root / "config" / "drive_calendar.yaml")
    calendar = infer_core_calendar(visits, config)
    future = date(max(visits["date"]).year + 1, 1, 10)
    meta = future_drive_meta(future, 1, config)
    score = build_future_scoring_frame(visits, calendar, meta)
    bundle = joblib.load(root / "models" / "seva_operational_model.joblib")
    p = bundle["gradient_boosting_model"].predict_proba(
        score[FEATURES].astype(float).to_numpy()
    )[:, 1]
    assert len(p) == len(score)
    assert ((p >= 0) & (p <= 1)).all()



def test_recent_visit_counts_use_inclusive_window_boundaries():
    as_of = date(2026, 3, 31)
    prior_dates = [
        date(2025, 12, 31),
        date(2026, 1, 30),
        date(2026, 3, 1),
        date(2026, 3, 31),
    ]

    assert _count_recent_visits(prior_dates, as_of, 30) == 2
    assert _count_recent_visits(prior_dates, as_of, 60) == 3
    assert _count_recent_visits(prior_dates, as_of, 180) == 4
    assert _count_recent_visits(prior_dates, as_of, 0) == 1
    assert _count_recent_visits([], as_of, 30) == 0



def test_state_snapshot_reproduces_existing_future_features():
    root = Path(__file__).resolve().parents[1]
    raw = load_raw_export(root / "sample_data" / "synthetic_guest_visits.csv")
    visits = collapse_to_visits(raw)
    config = load_config(root / "config" / "drive_calendar.yaml")
    calendar = infer_core_calendar(visits, config)
    future = max(visits["date"]) + timedelta(days=14)
    meta = future_drive_meta(future, 2, config)

    state_rows, previous_core_date = build_recipient_state_snapshot(visits, calendar)
    recent_counts = {}
    for guest_id, group in visits.groupby("guest_id"):
        dates = group["date"].tolist()
        recent_counts[str(guest_id)] = {
            f"visits_last_{days}": sum(0 <= (future - visit_date).days <= days for visit_date in dates)
            for days in [30, 60, 90, 180]
        }

    from_states = build_scoring_frame_from_states(
        state_rows, recent_counts, meta, previous_core_date
    )
    from_history = build_future_scoring_frame(visits, calendar, meta)

    left = from_states.sort_values("guest_id").reset_index(drop=True)
    right = from_history.sort_values("guest_id").reset_index(drop=True)
    assert left["guest_id"].tolist() == right["guest_id"].tolist()
    for feature in FEATURES:
        assert left[feature].tolist() == pytest.approx(right[feature].tolist())


def test_incremental_state_update_matches_full_history_replay():
    root = Path(__file__).resolve().parents[1]
    raw = load_raw_export(root / "sample_data" / "synthetic_guest_visits.csv")
    visits = collapse_to_visits(raw)
    config = load_config(root / "config" / "drive_calendar.yaml")
    calendar = infer_core_calendar(visits, config)

    base_states, _ = build_recipient_state_snapshot(visits, calendar)
    known_ids = sorted(visits["guest_id"].astype(str).unique())[:2]
    drive_date = max(visits["date"]) + timedelta(days=14)
    meta = future_drive_meta(drive_date, 2, config)
    drive_visits = pd.DataFrame([
        {
            "visit_id": "incremental-known",
            "guest_id": known_ids[0],
            "date": drive_date,
            "outreach_name": "Backpack Drive",
            "total_individuals": 3,
            "kids": 1,
            "adults": 2,
            "seniors": 0,
            "homeless": 0,
        },
        {
            "visit_id": "incremental-new",
            "guest_id": "NEW-GUEST-001",
            "date": drive_date,
            "outreach_name": "Backpack Drive",
            "total_individuals": 4,
            "kids": 2,
            "adults": 2,
            "seniors": 0,
            "homeless": 0,
        },
    ])

    incrementally_updated = advance_recipient_state_snapshot(
        base_states, drive_visits, meta
    )
    full_visits = pd.concat([visits, drive_visits], ignore_index=True)
    calendar_row = {
        **meta,
        "attendance": len(drive_visits),
        "calendar_source": "manual",
    }
    full_calendar = pd.concat(
        [calendar, pd.DataFrame([calendar_row])], ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    full_replay, _ = build_recipient_state_snapshot(full_visits, full_calendar)

    left = pd.DataFrame(incrementally_updated).sort_values("guest_id").reset_index(drop=True)
    right = pd.DataFrame(full_replay).sort_values("guest_id").reset_index(drop=True)
    assert left["guest_id"].tolist() == right["guest_id"].tolist()
    compare_columns = sorted(set(left.columns) & set(right.columns))
    for column in compare_columns:
        assert left[column].tolist() == right[column].tolist()
