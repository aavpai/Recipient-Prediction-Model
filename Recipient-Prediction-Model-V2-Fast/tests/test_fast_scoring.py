from datetime import date
from pathlib import Path
import joblib

from seva_model.calendar import future_drive_meta, infer_core_calendar
from seva_model.config import load_config
from seva_model.features import FEATURES, build_future_scoring_frame
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
