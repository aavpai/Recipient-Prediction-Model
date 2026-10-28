from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import numpy as np

from .calendar import future_drive_meta, infer_core_calendar, load_overrides
from .config import load_config
from .features import FEATURES, build_future_scoring_frame
from .io import collapse_to_visits, load_raw_export
from .modeling import load_bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Forecast returning recipients for an upcoming Seva drive.")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/seva_return_model.joblib")
    p.add_argument("--date", required=True, help="Upcoming drive date, YYYY-MM-DD")
    p.add_argument("--drive-position", type=int, choices=[1, 2], required=True)
    p.add_argument("--config", default="config/drive_calendar.yaml")
    p.add_argument("--calendar-overrides", default=None)
    p.add_argument("--output", default=None, help="Recipient-level CSV; defaults to outputs/predictions_DATE.csv")
    p.add_argument("--summary", default=None, help="JSON summary; defaults to outputs/forecast_DATE.json")
    args = p.parse_args()

    upcoming = date.fromisoformat(args.date)
    config = load_config(args.config)
    raw = load_raw_export(args.data)
    visits = collapse_to_visits(raw)
    overrides = load_overrides(args.calendar_overrides)
    calendar = infer_core_calendar(visits, config, overrides)
    meta = future_drive_meta(upcoming, args.drive_position, config)
    score = build_future_scoring_frame(visits, calendar, meta)

    bundle = load_bundle(args.model)
    features = bundle.get("features", FEATURES)
    model = bundle["gradient_boosting_model"]
    score["probability"] = model.predict_proba(score[features].astype(float).to_numpy())[:, 1]
    score["expected_people_contribution"] = score["probability"] * score["total_individuals"]

    expected_households = float(score["probability"].sum())
    expected_people = float(score["expected_people_contribution"].sum())

    # Poisson-binomial variance under an independence approximation. Real drive-level
    # shocks can make actual forecast uncertainty wider than this interval.
    variance = float(np.sum(score["probability"] * (1 - score["probability"])))
    sd = variance ** 0.5
    lo80 = max(0.0, expected_households - 1.2816 * sd)
    hi80 = expected_households + 1.2816 * sd

    special = [name for name in ["toy", "backpack", "thanksgiving", "christmas"] if meta.get(name)]
    summary = {
        "drive_date": upcoming.isoformat(),
        "drive_position": args.drive_position,
        "tefap_extra_food": bool(args.drive_position == 2),
        "special_overlays": special,
        "known_recipients_scored": int(len(score)),
        "predicted_returning_households": expected_households,
        "model_based_80pct_interval_independence_approx": [lo80, hi80],
        "predicted_people_in_returning_households": expected_people,
        "note": "This model forecasts returning known recipients, not first-time households.",
    }

    output = args.output or f"outputs/predictions_{upcoming.isoformat()}.csv"
    summary_path = args.summary or f"outputs/forecast_{upcoming.isoformat()}.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "guest_id", "probability", "total_individuals", "kids", "adults", "seniors",
        "days_since_last_any", "visits_last_60", "visits_last_90",
        "core_attendance_rate", "ordinary_d2_affinity", "consecutive_core_misses",
    ]
    score.sort_values("probability", ascending=False)[cols].to_csv(output, index=False)
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nRecipient-level predictions: {output}")


if __name__ == "__main__":
    main()
