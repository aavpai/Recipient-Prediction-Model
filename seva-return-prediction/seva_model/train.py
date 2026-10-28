from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .calendar import infer_core_calendar, load_overrides
from .config import load_config
from .features import build_training_panel
from .io import collapse_to_visits, load_raw_export
from .modeling import DateSplit, save_bundle, train_models


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Seva return-probability model.")
    p.add_argument("--data", required=True, help="Path to Seva Guest Visit export CSV")
    p.add_argument("--config", default="config/drive_calendar.yaml")
    p.add_argument("--calendar-overrides", default=None)
    p.add_argument("--train-start", default="2024-01-01")
    p.add_argument("--train-end", default="2025-12-31")
    p.add_argument("--test-start", default="2026-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument("--model", default="models/seva_return_model.joblib")
    p.add_argument("--metrics", default="outputs/model_metrics.csv")
    p.add_argument("--validation", default="outputs/drive_validation.csv")
    p.add_argument("--calendar-output", default="outputs/inferred_drive_calendar.csv")
    args = p.parse_args()

    config = load_config(args.config)
    raw = load_raw_export(args.data)
    visits = collapse_to_visits(raw)
    overrides = load_overrides(args.calendar_overrides)
    calendar = infer_core_calendar(visits, config, overrides)
    panel = build_training_panel(visits, calendar)

    split = DateSplit(
        train_start=parse_date(args.train_start),
        train_end=parse_date(args.train_end),
        test_start=parse_date(args.test_start),
        test_end=parse_date(args.test_end) if args.test_end else None,
    )
    bundle, metrics, validation = train_models(panel, split)

    # Enrich drive-level validation with actual total attendance and drive labels.
    totals = (
        visits.groupby("date")["guest_id"].nunique()
        .rename("total_actual_attendees")
        .reset_index()
    )
    calendar_cols = [
        "date", "position", "second_drive", "toy", "backpack",
        "thanksgiving", "christmas"
    ]
    validation = validation.merge(
        calendar[calendar_cols],
        left_on=["date", "drive_position"],
        right_on=["date", "position"],
        how="left",
    ).drop(columns=["position"])
    validation = validation.merge(totals, on="date", how="left")
    validation["first_time_attendees"] = (
        validation["total_actual_attendees"] - validation["actual_returners"]
    )
    def _special_label(r):
        labels = [name.title() for name in ["toy", "backpack", "thanksgiving", "christmas"] if int(r.get(name, 0))]
        return "+".join(labels)
    validation["special"] = validation.apply(_special_label, axis=1)

    for output in [args.model, args.metrics, args.validation, args.calendar_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    save_bundle(bundle, args.model)
    metrics.to_csv(args.metrics, index=False)
    validation.to_csv(args.validation, index=False)
    calendar.to_csv(args.calendar_output, index=False)

    print("Training complete")
    print(metrics.to_string(index=False))
    print(f"\nModel: {args.model}")
    print(f"Validation: {args.validation}")


if __name__ == "__main__":
    main()
