from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .calendar import infer_core_calendar, load_overrides
from .config import load_config
from .features import FEATURES, build_training_panel
from .io import collapse_to_visits, load_raw_export
from .modeling import fit_operational_model


def _probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate pooled recipient-level metrics from out-of-time probabilities."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    has_both_classes = len(np.unique(y)) == 2

    return {
        "roc_auc": float(roc_auc_score(y, p)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "observed_rate": float(y.mean()) if len(y) else None,
        "mean_predicted_probability": float(p.mean()) if len(p) else None,
        "recipient_drive_observations": int(len(y)),
    }


def _drive_metrics(drive_results: pd.DataFrame, predicted_column: str) -> dict:
    """Calculate aggregate forecast errors across distribution drives."""
    if drive_results.empty:
        raise ValueError("No drives were eligible for backtesting.")

    actual = drive_results["actual_returners"].astype(float).to_numpy()
    predicted = drive_results[predicted_column].astype(float).to_numpy()
    errors = predicted - actual
    absolute_errors = np.abs(errors)
    nonzero = actual != 0

    total_predicted = float(predicted.sum())
    total_actual = float(actual.sum())

    return {
        "drives": int(len(drive_results)),
        "total_predicted_returners": total_predicted,
        "total_actual_returners": total_actual,
        "aggregate_bias": float(total_predicted - total_actual),
        "aggregate_bias_pct": (
            float((total_predicted / total_actual - 1) * 100)
            if total_actual else None
        ),
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape_pct": (
            float(np.mean(absolute_errors[nonzero] / actual[nonzero]) * 100)
            if nonzero.any() else None
        ),
        "mean_error": float(errors.mean()),
        "within_20_recipients": int((absolute_errors <= 20).sum()),
        "within_20_pct": float((absolute_errors <= 20).mean() * 100),
    }


def _last_drive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Estimate return probability from prior-drive attendance and drive position.

    Rates are estimated from training rows only and grouped by whether the
    recipient attended the immediately preceding core drive and whether the
    upcoming drive is Drive 2.
    """
    global_rate = float(train["target"].mean())
    grouped = train.groupby(["attended_prev_core", "second_drive"])["target"].mean()
    rates = {
        (int(previous), int(position)): float(rate)
        for (previous, position), rate in grouped.items()
    }

    probabilities = [
        rates.get((int(previous), int(position)), global_rate)
        for previous, position in zip(
            test["attended_prev_core"], test["second_drive"]
        )
    ]
    return np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)


def _calibration_table(
    y: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Return an aggregate equal-frequency calibration table."""
    frame = pd.DataFrame({
        "actual_returned": np.asarray(y, dtype=int),
        "predicted_probability": np.asarray(probabilities, dtype=float),
    })
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "probability_bin",
                "observations",
                "mean_predicted_probability",
                "actual_return_rate",
            ]
        )

    q = min(bins, int(frame["predicted_probability"].nunique()))
    if q < 1:
        frame["probability_bin"] = 1
    else:
        frame["probability_bin"] = (
            pd.qcut(
                frame["predicted_probability"],
                q=q,
                labels=False,
                duplicates="drop",
            ) + 1
        )

    return (
        frame.groupby("probability_bin", as_index=False)
        .agg(
            observations=("actual_returned", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            actual_return_rate=("actual_returned", "mean"),
        )
        .sort_values("probability_bin")
        .reset_index(drop=True)
    )


def run_walk_forward_backtest(
    panel: pd.DataFrame,
    start_date: date,
    end_date: date | None = None,
    min_training_drives: int = 12,
    model_trainer=fit_operational_model,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Backtest one drive at a time, training only on earlier drives.

    Each test drive is scored using a fresh model fitted to all eligible
    recipient-drive observations from dates strictly before that drive. No
    recipient-level predictions or identifiers are written to output files.
    """
    if min_training_drives < 1:
        raise ValueError("min_training_drives must be at least 1.")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    start_date = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end_date = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    if data.empty:
        raise ValueError("Training panel is empty.")
    required = set(FEATURES + ["date", "target", "attended_prev_core", "second_drive"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Training panel is missing required columns: {missing}")

    test_dates = sorted(
        d for d in data["date"].dropna().unique()
        if d >= start_date and (end_date is None or d <= end_date)
    )

    drive_rows: list[dict] = []
    pooled_actual: list[np.ndarray] = []
    pooled_model_probability: list[np.ndarray] = []
    pooled_baseline_probability: list[np.ndarray] = []

    for drive_date in test_dates:
        train = data.loc[data["date"] < drive_date].copy()
        test = data.loc[data["date"] == drive_date].copy()

        if train["date"].nunique() < min_training_drives:
            continue
        if train.empty or test.empty or train["target"].nunique() < 2:
            continue

        model = model_trainer(train)
        model_probability = model.predict_proba(
            test[FEATURES].astype(float).to_numpy()
        )[:, 1]
        baseline_probability = _last_drive_baseline(train, test)
        actual_returners = int(test["target"].sum())

        row = {
            "date": drive_date.isoformat(),
            "drive_position": int(test["drive_position"].iloc[0]) if "drive_position" in test else None,
            "second_drive": int(test["second_drive"].iloc[0]),
            "toy": int(test["toy"].iloc[0]) if "toy" in test else 0,
            "backpack": int(test["backpack"].iloc[0]) if "backpack" in test else 0,
            "thanksgiving": int(test["thanksgiving"].iloc[0]) if "thanksgiving" in test else 0,
            "christmas": int(test["christmas"].iloc[0]) if "christmas" in test else 0,
            "training_through": max(train["date"]).isoformat(),
            "training_drives": int(train["date"].nunique()),
            "training_observations": int(len(train)),
            "known_recipients_scored": int(len(test)),
            "model_predicted_returners": float(model_probability.sum()),
            "baseline_predicted_returners": float(baseline_probability.sum()),
            "actual_returners": actual_returners,
        }
        row["model_error"] = row["model_predicted_returners"] - actual_returners
        row["model_abs_error"] = abs(row["model_error"])
        row["model_ape_pct"] = (
            row["model_abs_error"] / actual_returners * 100
            if actual_returners else None
        )
        row["baseline_error"] = row["baseline_predicted_returners"] - actual_returners
        row["baseline_abs_error"] = abs(row["baseline_error"])
        row["baseline_ape_pct"] = (
            row["baseline_abs_error"] / actual_returners * 100
            if actual_returners else None
        )
        drive_rows.append(row)

        # Only aggregate labels/probabilities are retained; Guest IDs are not.
        pooled_actual.append(test["target"].astype(int).to_numpy())
        pooled_model_probability.append(np.asarray(model_probability, dtype=float))
        pooled_baseline_probability.append(np.asarray(baseline_probability, dtype=float))

    if not drive_rows:
        raise ValueError(
            "No drives were eligible. Move the start date later, lower "
            "min_training_drives, or check the inferred drive calendar."
        )

    drive_results = pd.DataFrame(drive_rows)
    y = np.concatenate(pooled_actual)
    model_p = np.concatenate(pooled_model_probability)
    baseline_p = np.concatenate(pooled_baseline_probability)

    model_summary = {
        **_drive_metrics(drive_results, "model_predicted_returners"),
        **_probability_metrics(y, model_p),
    }
    baseline_summary = {
        **_drive_metrics(drive_results, "baseline_predicted_returners"),
        **_probability_metrics(y, baseline_p),
    }

    summary = {
        "method": "expanding-window walk-forward validation",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat() if end_date else None,
        "minimum_training_drives": int(min_training_drives),
        "model_specification": (
            "HistGradientBoostingClassifier(learning_rate=0.06, max_iter=250, "
            "max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=1.0, "
            "random_state=42)"
        ),
        "baseline_definition": (
            "Training-set return rate grouped by prior-core-drive attendance "
            "and upcoming drive position; fallback is the overall training-set rate."
        ),
        "test_drives": int(len(drive_results)),
        "first_test_drive": str(drive_results["date"].iloc[0]),
        "last_test_drive": str(drive_results["date"].iloc[-1]),
        "model": model_summary,
        "baseline": baseline_summary,
        "privacy": (
            "Only drive-level aggregates and pooled metrics are exported. "
            "Recipient IDs and recipient-level predictions are not exported."
        ),
        "interpretation": (
            "Predictions cover returning recipients already known before each "
            "drive. They do not include first-time recipients or total attendance."
        ),
    }

    calibration = _calibration_table(y, model_p)
    return drive_results, summary, calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an expanding-window historical backtest of the Seva return model."
    )
    parser.add_argument("--data", required=True, help="Path to the Seva Guest Visit export CSV.")
    parser.add_argument("--config", default="config/drive_calendar.yaml")
    parser.add_argument(
        "--calendar-overrides",
        default="config/drive_calendar_overrides.csv",
        help="Fixed historical drive-date calendar; pass an empty string to use inference.",
    )
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-training-drives", type=int, default=12)
    parser.add_argument("--drive-output", default="outputs/walk_forward_drive_results.csv")
    parser.add_argument("--summary-output", default="outputs/walk_forward_summary.json")
    parser.add_argument("--calibration-output", default="outputs/walk_forward_calibration.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_export(args.data)
    visits = collapse_to_visits(raw)
    overrides = load_overrides(args.calendar_overrides) if args.calendar_overrides else None
    calendar = infer_core_calendar(visits, config, overrides)
    panel = build_training_panel(visits, calendar)

    drive_results, summary, calibration = run_walk_forward_backtest(
        panel=panel,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date) if args.end_date else None,
        min_training_drives=args.min_training_drives,
    )
    summary["calendar"] = {
        "core_drives": int(len(calendar)),
        "first_date": str(calendar["date"].min()),
        "last_date": str(calendar["date"].max()),
        "source_counts": (
            calendar["calendar_source"].value_counts().to_dict()
            if "calendar_source" in calendar else {}
        ),
        "overrides_file": args.calendar_overrides or None,
    }

    for output in [args.drive_output, args.summary_output, args.calibration_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    drive_results.to_csv(args.drive_output, index=False)
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    calibration.to_csv(args.calibration_output, index=False)

    print(json.dumps(summary, indent=2))
    print("\nDrive-level results:")
    print(drive_results.to_string(index=False))
    print(f"\nDrive results: {args.drive_output}")
    print(f"Summary: {args.summary_output}")
    print(f"Calibration: {args.calibration_output}")


if __name__ == "__main__":
    main()
