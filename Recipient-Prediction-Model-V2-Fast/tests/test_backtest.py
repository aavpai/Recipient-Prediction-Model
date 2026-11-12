from datetime import date, timedelta

import numpy as np
import pandas as pd

from seva_model.backtest import run_walk_forward_backtest
from seva_model.features import FEATURES


class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, X):
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1 - p, p])


def make_panel():
    rng = np.random.default_rng(17)
    dates = [date(2025, 1, 4) + timedelta(days=14 * i) for i in range(4)]
    rows = []

    for drive_index, drive_date in enumerate(dates):
        for recipient_index in range(120):
            row = {feature: float(rng.normal()) for feature in FEATURES}
            row["date"] = drive_date
            row["drive_position"] = 1 if drive_index % 2 == 0 else 2
            row["second_drive"] = 1 if drive_index % 2 else 0
            row["attended_prev_core"] = int(recipient_index % 3 == 0)
            row["target"] = int((recipient_index + drive_index) % 6 == 0)
            row["guest_id"] = f"synthetic-{recipient_index}"
            rows.append(row)

    return pd.DataFrame(rows), dates


def test_walk_forward_trains_only_on_prior_drives_and_exports_aggregates():
    panel, dates = make_panel()
    training_cutoffs = []

    def fake_trainer(train):
        training_cutoffs.append(max(train["date"]))
        return ConstantProbabilityModel(train["target"].mean())

    drive_results, summary, calibration = run_walk_forward_backtest(
        panel,
        start_date=dates[2],
        min_training_drives=2,
        model_trainer=fake_trainer,
    )

    assert drive_results["date"].tolist() == [
        dates[2].isoformat(),
        dates[3].isoformat(),
    ]
    assert drive_results["training_drives"].tolist() == [2, 3]
    assert drive_results["training_through"].tolist() == [
        dates[1].isoformat(),
        dates[2].isoformat(),
    ]
    assert training_cutoffs == [dates[1], dates[2]]
    assert "guest_id" not in drive_results.columns
    assert summary["test_drives"] == 2
    assert summary["model"]["mae"] >= 0
    assert not calibration.empty


def test_walk_forward_skips_dates_without_enough_training_history():
    panel, dates = make_panel()

    def fake_trainer(train):
        return ConstantProbabilityModel(train["target"].mean())

    drive_results, summary, _ = run_walk_forward_backtest(
        panel,
        start_date=dates[0],
        min_training_drives=2,
        model_trainer=fake_trainer,
    )

    assert drive_results["date"].tolist() == [
        dates[2].isoformat(),
        dates[3].isoformat(),
    ]
    assert summary["first_test_drive"] == dates[2].isoformat()
