from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


@dataclass
class DateSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date | None = None


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "observed_rate": float(np.mean(y)),
        "mean_predicted_probability": float(np.mean(p)),
    }


def train_models(panel: pd.DataFrame, split: DateSplit) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(panel["date"]).dt.date
    train_mask = (dates >= split.train_start) & (dates <= split.train_end)
    test_mask = dates >= split.test_start
    if split.test_end is not None:
        test_mask &= dates <= split.test_end

    train = panel.loc[train_mask].copy()
    test = panel.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Training or test split is empty. Adjust date arguments.")

    X_train = train[FEATURES].astype(float).to_numpy()
    y_train = train["target"].astype(int).to_numpy()
    X_test = test[FEATURES].astype(float).to_numpy()
    y_test = test["target"].astype(int).to_numpy()

    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    logistic.fit(X_train, y_train)

    boosting = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=42,
    )
    boosting.fit(X_train, y_train)

    p_logit = logistic.predict_proba(X_test)[:, 1]
    p_boost = boosting.predict_proba(X_test)[:, 1]

    rows = []
    for name, p in [("logistic", p_logit), ("gradient_boosting", p_boost)]:
        rows.append({"model": name, **_metrics(y_test, p)})

    active = test["days_since_last_any"].astype(float).to_numpy() <= 180
    if active.any() and len(np.unique(y_test[active])) == 2:
        rows.append({"model": "gradient_boosting_active180", **_metrics(y_test[active], p_boost[active])})

    metrics = pd.DataFrame(rows)

    validation = test[["date", "drive_position", "target"]].copy()
    validation["predicted_probability"] = p_boost
    by_drive = (
        validation.groupby(["date", "drive_position"], as_index=False)
        .agg(
            known_recipients_scored=("target", "size"),
            predicted_returners=("predicted_probability", "sum"),
            actual_returners=("target", "sum"),
        )
    )

    bundle = {
        "features": FEATURES,
        "logistic_model": logistic,
        "gradient_boosting_model": boosting,
        "split": {
            "train_start": split.train_start.isoformat(),
            "train_end": split.train_end.isoformat(),
            "test_start": split.test_start.isoformat(),
            "test_end": split.test_end.isoformat() if split.test_end else None,
        },
        "target_definition": "Known recipient attends a specific upcoming core Seva drive",
        "notes": {
            "drive_2": "TEFAP / extra-food drive",
            "identification_limit": (
                "Drive 2 and TEFAP always occur together, so historical data cannot "
                "separate a later-in-month effect from the TEFAP/more-food effect."
            ),
        },
    }
    return bundle, metrics, by_drive


def save_bundle(bundle: dict, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)
