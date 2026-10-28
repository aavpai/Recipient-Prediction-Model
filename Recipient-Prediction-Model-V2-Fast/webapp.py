from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import hashlib
import math

import joblib
import numpy as np
from flask import Flask, redirect, render_template, request, url_for

from seva_model.calendar import future_drive_meta, infer_core_calendar
from seva_model.config import load_config
from seva_model.features import FEATURES, build_future_scoring_frame
from seva_model.io import collapse_to_visits, load_raw_export

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "drive_calendar.yaml"
MODEL_PATH = BASE_DIR / "models" / "seva_operational_model.joblib"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

CONFIG = load_config(CONFIG_PATH)
MODEL_BUNDLE = joblib.load(MODEL_PATH)
MODEL = MODEL_BUNDLE["gradient_boosting_model"]
MODEL_FEATURES = MODEL_BUNDLE.get("features", FEATURES)
TRAINED_THROUGH = MODEL_BUNDLE.get("trained_through", "2026-09-26")

# Tiny in-process cache. If the same CSV is submitted twice while this worker is
# alive, avoid parsing and rebuilding its calendar again.
_CACHE: dict[str, object] = {
    "sha256": None,
    "visits": None,
    "calendar": None,
}


def _drive_label(meta: dict) -> str:
    base = (
        "Drive 2 — TEFAP / extra food"
        if int(meta["position"]) == 2
        else "Drive 1 — Regular distribution"
    )
    specials = [
        name.title()
        for name in ["toy", "backpack", "thanksgiving", "christmas"]
        if int(meta.get(name, 0))
    ]
    if specials:
        base += " + " + " + ".join(specials)
    return base


def _probability_buckets(probabilities: np.ndarray) -> list[dict]:
    cuts = [
        ("75–100%", probabilities >= 0.75),
        ("50–75%", (probabilities >= 0.50) & (probabilities < 0.75)),
        ("25–50%", (probabilities >= 0.25) & (probabilities < 0.50)),
        ("Under 25%", probabilities < 0.25),
    ]
    return [{"label": label, "count": int(mask.sum())} for label, mask in cuts]


def _load_history(file_storage):
    payload = file_storage.read()
    if not payload:
        raise ValueError("The uploaded CSV is empty.")

    digest = hashlib.sha256(payload).hexdigest()
    if _CACHE["sha256"] == digest:
        return _CACHE["visits"], _CACHE["calendar"]

    raw = load_raw_export(BytesIO(payload))
    visits = collapse_to_visits(raw)
    if visits.empty:
        raise ValueError("The uploaded export contains no usable recipient visits.")

    calendar = infer_core_calendar(visits, CONFIG)
    _CACHE.update({"sha256": digest, "visits": visits, "calendar": calendar})
    return visits, calendar


@app.get("/")
def home():
    return render_template(
        "index.html",
        result=None,
        error=None,
        trained_through=TRAINED_THROUGH,
    )


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    if request.method == "GET":
        return redirect(url_for("home"))

    try:
        upload = request.files.get("data_file")
        if upload is None or not upload.filename:
            raise ValueError("Choose the cumulative Seva Guest Visit CSV first.")
        if not upload.filename.lower().endswith(".csv"):
            raise ValueError("The uploaded file must be a CSV export.")

        requested_date = date.fromisoformat(request.form.get("drive_date", ""))
        position = int(request.form.get("drive_position", "0"))
        if position not in (1, 2):
            raise ValueError("Choose Drive 1 or Drive 2.")

        # FAST PATH: no model fitting occurs here.
        visits, calendar = _load_history(upload)

        latest_visit = max(visits["date"])
        if requested_date <= latest_visit:
            raise ValueError(
                "The forecast date must be after the latest visit in the uploaded "
                f"history ({latest_visit.isoformat()})."
            )

        meta = future_drive_meta(requested_date, position, CONFIG)
        score = build_future_scoring_frame(visits, calendar, meta)
        if score.empty:
            raise ValueError("No previously observed recipients are available to score.")

        missing = [f for f in MODEL_FEATURES if f not in score.columns]
        if missing:
            raise ValueError(f"Scoring data is missing model features: {missing}")

        probabilities = MODEL.predict_proba(
            score[MODEL_FEATURES].astype(float).to_numpy()
        )[:, 1]
        score["probability"] = probabilities
        score["expected_people_contribution"] = (
            score["probability"] * score["total_individuals"]
        )

        expected_households = float(probabilities.sum())
        expected_people = float(score["expected_people_contribution"].sum())

        # Conditional Poisson-binomial variance. This reflects uncertainty in
        # the individual attendance draws, not all possible model error.
        variance = float(np.sum(probabilities * (1.0 - probabilities)))
        sd = math.sqrt(max(variance, 0.0))
        lo80 = max(0.0, expected_households - 1.2816 * sd)
        hi80 = expected_households + 1.2816 * sd

        result = {
            "drive_date": requested_date.strftime("%B %d, %Y").replace(" 0", " "),
            "drive_date_iso": requested_date.isoformat(),
            "drive_label": _drive_label(meta),
            "predicted_returning_households": round(expected_households),
            "predicted_returning_households_exact": round(expected_households, 1),
            "interval_low": round(lo80),
            "interval_high": round(hi80),
            "predicted_people": round(expected_people),
            "known_recipients": int(len(score)),
            "active_180": int((score["days_since_last_any"] <= 180).sum()),
            "high_probability": int((probabilities >= 0.50).sum()),
            "latest_visit": latest_visit.strftime("%B %d, %Y").replace(" 0", " "),
            "buckets": _probability_buckets(probabilities),
            "mean_probability": round(float(probabilities.mean()) * 100, 1),
            "specials": [
                name.title()
                for name in ["toy", "backpack", "thanksgiving", "christmas"]
                if int(meta.get(name, 0))
            ],
            "trained_through": TRAINED_THROUGH,
        }
        return render_template(
            "index.html",
            result=result,
            error=None,
            trained_through=TRAINED_THROUGH,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            result=None,
            error=str(exc),
            trained_through=TRAINED_THROUGH,
        ), 400


@app.errorhandler(413)
def too_large(_exc):
    return render_template(
        "index.html",
        result=None,
        error="That CSV is larger than the 50 MB upload limit.",
        trained_through=TRAINED_THROUGH,
    ), 413


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "pretrained-fast-scoring",
        "model_trained_through": TRAINED_THROUGH,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
