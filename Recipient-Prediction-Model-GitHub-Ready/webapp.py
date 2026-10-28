from __future__ import annotations

from datetime import date
from pathlib import Path
import math

import numpy as np
from flask import Flask, render_template, request

from seva_model.calendar import future_drive_meta, infer_core_calendar
from seva_model.config import load_config
from seva_model.features import FEATURES, build_future_scoring_frame, build_training_panel
from seva_model.io import collapse_to_visits, load_raw_export
from seva_model.modeling import fit_operational_model

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "drive_calendar.yaml"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def _drive_label(meta: dict) -> str:
    base = "Drive 2 — TEFAP / extra food" if int(meta["position"]) == 2 else "Drive 1 — Regular distribution"
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


@app.get("/")
def home():
    return render_template("index.html", result=None, error=None)


@app.post("/forecast")
def forecast():
    try:
        upload = request.files.get("data_file")
        if upload is None or not upload.filename:
            raise ValueError("Choose a Seva Guest Visit CSV export first.")
        if not upload.filename.lower().endswith(".csv"):
            raise ValueError("The uploaded file must be a CSV export.")

        requested_date = date.fromisoformat(request.form.get("drive_date", ""))
        position = int(request.form.get("drive_position", "0"))
        if position not in (1, 2):
            raise ValueError("Choose Drive 1 or Drive 2.")

        config = load_config(CONFIG_PATH)
        raw = load_raw_export(upload.stream)
        visits = collapse_to_visits(raw)
        if visits.empty:
            raise ValueError("The uploaded export contains no usable recipient visits.")

        latest_visit = max(visits["date"])
        if requested_date <= latest_visit:
            raise ValueError(
                f"The forecast date must be after the latest visit in the export ({latest_visit.isoformat()})."
            )

        calendar = infer_core_calendar(visits, config)
        panel = build_training_panel(visits, calendar)
        if panel.empty:
            raise ValueError("There is not enough historical information to train a return model.")

        model = fit_operational_model(panel)
        meta = future_drive_meta(requested_date, position, config)
        score = build_future_scoring_frame(visits, calendar, meta)
        if score.empty:
            raise ValueError("No previously observed recipients are available to score.")

        probabilities = model.predict_proba(score[FEATURES].astype(float).to_numpy())[:, 1]
        score["probability"] = probabilities
        score["expected_people_contribution"] = score["probability"] * score["total_individuals"]

        expected_households = float(probabilities.sum())
        expected_people = float(score["expected_people_contribution"].sum())
        variance = float(np.sum(probabilities * (1.0 - probabilities)))
        sd = math.sqrt(max(variance, 0.0))
        lo80 = max(0.0, expected_households - 1.2816 * sd)
        hi80 = expected_households + 1.2816 * sd

        recent_active = int((score["days_since_last_any"] <= 180).sum())
        high_probability = int((probabilities >= 0.50).sum())

        result = {
            "drive_date": requested_date.strftime("%B %-d, %Y"),
            "drive_date_iso": requested_date.isoformat(),
            "drive_label": _drive_label(meta),
            "predicted_returning_households": round(expected_households),
            "predicted_returning_households_exact": round(expected_households, 1),
            "interval_low": round(lo80),
            "interval_high": round(hi80),
            "predicted_people": round(expected_people),
            "known_recipients": int(len(score)),
            "active_180": recent_active,
            "high_probability": high_probability,
            "latest_visit": latest_visit.strftime("%B %-d, %Y"),
            "buckets": _probability_buckets(probabilities),
            "mean_probability": round(float(probabilities.mean()) * 100, 1),
            "specials": [
                name.title()
                for name in ["toy", "backpack", "thanksgiving", "christmas"]
                if int(meta.get(name, 0))
            ],
        }
        return render_template("index.html", result=result, error=None)
    except Exception as exc:
        return render_template("index.html", result=None, error=str(exc)), 400


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
