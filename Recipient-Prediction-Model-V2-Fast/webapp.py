from __future__ import annotations

from datetime import date, timedelta
from functools import wraps
from pathlib import Path
import hashlib
import hmac
import math
import os
import secrets
from time import perf_counter

import joblib
import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for
from flask_wtf.csrf import CSRFProtect

from seva_model import storage
from seva_model.calendar import future_drive_meta, infer_core_calendar
from seva_model.config import load_config
from seva_model.features import (
    FEATURES,
    build_future_scoring_frame,
    build_scoring_frame_from_states,
)
from seva_model.io import collapse_to_visits, load_raw_export

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "drive_calendar.yaml"
MODEL_PATH = BASE_DIR / "models" / "seva_operational_model.joblib"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_urlsafe(32)
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
csrf = CSRFProtect(app)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
AUTH_READY = bool(APP_PASSWORD and os.environ.get("SECRET_KEY"))

CONFIG = load_config(CONFIG_PATH)
MODEL_BUNDLE = joblib.load(MODEL_PATH)
MODEL = MODEL_BUNDLE["gradient_boosting_model"]
MODEL_FEATURES = MODEL_BUNDLE.get("features", FEATURES)
TRAINED_THROUGH = MODEL_BUNDLE.get("trained_through", "2026-09-26")

# One-worker cache for the legacy cumulative-upload path. The file is hashed as
# a stream; it is never copied into a second large bytes object.
_CACHE: dict[str, object] = {
    "sha256": None,
    "visits": None,
    "calendar": None,
}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not AUTH_READY:
            return render_template(
                "setup_required.html",
                missing_password=not bool(APP_PASSWORD),
                missing_secret=not bool(os.environ.get("SECRET_KEY")),
            ), 503
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


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


def _parse_upload(file_storage):
    """Hash and parse a CSV upload without retaining a second full file copy."""
    stream = file_storage.stream
    stream.seek(0)
    if not stream.read(1):
        raise ValueError("The uploaded CSV is empty.")

    stream.seek(0)
    digest_builder = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    stream.seek(0)

    raw = load_raw_export(stream)
    visits = collapse_to_visits(raw)
    del raw
    if visits.empty:
        raise ValueError("The uploaded export contains no usable recipient visits.")
    return visits, digest


def _load_history(file_storage):
    digest_started = perf_counter()
    visits, digest = _parse_upload(file_storage)
    if _CACHE["sha256"] == digest:
        app.logger.info("History cache hit; skipping calendar inference.")
        return _CACHE["visits"], _CACHE["calendar"]

    calendar = infer_core_calendar(visits, CONFIG)
    _CACHE.update({"sha256": digest, "visits": visits, "calendar": calendar})
    app.logger.info(
        "Cumulative history parsed: visits=%s core_drives=%s elapsed_seconds=%.2f",
        len(visits),
        len(calendar),
        perf_counter() - digest_started,
    )
    return visits, calendar


def _make_result(score, requested_date: date, meta: dict) -> dict:
    if score.empty:
        raise ValueError("No previously observed recipients are available to score.")

    missing = [feature for feature in MODEL_FEATURES if feature not in score.columns]
    if missing:
        raise ValueError(f"Scoring data is missing model features: {missing}")

    probabilities = MODEL.predict_proba(
        score[MODEL_FEATURES].astype(float).to_numpy()
    )[:, 1]
    score = score.copy()
    score["probability"] = probabilities
    score["expected_people_contribution"] = (
        score["probability"] * score["total_individuals"]
    )

    expected_households = float(probabilities.sum())
    expected_people = float(score["expected_people_contribution"].sum())
    variance = float(np.sum(probabilities * (1.0 - probabilities)))
    sd = math.sqrt(max(variance, 0.0))
    lo80 = max(0.0, expected_households - 1.2816 * sd)
    hi80 = expected_households + 1.2816 * sd

    specials = [
        name.title()
        for name in ["toy", "backpack", "thanksgiving", "christmas"]
        if int(meta.get(name, 0))
    ]

    return {
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
        "buckets": _probability_buckets(probabilities),
        "mean_probability": round(float(probabilities.mean()) * 100, 1),
        "specials": specials,
        "trained_through": TRAINED_THROUGH,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_READY:
        return render_template(
            "setup_required.html",
            missing_password=not bool(APP_PASSWORD),
            missing_secret=not bool(os.environ.get("SECRET_KEY")),
        ), 503
    if request.method == "GET":
        return render_template("login.html", error=None)

    supplied = request.form.get("password", "")
    if hmac.compare_digest(supplied, APP_PASSWORD):
        session.clear()
        session["authenticated"] = True
        session.permanent = True
        return redirect(url_for("home"))
    return render_template("login.html", error="That password was not correct."), 401


@app.get("/")
@login_required
def home():
    db_status = None
    db_error = None
    if storage.is_configured():
        try:
            db_status = storage.get_status()
        except Exception:
            app.logger.exception("Could not read persistent recipient history status.")
            db_error = "The database is configured but could not be reached."
    return render_template(
        "index.html",
        result=None,
        error=None,
        trained_through=TRAINED_THROUGH,
        database_configured=storage.is_configured(),
        db_status=db_status,
        db_error=db_error,
    )


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.post("/initialize")
@login_required
def initialize():
    started = perf_counter()
    try:
        if not storage.is_configured():
            raise ValueError("Persistent storage is not configured. Add DATABASE_URL in Render first.")
        upload = request.files.get("data_file")
        if upload is None or not upload.filename:
            raise ValueError("Choose the cumulative/lifetime Guest Visit CSV.")
        if not upload.filename.lower().endswith(".csv"):
            raise ValueError("The uploaded file must be a CSV export.")

        visits, digest = _parse_upload(upload)
        calendar = infer_core_calendar(visits, CONFIG)
        summary = storage.initialize_history(visits, calendar, digest)
        app.logger.info(
            "Lifetime history initialized: recipients=%s visits=%s core_drives=%s elapsed_seconds=%.2f",
            summary["recipients"], summary["visits"], summary["core_drives"],
            perf_counter() - started,
        )
        return redirect(url_for("home"))
    except Exception as exc:
        app.logger.exception("Lifetime initialization failed.")
        return _render_home_error(str(exc)), 400


@app.post("/update")
@login_required
def update():
    started = perf_counter()
    try:
        if not storage.is_configured():
            raise ValueError("Persistent storage is not configured. Add DATABASE_URL in Render first.")
        upload = request.files.get("data_file")
        if upload is None or not upload.filename:
            raise ValueError("Choose the CSV for the latest completed drive.")
        if not upload.filename.lower().endswith(".csv"):
            raise ValueError("The uploaded file must be a CSV export.")

        drive_date = date.fromisoformat(request.form.get("drive_date", ""))
        position = int(request.form.get("drive_position", "0"))
        if position not in (1, 2):
            raise ValueError("Choose Drive 1 or Drive 2.")

        visits, digest = _parse_upload(upload)
        if visits["date"].nunique() != 1 or visits["date"].iloc[0] != drive_date:
            raise ValueError("Every row in the uploaded file must match the selected drive date.")

        meta = future_drive_meta(drive_date, position, CONFIG)
        summary = storage.apply_drive(visits, meta, digest)
        app.logger.info(
            "Incremental drive imported: date=%s position=%s attendees=%s elapsed_seconds=%.2f",
            drive_date.isoformat(), position, summary["attendees"], perf_counter() - started,
        )
        return redirect(url_for("home"))
    except Exception as exc:
        app.logger.exception("Incremental drive import failed.")
        return _render_home_error(str(exc)), 400


def _render_home_error(message: str):
    db_status = None
    db_error = None
    if storage.is_configured():
        try:
            db_status = storage.get_status()
        except Exception:
            db_error = "The database is configured but could not be reached."
    return render_template(
        "index.html",
        result=None,
        error=message,
        trained_through=TRAINED_THROUGH,
        database_configured=storage.is_configured(),
        db_status=db_status,
        db_error=db_error,
    )


@app.route("/forecast", methods=["GET", "POST"])
@login_required
def forecast():
    if request.method == "GET":
        return redirect(url_for("home"))
    request_started = perf_counter()
    try:
        requested_date = date.fromisoformat(request.form.get("drive_date", ""))
        position = int(request.form.get("drive_position", "0"))
        if position not in (1, 2):
            raise ValueError("Choose Drive 1 or Drive 2.")

        meta = future_drive_meta(requested_date, position, CONFIG)

        if storage.is_configured():
            state_rows, recent_counts, previous_core_date = storage.fetch_scoring_state(
                requested_date
            )
            score = build_scoring_frame_from_states(
                state_rows, recent_counts, meta, previous_core_date
            )
        else:
            upload = request.files.get("data_file")
            if upload is None or not upload.filename:
                raise ValueError(
                    "Upload the cumulative/lifetime Guest Visit CSV, or configure DATABASE_URL "
                    "to enable one-time initialization and incremental updates."
                )
            if not upload.filename.lower().endswith(".csv"):
                raise ValueError("The uploaded file must be a CSV export.")
            visits, calendar = _load_history(upload)
            latest_visit = max(visits["date"])
            if requested_date <= latest_visit:
                raise ValueError(
                    "The forecast date must be after the latest visit in the uploaded "
                    f"history ({latest_visit.isoformat()})."
                )
            score = build_future_scoring_frame(visits, calendar, meta)

        result = _make_result(score, requested_date, meta)
        app.logger.info(
            "Forecast complete: drive_date=%s position=%s recipients=%s elapsed_seconds=%.2f",
            requested_date.isoformat(), position, result["known_recipients"],
            perf_counter() - request_started,
        )
        db_status = storage.get_status() if storage.is_configured() else None
        return render_template(
            "index.html",
            result=result,
            error=None,
            trained_through=TRAINED_THROUGH,
            database_configured=storage.is_configured(),
            db_status=db_status,
            db_error=None,
        )
    except Exception as exc:
        app.logger.exception("Forecast request failed.")
        return _render_home_error(str(exc)), 400


@app.errorhandler(413)
def too_large(_exc):
    return _render_home_error(
        "That CSV is larger than the 50 MB upload limit. Export only the required Guest Visit columns if possible."
    ), 413


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "incremental-postgres" if storage.is_configured() else "pretrained-fast-scoring",
        "model_trained_through": TRAINED_THROUGH,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
