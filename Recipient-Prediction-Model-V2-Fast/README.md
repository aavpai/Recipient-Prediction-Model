# Seva Recipient Forecast — Fast V2

A password-protected web app for estimating the probability that previously observed Seva recipients return for a specific upcoming distribution drive.

## Forecast target

For each known recipient i and upcoming drive t:

    P(recipient i attends drive t | recipient history + upcoming drive characteristics)

Summing recipient probabilities gives the expected number of returning households. This is not a total-attendance forecast; it does not include first-time households.

## Drive calendar

- Two core drives per month.
- Drive 2 is always TEFAP / extra food.
- April Drive 1: Toy Drive.
- July Drive 2: Backpack Drive + TEFAP.
- November Drive 2: Thanksgiving Drive + TEFAP.
- December Drive 2: Christmas Drive + TEFAP.

Drive 2 and TEFAP always occur together historically, so the model cannot separately identify a later-in-month effect from the extra-food/TEFAP effect.

## Faster request path

The web app loads a pretrained model once at startup. Forecast requests do not retrain the model. CSV imports select only fields required by the pipeline and hash the upload as a stream rather than copying the entire file into memory.

The serialized model was created with scikit-learn 1.8.0, so dependencies pin that version to avoid incompatible model deserialization.

## Incremental history mode

When DATABASE_URL is configured:

1. Upload the cumulative/lifetime Guest Visit export once.
2. The app stores normalized visit rows and compact recipient states in PostgreSQL.
3. After each completed drive, upload only that drive's CSV and select its date and position.
4. The app updates recipient state incrementally; it does not replay the entire lifetime history.
5. Forecast the next drive without uploading a CSV.

A drive date can only be imported once. Duplicate imports are rejected.

Without PostgreSQL, the app retains a temporary fast-scoring mode that requires the cumulative/lifetime export for every forecast.

## Security and privacy

The dashboard requires APP_PASSWORD and SECRET_KEY. Recipient-level records are stored in PostgreSQL and include Guest IDs, visit dates, household counts, and a homelessness indicator.

Never commit real Guest Visit exports, database URLs, passwords, or secret keys to GitHub. Keep access restricted to authorized Seva staff. Use a database plan with a backup/retention policy appropriate for recipient-level data; Render Free Postgres expires after 30 days and has no backups.

## Deployment

See DEPLOY.md for Render configuration, database setup, initialization, and ongoing use.

Required Render environment variables:

- SECRET_KEY
- APP_PASSWORD
- DATABASE_URL

The Render service root directory must be Recipient-Prediction-Model-V2-Fast.

## Development and tests

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q

GitHub Actions runs the test suite on pushes and pull requests.


## Historical walk-forward backtesting

The model forecasts the probability that a **previously observed recipient** attends a future core drive. Summing those probabilities estimates returning households; it does not include first-time recipients or total attendance.

The backtesting command evaluates the model one drive at a time. For each test drive, it trains a fresh model using only recipient-drive observations from earlier drives, predicts the next drive, records actual returning recipients, and advances the history. It also compares the model with a simple benchmark based on prior-drive attendance and the upcoming drive position.

Run locally with the cumulative/lifetime Guest Visit export:

```bash
python -m seva_model.backtest \
  --data "/path/to/Guest-Visit-Export.csv" \
  --start-date 2025-01-01 \
  --end-date 2026-09-26 \
  --calendar-overrides config/drive_calendar_overrides.csv
```

The command writes three aggregate-only files:

- `outputs/walk_forward_drive_results.csv` — one row per drive, including the training cutoff and predicted versus actual returning households.
- `outputs/walk_forward_summary.json` — drive-level errors, recipient-level discrimination and probability metrics, and baseline comparison.
- `outputs/walk_forward_calibration.csv` — predicted versus observed return rates by probability bin.

No Guest IDs or recipient-level predictions are exported. **Never commit the real Guest Visit export**; it contains sensitive recipient information.

The fixed historical calendar override prevents event classification from changing based on observed attendance during the backtest. It excludes the 2024-12-14 mini-drive. The June 2025 export contains only one Saturday distribution and is omitted pending schedule verification. Update the override file when confirmed drive dates become available.

The default backtest begins in 2025 and requires at least 12 earlier core drives for training. Use `--min-training-drives` to change that threshold.
