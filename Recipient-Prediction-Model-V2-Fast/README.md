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
