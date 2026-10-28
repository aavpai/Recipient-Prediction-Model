# Seva Recipient Forecast — Fast V2

A web app for forecasting how many **previously observed Seva recipients** are likely to attend a specific upcoming core food distribution drive.

## Why V2 is faster

The old web version retrained the full gradient-boosting model every time a user clicked **Generate Forecast**. On the real lifetime export that training step takes roughly 25 seconds and can exceed low-cost web-host request/resource limits.

V2 uses **train once, predict many times**:

1. A validated model is trained offline and saved in `models/seva_operational_model.joblib`.
2. The web app loads that model once when the server starts.
3. When a cumulative/lifetime Seva CSV is uploaded, the app only cleans the data, reconstructs current recipient histories, and scores the upcoming drive.
4. No model fitting occurs during the web request.

On the current lifetime export, the scoring pipeline takes only a few seconds in local testing rather than ~25 seconds for retraining.

## What to upload

For the current V2 app, upload the **current cumulative/lifetime Guest Visit export**, not only the most recent drive. The model needs prior history to calculate recency, frequency, Drive-1/Drive-2 behavior, missed drives, special-drive history, and household composition.

The uploaded CSV is processed in memory and is **not committed to GitHub**.

## Drive structure

- Two core drives per month.
- Drive 2 is always TEFAP / extra food.
- April Drive 1: Toy Drive.
- July Drive 2: Backpack Drive + TEFAP.
- November Drive 2: Thanksgiving Drive + TEFAP.
- December Drive 2: Christmas Drive + TEFAP.

## Model target

For every known recipient `i` and upcoming drive `t`:

`P(recipient i attends drive t | recipient history + upcoming drive characteristics)`

Summing recipient probabilities gives the expected number of **returning households**. The model does not yet forecast first-time households.

## Render deployment

Use these settings:

- **Root Directory:** the directory containing `webapp.py` and `requirements.txt`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn webapp:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
- **Health Check Path:** `/health`

The included `render.yaml` contains the same configuration and pins Python 3.12.8.

## Privacy

Do not commit the real Seva recipient CSV to GitHub. Keep the repository private if you want the fitted model and project code restricted as well.

## Retraining

The bundled operational model is trained through the current historical export used to build V2. As new months accumulate, retrain offline using the local training scripts, validate chronologically, then replace `models/seva_operational_model.joblib`.
