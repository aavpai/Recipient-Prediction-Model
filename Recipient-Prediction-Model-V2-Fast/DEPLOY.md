# Deploy the Seva Recipient Forecast app

## Architecture

- GitHub stores application code and the pretrained model artifact.
- Render runs the Flask application.
- PostgreSQL stores normalized recipient visit history and compact recipient state.
- The application password protects all dashboard and recipient-data routes.

GitHub Pages cannot run the Python model. Keep the custom domain attached to the Render web service, not GitHub Pages.

## Render service settings

The service root directory must be:

    Recipient-Prediction-Model-V2-Fast

Use:

- Runtime: Python 3
- Build command: pip install -r requirements.txt
- Start command: gunicorn webapp:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 --max-requests 200 --max-requests-jitter 20
- Health check path: /health

## Required environment variables

Set these in Render → your web service → Environment:

| Variable | Value |
|---|---|
| SECRET_KEY | A long random secret. Render can generate one. |
| APP_PASSWORD | A strong password used to sign in to the dashboard. |
| DATABASE_URL | PostgreSQL connection string. Prefer the provider's private/internal URL when available. |

Never commit these values to GitHub or send them in chat.

The included render.yaml declares these variables for Blueprint-based deployments. Existing services may need the variables added manually in the Render dashboard.

## Database choice and durability

The app expects PostgreSQL through DATABASE_URL. A persistent database is required for incremental updates. Do not use Render's Free Postgres for recipient history that must be retained: Free databases expire after 30 days and do not include backups. Use a paid database plan with backups, or another managed PostgreSQL provider with a retention/backup policy you have reviewed.

Keep the database in the same region as the Render web service when possible. Restrict database access to the app where the provider supports private networking or IP allowlists.

## First-time setup

1. Add SECRET_KEY, APP_PASSWORD, and DATABASE_URL in Render.
2. Redeploy the web service.
3. Open the Render URL and sign in using APP_PASSWORD.
4. On the dashboard, upload the cumulative/lifetime Guest Visit export once under “Initialize recipient history”.
5. The app normalizes the visit rows, infers the historical core-drive calendar, and creates recipient state.
6. Check the dashboard counts before proceeding.

Initialization is intentionally one-time. If the baseline file is wrong, do not upload it repeatedly; correct the baseline through a controlled database reset/reinitialization process.

## Normal operation after initialization

After each completed drive:

1. Export that drive's Guest Visit CSV.
2. Upload only that drive's Guest Visit CSV under “Append a completed drive”.
3. Select the actual drive date and position (Drive 1 or Drive 2).
4. The app updates all existing recipients' drive opportunities and attendance state, adds newly observed recipients, and records the drive.
5. Generate a forecast for the next drive without uploading a CSV.

A drive date can only be imported once. Re-uploading an already-imported date is rejected to prevent double-counting.

Special-drive overlays are assigned from the configured calendar:
- April Drive 1: Toy Drive
- July Drive 2: Backpack Drive + TEFAP
- November Drive 2: Thanksgiving Drive + TEFAP
- December Drive 2: Christmas Drive + TEFAP

## Privacy

The database stores Guest IDs, visit dates, household counts, homelessness indicator, and derived recipient state. These are sensitive recipient-level records even though names and street addresses are not part of the model.

- Keep the GitHub repository free of real Guest Visit exports.
- Keep DATABASE_URL, APP_PASSWORD, and SECRET_KEY in Render environment settings only.
- Restrict dashboard access to authorized Seva staff.
- Use a managed database with backups and access controls.
- Do not expose database credentials or recipient exports in screenshots, logs, or issue reports.

The app processes uploaded CSV files in memory and writes normalized records to PostgreSQL only when the user initializes history or imports a drive.

## Temporary mode without PostgreSQL

If DATABASE_URL is not configured, the dashboard can still score a forecast by uploading the cumulative/lifetime export each time. That mode does not retain history between requests and cannot use incremental updates.
