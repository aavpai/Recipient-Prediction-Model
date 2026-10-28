[README.md](https://github.com/user-attachments/files/34099940/README.md)
# Seva Return Prediction

A reproducible model for forecasting whether **previously observed Seva recipients** will attend a specific upcoming core food distribution drive.

## What the model predicts

For every known recipient `i` and upcoming core drive `t`, the model estimates:

`P(recipient i attends drive t | recipient history + upcoming drive characteristics)`

Adding those recipient-level probabilities gives the forecast number of **returning households** for the drive.

The model does **not** currently forecast first-time households. Total attendance therefore requires a separate new-household forecast:

`Total households = predicted returning households + predicted new households`

## Seva drive structure encoded in V1

- Two core drives each month.
- Drive 2 is always the **TEFAP / extra-food** drive.
- April Drive 1: **Toy Drive**.
- July Drive 2: **Backpack Drive + TEFAP**.
- November Drive 2: **Thanksgiving Drive + TEFAP**.
- December Drive 2: **Christmas Drive + TEFAP**.

Because Drive 2 and TEFAP always occur together in the historical data, the model cannot causally separate a pure "later in the month" effect from the TEFAP/more-food effect.

## Main features

The feature set includes:

- Days since last visit and last core drive.
- Visits in the last 30/60/90/180 days.
- Lifetime visits to date.
- Core-drive attendance rate.
- Drive 1 and Drive 2 historical attendance rates.
- **Ordinary Drive-2 affinity**, calculated outside special-drive observations.
- Whether the recipient attended the previous core drive.
- Consecutive missed core drives.
- Latest household size and counts of children/adults/seniors.
- Prior attendance at Toy/Backpack/Thanksgiving/Christmas drives.
- Upcoming drive type and special-drive interactions.
- Month seasonality.

Recipient names, addresses, and dates of birth are **not model features**.

## Reference V1 validation on the current Seva export

Using a chronological split (train: 2024–2025; test: 2026), the current repository reproduces the V1 results at approximately:

- Gradient boosting ROC AUC: **0.931**.
- Gradient boosting PR AUC: **0.548**.
- Active-within-180-days ROC AUC: **0.851**.

Aggregate, privacy-safe reference outputs are in `outputs/reference_v1/`. They contain drive-level counts only, not recipient identities.

## Models

V1 compares:

1. Standardized logistic regression as an interpretable baseline.
2. Histogram gradient boosting as the main nonlinear prediction model.

The validation split is chronological rather than random to reduce leakage from the same households appearing throughout the dataset.

## Repository layout

```text
seva-return-prediction/
├── README.md
├── SECURITY.md
├── requirements.txt
├── Makefile
├── config/
│   ├── drive_calendar.yaml
│   └── drive_calendar_overrides.example.csv
├── seva_model/
│   ├── calendar.py
│   ├── config.py
│   ├── features.py
│   ├── io.py
│   ├── modeling.py
│   ├── predict.py
│   └── train.py
├── scripts/
│   └── generate_synthetic_data.py
├── sample_data/
│   └── synthetic_guest_visits.csv
├── tests/
│   └── test_pipeline.py
├── outputs/
│   └── reference_v1/
├── data/                 # private; ignored by Git
└── models/               # generated locally; ignored by Git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

## Try it with synthetic data

```bash
python scripts/generate_synthetic_data.py
python -m seva_model.train \
  --data sample_data/synthetic_guest_visits.csv \
  --train-start 2024-01-01 \
  --train-end 2025-12-31 \
  --test-start 2026-01-01
```

## Train on the real Seva export

1. Put the private export in `data/guest_visits.csv`.
2. Never commit that file.
3. Run:

```bash
python -m seva_model.train \
  --data data/guest_visits.csv \
  --train-start 2024-01-01 \
  --train-end 2025-12-31 \
  --test-start 2026-01-01
```

This produces:

- `models/seva_return_model.joblib`
- `outputs/model_metrics.csv`
- `outputs/drive_validation.csv`
- `outputs/inferred_drive_calendar.csv`

## Forecast an upcoming drive

Example: November Drive 1

```bash
python -m seva_model.predict \
  --data data/guest_visits.csv \
  --model models/seva_return_model.joblib \
  --date 2026-11-14 \
  --drive-position 1
```

Example: November Drive 2, automatically tagged as **TEFAP + Thanksgiving**:

```bash
python -m seva_model.predict \
  --data data/guest_visits.csv \
  --model models/seva_return_model.joblib \
  --date 2026-11-28 \
  --drive-position 2
```

The forecast summary contains:

- Number of known recipients scored.
- Expected returning households, calculated as `sum(p_i)`.
- A model-based 80% interval under an independence approximation.
- Expected people in returning households, calculated as `sum(p_i * household_size_i)`.
- Drive attributes and special overlays.

The recipient-level CSV is saved locally and contains Guest IDs, so it is ignored by Git by default.

## Core-drive calendar

The raw export does not contain a reliable explicit `Drive 1` / `Drive 2` flag. By default V1 identifies the two core drives in each month as the **two largest Saturday distributions**, then orders them chronologically.

For production use, exact dates are better. Create a CSV like:

```csv
date,position
2026-01-10,1
2026-01-24,2
```

and pass it with:

```bash
--calendar-overrides config/my_exact_drive_calendar.csv
```

That replaces the heuristic for months included in the override file.

## Interpretation caution

A high Drive-2 affinity means a recipient has historically favored the second/TEFAP/more-food drive. It should **not** be labeled a "paycheck-to-paycheck score." The current observational data do not identify why a household favors Drive 2.

## Privacy

See [SECURITY.md](SECURITY.md). No real recipient data are included in this repository.
