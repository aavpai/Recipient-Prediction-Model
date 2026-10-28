from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Visit ID",
    "Outreach on",
    "Guest ID",
    "Tracking Method",
    "Total Individuals",
    "0 to 17",
    "18 to 61",
    "62+",
    "Homeless",
]


def load_raw_export(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def collapse_to_visits(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse Pantry/TEFAP tracking rows into one actual visit.

    A Visit ID can appear multiple times because tracking methods are recorded as
    separate rows. The model must not count those rows as separate attendances.
    A second deduplication collapses multiple Visit IDs for the same Guest ID on
    the same calendar date into one attendance opportunity.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["Outreach on"], errors="raise").dt.date

    for c in ["Total Individuals", "0 to 17", "18 to 61", "62+"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["homeless"] = df["Homeless"].str.strip().str.lower().eq("yes").astype(int)

    # One record per Visit ID first.
    visits = (
        df.sort_values(["Visit ID", "Tracking Method"])
        .drop_duplicates(subset=["Visit ID"], keep="first")
        .rename(
            columns={
                "Guest ID": "guest_id",
                "Visit ID": "visit_id",
                "Outreach Name": "outreach_name",
                "Total Individuals": "total_individuals",
                "0 to 17": "kids",
                "18 to 61": "adults",
                "62+": "seniors",
            }
        )
    )

    cols = [
        "visit_id",
        "guest_id",
        "date",
        "outreach_name",
        "total_individuals",
        "kids",
        "adults",
        "seniors",
        "homeless",
    ]
    visits = visits[cols].copy()

    # Same recipient on same day = one attendance for modeling purposes.
    visits = (
        visits.sort_values(["guest_id", "date", "visit_id"])
        .drop_duplicates(subset=["guest_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return visits
