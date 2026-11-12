from __future__ import annotations

from pathlib import Path
from typing import IO

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

OPTIONAL_COLUMNS = ["Outreach Name"]


def load_raw_export(path: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read only fields used by the forecasting pipeline.

    Restricting columns matters on small web instances: real exports can contain
    extra administrative fields that are not needed for prediction. The input
    can be a path or a seekable upload stream.
    """
    wanted = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in wanted,
        low_memory=False,
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
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
    needed = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = raw.loc[:, [column for column in needed if column in raw.columns]].copy()
    df["date"] = pd.to_datetime(df["Outreach on"], errors="raise").dt.date

    numeric_columns = ["Total Individuals", "0 to 17", "18 to 61", "62+"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int32")

    df["homeless"] = (
        df["Homeless"].str.strip().str.lower().eq("yes").astype("int8")
    )

    rename_map = {
        "Guest ID": "guest_id",
        "Visit ID": "visit_id",
        "Outreach Name": "outreach_name",
        "Total Individuals": "total_individuals",
        "0 to 17": "kids",
        "18 to 61": "adults",
        "62+": "seniors",
    }

    # One record per Visit ID first. The export can contain several tracking
    # rows for a single visit; those rows are not separate household arrivals.
    visits = (
        df.sort_values(["Visit ID", "Tracking Method"], kind="mergesort")
        .drop_duplicates(subset=["Visit ID"], keep="first")
        .rename(columns=rename_map)
    )

    if "outreach_name" not in visits.columns:
        visits["outreach_name"] = ""

    columns = [
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
    visits = visits.loc[:, columns]

    # Same recipient on the same date = one attendance for modeling purposes.
    visits = (
        visits.sort_values(["guest_id", "date", "visit_id"], kind="mergesort")
        .drop_duplicates(subset=["guest_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return visits
