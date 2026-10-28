from __future__ import annotations

from pathlib import Path
import calendar as pycalendar
import pandas as pd

from .config import special_flags


def _weekday_number(name: str) -> int:
    names = {name.lower(): i for i, name in enumerate(pycalendar.day_name)}
    try:
        return names[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown weekday in config: {name}") from exc


def load_overrides(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if not {"date", "position"}.issubset(df.columns):
        raise ValueError("Calendar override CSV must contain date,position")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["position"] = df["position"].astype(int)
    return df


def infer_core_calendar(
    visits: pd.DataFrame,
    config: dict,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Infer the two core monthly drive dates from observed attendance.

    Default heuristic: among configured-weekday distributions in each month,
    select the N dates with the largest attendance and order them chronologically.
    Exact dates can be supplied in an override CSV.
    """
    counts = visits.groupby("date")["guest_id"].nunique().rename("attendance").reset_index()
    counts["date"] = pd.to_datetime(counts["date"]).dt.date
    counts["year"] = counts["date"].map(lambda d: d.year)
    counts["month"] = counts["date"].map(lambda d: d.month)
    counts["weekday"] = counts["date"].map(lambda d: d.weekday())

    target_weekday = _weekday_number(config.get("core_drive_weekday", "Saturday"))
    n_core = int(config.get("core_drives_per_month", 2))

    rows = []
    for (year, month), g in counts.groupby(["year", "month"]):
        override_month = None
        if overrides is not None:
            override_month = overrides[
                overrides["date"].map(lambda d: (d.year, d.month) == (year, month))
            ]

        if override_month is not None and not override_month.empty:
            selected = override_month.sort_values("position")
            for _, r in selected.iterrows():
                date = r["date"]
                att = counts.loc[counts["date"] == date, "attendance"]
                rows.append(
                    {
                        "date": date,
                        "year": year,
                        "month": month,
                        "position": int(r["position"]),
                        "attendance": int(att.iloc[0]) if len(att) else 0,
                        "calendar_source": "override",
                    }
                )
            continue

        candidates = g[g["weekday"] == target_weekday]
        if len(candidates) < n_core:
            continue
        selected = candidates.nlargest(n_core, "attendance").sort_values("date")
        for pos, (_, r) in enumerate(selected.iterrows(), start=1):
            rows.append(
                {
                    "date": r["date"],
                    "year": year,
                    "month": month,
                    "position": pos,
                    "attendance": int(r["attendance"]),
                    "calendar_source": "inferred",
                }
            )

    cal = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if cal.empty:
        raise ValueError("No core drives could be inferred from this export.")

    cal["second_drive"] = (cal["position"] == 2).astype(int)
    for special_name in config.get("special_overlays", {}):
        cal[special_name] = [
            special_flags(m, p, config)[special_name]
            for m, p in zip(cal["month"], cal["position"])
        ]
    special_cols = list(config.get("special_overlays", {}).keys())
    cal["special"] = cal[special_cols].max(axis=1) if special_cols else 0
    return cal


def future_drive_meta(d, position: int, config: dict) -> dict:
    flags = special_flags(d.month, position, config)
    return {
        "date": d,
        "year": d.year,
        "month": d.month,
        "position": int(position),
        "second_drive": int(position == 2),
        **flags,
        "special": int(any(flags.values())),
    }
