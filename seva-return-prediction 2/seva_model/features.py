from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from datetime import date
import pandas as pd

SPECIALS = ["toy", "backpack", "thanksgiving", "christmas"]

FEATURES = [
    "second_drive", "toy", "backpack", "thanksgiving", "christmas",
    "days_since_prev_core_drive",
    "log_days_since_last_any", "log_days_since_last_core",
    "all_prior_visits", "prior_core_visits", "prior_core_opps",
    "core_attendance_rate", "d1_rate", "d2_rate", "ordinary_d2_affinity",
    "attended_prev_core", "consecutive_core_misses",
    "visits_last_30", "visits_last_60", "visits_last_90", "visits_last_180",
    "total_individuals", "kids", "adults", "seniors", "homeless",
    "prior_toy_attended", "prior_backpack_attended",
    "prior_thanksgiving_attended", "prior_christmas_attended",
    "second_x_affinity", "toy_x_kids", "backpack_x_kids",
    "thanksgiving_x_household", "christmas_x_household",
    "month_sin", "month_cos",
]


def _new_state(d: date, visit: dict) -> dict:
    return {
        "first_date": d,
        "last_any_date": d,
        "last_core_date": None,
        "all_visit_dates": [d],
        "core_opps": 0,
        "core_visits": 0,
        "d1_opps": 0,
        "d2_opps": 0,
        "d1_visits": 0,
        "d2_visits": 0,
        "ordinary_d1_visits": 0,
        "ordinary_d2_visits": 0,
        "attended_prev_core": 0,
        "consecutive_misses": 0,
        "total_individuals": int(visit["total_individuals"]),
        "kids": int(visit["kids"]),
        "adults": int(visit["adults"]),
        "seniors": int(visit["seniors"]),
        "homeless": int(visit["homeless"]),
        "prior_toy_attended": 0,
        "prior_backpack_attended": 0,
        "prior_thanksgiving_attended": 0,
        "prior_christmas_attended": 0,
    }


def _feature_row(gid: str, st: dict, d: date, meta: dict, previous_core_date: date | None) -> dict:
    prior_dates = st["all_visit_dates"]

    def recent(days: int) -> int:
        return sum(1 for x in prior_dates if (d - x).days <= days)

    ordinary_n = st["ordinary_d1_visits"] + st["ordinary_d2_visits"]
    d2_affinity = st["ordinary_d2_visits"] / ordinary_n if ordinary_n else 0.5
    days_last_any = (d - st["last_any_date"]).days
    days_last_core = (d - st["last_core_date"]).days if st["last_core_date"] else 999

    row = {
        "guest_id": gid,
        "date": d,
        "year": d.year,
        "month": d.month,
        "second_drive": int(meta["second_drive"]),
        "toy": int(meta.get("toy", 0)),
        "backpack": int(meta.get("backpack", 0)),
        "thanksgiving": int(meta.get("thanksgiving", 0)),
        "christmas": int(meta.get("christmas", 0)),
        "days_since_prev_core_drive": (d - previous_core_date).days if previous_core_date else 14,
        "days_since_last_any": days_last_any,
        "days_since_last_core": days_last_core,
        "log_days_since_last_any": math.log1p(days_last_any),
        "log_days_since_last_core": math.log1p(days_last_core),
        "all_prior_visits": len(prior_dates),
        "prior_core_visits": st["core_visits"],
        "prior_core_opps": st["core_opps"],
        "core_attendance_rate": st["core_visits"] / st["core_opps"] if st["core_opps"] else 0.0,
        "d1_rate": st["d1_visits"] / st["d1_opps"] if st["d1_opps"] else 0.0,
        "d2_rate": st["d2_visits"] / st["d2_opps"] if st["d2_opps"] else 0.0,
        "ordinary_d2_affinity": d2_affinity,
        "attended_prev_core": st["attended_prev_core"],
        "consecutive_core_misses": st["consecutive_misses"],
        "visits_last_30": recent(30),
        "visits_last_60": recent(60),
        "visits_last_90": recent(90),
        "visits_last_180": recent(180),
        "total_individuals": st["total_individuals"],
        "kids": st["kids"],
        "adults": st["adults"],
        "seniors": st["seniors"],
        "homeless": st["homeless"],
        "prior_toy_attended": st["prior_toy_attended"],
        "prior_backpack_attended": st["prior_backpack_attended"],
        "prior_thanksgiving_attended": st["prior_thanksgiving_attended"],
        "prior_christmas_attended": st["prior_christmas_attended"],
    }
    row["second_x_affinity"] = row["second_drive"] * row["ordinary_d2_affinity"]
    row["toy_x_kids"] = row["toy"] * row["kids"]
    row["backpack_x_kids"] = row["backpack"] * row["kids"]
    row["thanksgiving_x_household"] = row["thanksgiving"] * row["total_individuals"]
    row["christmas_x_household"] = row["christmas"] * row["total_individuals"]
    row["month_sin"] = math.sin(2 * math.pi * d.month / 12)
    row["month_cos"] = math.cos(2 * math.pi * d.month / 12)
    return row


def _visits_by_date(visits: pd.DataFrame) -> dict:
    out = defaultdict(dict)
    for r in visits.to_dict("records"):
        out[r["date"]][str(r["guest_id"])] = r
    return out


def build_training_panel(visits: pd.DataFrame, core_calendar: pd.DataFrame) -> pd.DataFrame:
    by_date = _visits_by_date(visits)
    core_meta = {r["date"]: r for r in core_calendar.to_dict("records")}
    core_attendance = {
        (str(r["guest_id"]), r["date"])
        for r in visits.to_dict("records")
        if r["date"] in core_meta
    }

    states = {}
    rows = []
    previous_core_date = None
    timeline = sorted(set(by_date) | set(core_meta))

    for d in timeline:
        is_core = d in core_meta
        preexisting = []

        if is_core:
            meta = core_meta[d]
            preexisting = [gid for gid, st in states.items() if st["first_date"] < d]
            for gid in preexisting:
                row = _feature_row(gid, states[gid], d, meta, previous_core_date)
                row["target"] = int((gid, d) in core_attendance)
                row["drive_position"] = int(meta["position"])
                rows.append(row)

            # All known recipients have an opportunity on this core drive.
            for gid in preexisting:
                st = states[gid]
                st["core_opps"] += 1
                st["d2_opps" if int(meta["second_drive"]) else "d1_opps"] += 1

        todays = set()
        for gid, v in by_date.get(d, {}).items():
            todays.add(gid)
            if gid not in states:
                states[gid] = _new_state(d, v)
            else:
                st = states[gid]
                st["last_any_date"] = d
                st["all_visit_dates"].append(d)
                for k in ["total_individuals", "kids", "adults", "seniors", "homeless"]:
                    st[k] = int(v[k])

        if is_core:
            meta = core_meta[d]
            # New recipients seen today did have today's opportunity, but were not
            # prediction candidates because this is their first observed visit.
            for gid in todays:
                st = states[gid]
                if st["first_date"] == d:
                    st["core_opps"] += 1
                    st["d2_opps" if int(meta["second_drive"]) else "d1_opps"] += 1

            for gid in preexisting:
                st = states[gid]
                attended = int(gid in todays)
                if attended:
                    _apply_core_attendance(st, d, meta)
                else:
                    st["consecutive_misses"] += 1
                st["attended_prev_core"] = attended

            for gid in todays:
                st = states[gid]
                if st["first_date"] == d:
                    _apply_core_attendance(st, d, meta)
                    st["attended_prev_core"] = 1

            previous_core_date = d

    return pd.DataFrame(rows)


def _apply_core_attendance(st: dict, d: date, meta: dict) -> None:
    st["core_visits"] += 1
    st["last_core_date"] = d
    second = int(meta["second_drive"])
    st["d2_visits" if second else "d1_visits"] += 1

    if not int(meta.get("special", 0)):
        st["ordinary_d2_visits" if second else "ordinary_d1_visits"] += 1

    for name in SPECIALS:
        if int(meta.get(name, 0)):
            st[f"prior_{name}_attended"] = 1
    st["consecutive_misses"] = 0


def build_future_scoring_frame(
    visits: pd.DataFrame,
    core_calendar: pd.DataFrame,
    future_meta: dict,
) -> pd.DataFrame:
    """Reconstruct recipient state through the export and score a future drive.

    All observed visits in the export may update recency. Only inferred/overridden
    core drives update core-drive opportunity and preference features.
    """
    future_date = future_meta["date"]
    if future_date <= max(visits["date"]):
        raise ValueError(
            "Future scoring date must be after the latest visit in the supplied export. "
            "Use validation code for historical drive scoring."
        )

    by_date = _visits_by_date(visits)
    core_meta = {r["date"]: r for r in core_calendar.to_dict("records")}
    states = {}
    previous_core_date = None

    for d in sorted(set(by_date) | set(core_meta)):
        if d >= future_date:
            break
        is_core = d in core_meta
        meta = core_meta.get(d)
        preexisting = [gid for gid, st in states.items() if st["first_date"] < d] if is_core else []

        if is_core:
            for gid in preexisting:
                st = states[gid]
                st["core_opps"] += 1
                st["d2_opps" if int(meta["second_drive"]) else "d1_opps"] += 1

        todays = set()
        for gid, v in by_date.get(d, {}).items():
            todays.add(gid)
            if gid not in states:
                states[gid] = _new_state(d, v)
            else:
                st = states[gid]
                st["last_any_date"] = d
                st["all_visit_dates"].append(d)
                for k in ["total_individuals", "kids", "adults", "seniors", "homeless"]:
                    st[k] = int(v[k])

        if is_core:
            for gid in todays:
                st = states[gid]
                if st["first_date"] == d:
                    st["core_opps"] += 1
                    st["d2_opps" if int(meta["second_drive"]) else "d1_opps"] += 1

            for gid in preexisting:
                st = states[gid]
                attended = int(gid in todays)
                if attended:
                    _apply_core_attendance(st, d, meta)
                else:
                    st["consecutive_misses"] += 1
                st["attended_prev_core"] = attended

            for gid in todays:
                st = states[gid]
                if st["first_date"] == d:
                    _apply_core_attendance(st, d, meta)
                    st["attended_prev_core"] = 1

            previous_core_date = d

    rows = []
    for gid, st in states.items():
        if st["first_date"] < future_date:
            row = _feature_row(gid, st, future_date, future_meta, previous_core_date)
            row["drive_position"] = int(future_meta["position"])
            rows.append(row)
    return pd.DataFrame(rows)
