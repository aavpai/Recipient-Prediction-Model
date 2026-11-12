from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .features import build_recipient_state_snapshot


STATE_COLUMNS = [
    "guest_id",
    "first_date",
    "last_any_date",
    "last_core_date",
    "all_prior_visits",
    "core_opps",
    "core_visits",
    "d1_opps",
    "d2_opps",
    "d1_visits",
    "d2_visits",
    "ordinary_d1_visits",
    "ordinary_d2_visits",
    "attended_prev_core",
    "consecutive_misses",
    "total_individuals",
    "kids",
    "adults",
    "seniors",
    "homeless",
    "prior_toy_attended",
    "prior_backpack_attended",
    "prior_thanksgiving_attended",
    "prior_christmas_attended",
]

VISIT_INSERT = """
INSERT INTO recipient_visits (
    guest_id, visit_date, visit_id, total_individuals, kids, adults,
    seniors, homeless, outreach_name
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (guest_id, visit_date) DO UPDATE SET
    visit_id = EXCLUDED.visit_id,
    total_individuals = EXCLUDED.total_individuals,
    kids = EXCLUDED.kids,
    adults = EXCLUDED.adults,
    seniors = EXCLUDED.seniors,
    homeless = EXCLUDED.homeless,
    outreach_name = EXCLUDED.outreach_name
"""

STATE_INSERT = """
INSERT INTO recipient_state (
    guest_id, first_date, last_any_date, last_core_date, all_prior_visits,
    core_opps, core_visits, d1_opps, d2_opps, d1_visits, d2_visits,
    ordinary_d1_visits, ordinary_d2_visits, attended_prev_core,
    consecutive_misses, total_individuals, kids, adults, seniors, homeless,
    prior_toy_attended, prior_backpack_attended,
    prior_thanksgiving_attended, prior_christmas_attended
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""

STATE_UPSERT_ATTENDEE = """
INSERT INTO recipient_state (
    guest_id, first_date, last_any_date, last_core_date, all_prior_visits,
    core_opps, core_visits, d1_opps, d2_opps, d1_visits, d2_visits,
    ordinary_d1_visits, ordinary_d2_visits, attended_prev_core,
    consecutive_misses, total_individuals, kids, adults, seniors, homeless,
    prior_toy_attended, prior_backpack_attended,
    prior_thanksgiving_attended, prior_christmas_attended
) VALUES (
    %s, %s, %s, %s, 1, 1, 1, %s, %s, %s, %s, %s, %s,
    1, 0, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (guest_id) DO UPDATE SET
    last_any_date = EXCLUDED.last_any_date,
    last_core_date = EXCLUDED.last_core_date,
    all_prior_visits = recipient_state.all_prior_visits + 1,
    core_visits = recipient_state.core_visits + 1,
    d1_visits = recipient_state.d1_visits + EXCLUDED.d1_visits,
    d2_visits = recipient_state.d2_visits + EXCLUDED.d2_visits,
    ordinary_d1_visits = recipient_state.ordinary_d1_visits + EXCLUDED.ordinary_d1_visits,
    ordinary_d2_visits = recipient_state.ordinary_d2_visits + EXCLUDED.ordinary_d2_visits,
    attended_prev_core = 1,
    consecutive_misses = 0,
    total_individuals = EXCLUDED.total_individuals,
    kids = EXCLUDED.kids,
    adults = EXCLUDED.adults,
    seniors = EXCLUDED.seniors,
    homeless = EXCLUDED.homeless,
    prior_toy_attended = GREATEST(recipient_state.prior_toy_attended, EXCLUDED.prior_toy_attended),
    prior_backpack_attended = GREATEST(recipient_state.prior_backpack_attended, EXCLUDED.prior_backpack_attended),
    prior_thanksgiving_attended = GREATEST(recipient_state.prior_thanksgiving_attended, EXCLUDED.prior_thanksgiving_attended),
    prior_christmas_attended = GREATEST(recipient_state.prior_christmas_attended, EXCLUDED.prior_christmas_attended)
"""

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS seva_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recipient_visits (
        guest_id TEXT NOT NULL,
        visit_date DATE NOT NULL,
        visit_id TEXT NOT NULL DEFAULT '',
        total_individuals INTEGER NOT NULL DEFAULT 0,
        kids INTEGER NOT NULL DEFAULT 0,
        adults INTEGER NOT NULL DEFAULT 0,
        seniors INTEGER NOT NULL DEFAULT 0,
        homeless SMALLINT NOT NULL DEFAULT 0,
        outreach_name TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (guest_id, visit_date)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS recipient_visits_date_idx
    ON recipient_visits (visit_date)
    """,
    """
    CREATE TABLE IF NOT EXISTS core_drives (
        drive_date DATE PRIMARY KEY,
        position SMALLINT NOT NULL CHECK (position IN (1, 2)),
        second_drive SMALLINT NOT NULL,
        toy SMALLINT NOT NULL DEFAULT 0,
        backpack SMALLINT NOT NULL DEFAULT 0,
        thanksgiving SMALLINT NOT NULL DEFAULT 0,
        christmas SMALLINT NOT NULL DEFAULT 0,
        special SMALLINT NOT NULL DEFAULT 0,
        calendar_source TEXT NOT NULL DEFAULT 'inferred'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recipient_state (
        guest_id TEXT PRIMARY KEY,
        first_date DATE NOT NULL,
        last_any_date DATE NOT NULL,
        last_core_date DATE,
        all_prior_visits INTEGER NOT NULL,
        core_opps INTEGER NOT NULL,
        core_visits INTEGER NOT NULL,
        d1_opps INTEGER NOT NULL,
        d2_opps INTEGER NOT NULL,
        d1_visits INTEGER NOT NULL,
        d2_visits INTEGER NOT NULL,
        ordinary_d1_visits INTEGER NOT NULL,
        ordinary_d2_visits INTEGER NOT NULL,
        attended_prev_core SMALLINT NOT NULL,
        consecutive_misses INTEGER NOT NULL,
        total_individuals INTEGER NOT NULL,
        kids INTEGER NOT NULL,
        adults INTEGER NOT NULL,
        seniors INTEGER NOT NULL,
        homeless SMALLINT NOT NULL,
        prior_toy_attended SMALLINT NOT NULL,
        prior_backpack_attended SMALLINT NOT NULL,
        prior_thanksgiving_attended SMALLINT NOT NULL,
        prior_christmas_attended SMALLINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_log (
        import_id BIGSERIAL PRIMARY KEY,
        import_type TEXT NOT NULL,
        drive_date DATE,
        drive_position SMALLINT,
        file_sha256 TEXT NOT NULL,
        visits_loaded INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
]


def is_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured.")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return psycopg.connect(url, row_factory=dict_row, connect_timeout=10)


def ensure_schema() -> None:
    with _connect() as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


def _get_meta(conn, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM seva_meta WHERE key = %s", (key,)
    ).fetchone()
    return row["value"] if row else None


def _set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO seva_meta (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        (key, value),
    )


def _visit_values(visits) -> list[tuple]:
    values = []
    for row in visits.to_dict("records"):
        guest_id = str(row["guest_id"]).strip()
        if not guest_id or guest_id.lower() == "nan":
            raise ValueError("The CSV contains a visit with a blank Guest ID.")
        values.append((
            guest_id,
            row["date"],
            str(row.get("visit_id", "")),
            int(row["total_individuals"]),
            int(row["kids"]),
            int(row["adults"]),
            int(row["seniors"]),
            int(row["homeless"]),
            str(row.get("outreach_name", "")),
        ))
    return values


def _drive_values(calendar) -> list[tuple]:
    values = []
    for row in calendar.to_dict("records"):
        values.append((
            row["date"],
            int(row["position"]),
            int(row["second_drive"]),
            int(row.get("toy", 0)),
            int(row.get("backpack", 0)),
            int(row.get("thanksgiving", 0)),
            int(row.get("christmas", 0)),
            int(row.get("special", 0)),
            str(row.get("calendar_source", "inferred")),
        ))
    return values


def _state_values(state_rows: list[dict]) -> list[tuple]:
    return [tuple(row[column] for column in STATE_COLUMNS) for row in state_rows]


def _insert_visits(conn, values: list[tuple]) -> None:
    with conn.cursor() as cur:
        cur.executemany(VISIT_INSERT, values)


def _insert_drives(conn, values: list[tuple]) -> None:
    if not values:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO core_drives (
                drive_date, position, second_drive, toy, backpack,
                thanksgiving, christmas, special, calendar_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (drive_date) DO NOTHING
            """,
            values,
        )


def _insert_states(conn, values: list[tuple]) -> None:
    if not values:
        return
    with conn.cursor() as cur:
        cur.executemany(STATE_INSERT, values)


def get_status() -> dict[str, Any]:
    if not is_configured():
        return {"configured": False, "initialized": False}
    ensure_schema()
    with _connect() as conn:
        initialized = _get_meta(conn, "initialized") == "true"
        latest_core = _get_meta(conn, "last_core_date")
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM recipient_state) AS recipients,
                (SELECT COUNT(*) FROM recipient_visits) AS visits,
                (SELECT COUNT(*) FROM core_drives) AS core_drives,
                (SELECT MAX(visit_date) FROM recipient_visits) AS latest_visit
            """
        ).fetchone()
    return {
        "configured": True,
        "initialized": initialized,
        "recipients": int(counts["recipients"]),
        "visits": int(counts["visits"]),
        "core_drives": int(counts["core_drives"]),
        "latest_visit": counts["latest_visit"],
        "latest_core_drive": date.fromisoformat(latest_core) if latest_core else None,
    }


def initialize_history(visits, calendar, file_sha256: str) -> dict[str, int]:
    """Initialize persistent state once from the cumulative/lifetime export."""
    if not is_configured():
        raise RuntimeError("Persistent storage is not configured. Set DATABASE_URL first.")
    if visits.empty or calendar.empty:
        raise ValueError("The lifetime export does not contain enough visit history.")

    state_rows, previous_core_date = build_recipient_state_snapshot(visits, calendar)
    visit_values = _visit_values(visits)
    drive_values = _drive_values(calendar)
    state_values = _state_values(state_rows)

    ensure_schema()
    with _connect() as conn:
        if _get_meta(conn, "initialized") == "true":
            raise ValueError(
                "Lifetime history has already been initialized. Use the latest-drive update form instead."
            )
        _insert_visits(conn, visit_values)
        _insert_drives(conn, drive_values)
        _insert_states(conn, state_values)
        _set_meta(conn, "initialized", "true")
        if previous_core_date:
            _set_meta(conn, "last_core_date", previous_core_date.isoformat())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO import_log (
                    import_type, file_sha256, visits_loaded
                ) VALUES ('baseline', %s, %s)
                """,
                (file_sha256, len(visit_values)),
            )

    return {"recipients": len(state_values), "visits": len(visit_values), "core_drives": len(drive_values)}


def apply_drive(visits, drive_meta: dict, file_sha256: str) -> dict[str, int]:
    """Append one new drive to saved state; never replay the lifetime history."""
    if not is_configured():
        raise RuntimeError("Persistent storage is not configured. Set DATABASE_URL first.")
    if visits.empty:
        raise ValueError("The drive export contains no usable recipient visits.")
    if visits["date"].nunique() != 1 or visits["date"].iloc[0] != drive_meta["date"]:
        raise ValueError("Every row in this upload must belong to the selected drive date.")

    visit_values = _visit_values(visits)
    d = drive_meta["date"]
    position = int(drive_meta["position"])
    special = int(drive_meta.get("special", 0))
    d1 = int(position == 1)
    d2 = int(position == 2)
    ordinary_d1 = int(d1 and not special)
    ordinary_d2 = int(d2 and not special)

    ensure_schema()
    with _connect() as conn:
        if _get_meta(conn, "initialized") != "true":
            raise ValueError("Upload the lifetime export first to initialize recipient history.")

        latest_core = _get_meta(conn, "last_core_date")
        if latest_core and d <= date.fromisoformat(latest_core):
            raise ValueError(
                f"This drive date must be later than the last imported core drive ({latest_core})."
            )

        latest_visit = conn.execute(
            "SELECT MAX(visit_date) AS latest_visit FROM recipient_visits"
        ).fetchone()["latest_visit"]
        if latest_visit and d <= latest_visit:
            raise ValueError(
                f"This drive date must be later than the latest imported visit ({latest_visit})."
            )

        duplicate = conn.execute(
            "SELECT 1 FROM core_drives WHERE drive_date = %s", (d,)
        ).fetchone()
        if duplicate:
            raise ValueError("This drive date has already been imported. No records were changed.")

        # Everyone already in the system receives one new opportunity.
        conn.execute(
            """
            UPDATE recipient_state
            SET core_opps = core_opps + 1,
                d1_opps = d1_opps + %s,
                d2_opps = d2_opps + %s,
                consecutive_misses = consecutive_misses + 1,
                attended_prev_core = 0
            """,
            (d1, d2),
        )

        attendee_values = []
        for row in visits.to_dict("records"):
            attendee_values.append((
                str(row["guest_id"]).strip(),
                d, d,
                d1, d2, d1, d2, ordinary_d1, ordinary_d2,
                int(row["total_individuals"]),
                int(row["kids"]),
                int(row["adults"]),
                int(row["seniors"]),
                int(row["homeless"]),
                int(drive_meta.get("toy", 0)),
                int(drive_meta.get("backpack", 0)),
                int(drive_meta.get("thanksgiving", 0)),
                int(drive_meta.get("christmas", 0)),
            ))
        with conn.cursor() as cur:
            cur.executemany(STATE_UPSERT_ATTENDEE, attendee_values)

        _insert_visits(conn, visit_values)
        _insert_drives(conn, [(
            d, position, int(drive_meta["second_drive"]),
            int(drive_meta.get("toy", 0)),
            int(drive_meta.get("backpack", 0)),
            int(drive_meta.get("thanksgiving", 0)),
            int(drive_meta.get("christmas", 0)),
            special, "incremental_upload",
        )])
        _set_meta(conn, "last_core_date", d.isoformat())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO import_log (
                    import_type, drive_date, drive_position, file_sha256, visits_loaded
                ) VALUES ('drive_update', %s, %s, %s, %s)
                """,
                (d, position, file_sha256, len(visit_values)),
            )

    return {"attendees": len(visit_values), "drive_position": position}


def fetch_scoring_state(future_date: date) -> tuple[list[dict], date | None]:
    """Fetch compact state plus rolling counts for one upcoming drive."""
    if not is_configured():
        raise RuntimeError("Persistent storage is not configured. Set DATABASE_URL first.")
    ensure_schema()
    with _connect() as conn:
        if _get_meta(conn, "initialized") != "true":
            raise ValueError("Initialize the lifetime export before forecasting.")

        latest_visit = conn.execute(
            "SELECT MAX(visit_date) AS latest_visit FROM recipient_visits"
        ).fetchone()["latest_visit"]
        if latest_visit and future_date <= latest_visit:
            raise ValueError(
                f"Forecast date must be after the latest imported visit ({latest_visit})."
            )

        cutoff_180 = future_date - timedelta(days=180)
        rows = conn.execute(
            """
            SELECT
                s.*,
                COUNT(v.visit_date) FILTER (WHERE v.visit_date >= %s) AS visits_last_30,
                COUNT(v.visit_date) FILTER (WHERE v.visit_date >= %s) AS visits_last_60,
                COUNT(v.visit_date) FILTER (WHERE v.visit_date >= %s) AS visits_last_90,
                COUNT(v.visit_date) AS visits_last_180
            FROM recipient_state AS s
            LEFT JOIN recipient_visits AS v
                ON v.guest_id = s.guest_id
                AND v.visit_date >= %s
                AND v.visit_date <= %s
            GROUP BY s.guest_id
            ORDER BY s.guest_id
            """,
            (
                future_date - timedelta(days=30),
                future_date - timedelta(days=60),
                future_date - timedelta(days=90),
                cutoff_180,
                future_date,
            ),
        ).fetchall()
        previous = _get_meta(conn, "last_core_date")

    state_rows = [dict(row) for row in rows]
    recent_counts = {
        row["guest_id"]: {
            "visits_last_30": int(row["visits_last_30"]),
            "visits_last_60": int(row["visits_last_60"]),
            "visits_last_90": int(row["visits_last_90"]),
            "visits_last_180": int(row["visits_last_180"]),
        }
        for row in state_rows
    }
    return state_rows, recent_counts, date.fromisoformat(previous) if previous else None
