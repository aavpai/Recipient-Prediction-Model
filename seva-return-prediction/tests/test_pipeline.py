from datetime import date
import pandas as pd

from seva_model.config import special_flags
from seva_model.io import collapse_to_visits


def test_duplicate_tracking_rows_are_one_visit():
    raw = pd.DataFrame([
        {
            "Visit ID":"1", "Outreach on":"2026-01-10 10:00:00", "Guest ID":"A",
            "Tracking Method":"Pantry Food", "Total Individuals":"3", "0 to 17":"1",
            "18 to 61":"2", "62+":"0", "Homeless":"No", "Outreach Name":"Daily Distribution"
        },
        {
            "Visit ID":"1", "Outreach on":"2026-01-10 10:00:00", "Guest ID":"A",
            "Tracking Method":"TEFAP Food", "Total Individuals":"3", "0 to 17":"1",
            "18 to 61":"2", "62+":"0", "Homeless":"No", "Outreach Name":"Daily Distribution"
        },
    ])
    visits = collapse_to_visits(raw)
    assert len(visits) == 1


def test_special_overlay_rules():
    config = {
        "special_overlays": {
            "toy": {"month":4, "drive_position":1},
            "backpack": {"month":7, "drive_position":2},
            "thanksgiving": {"month":11, "drive_position":2},
            "christmas": {"month":12, "drive_position":2},
        }
    }
    assert special_flags(4,1,config)["toy"] == 1
    assert special_flags(4,2,config)["toy"] == 0
    assert special_flags(7,2,config)["backpack"] == 1
