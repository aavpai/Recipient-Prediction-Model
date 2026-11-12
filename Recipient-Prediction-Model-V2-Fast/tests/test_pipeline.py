from datetime import date
import pandas as pd

from seva_model.config import special_flags
from seva_model.io import collapse_to_visits, load_raw_export


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



def test_csv_loader_ignores_unused_export_columns(tmp_path):
    source = pd.DataFrame([{
        "Visit ID": "100",
        "Outreach on": "2026-03-14 10:00:00",
        "Guest ID": "G-100",
        "Tracking Method": "Pantry Food",
        "Total Individuals": "4",
        "0 to 17": "2",
        "18 to 61": "2",
        "62+": "0",
        "Homeless": "No",
        "Outreach Name": "Regular Distribution",
        "Unused Administrative Notes": "not needed by the model",
    }])
    csv_path = tmp_path / "export.csv"
    source.to_csv(csv_path, index=False)

    raw = load_raw_export(csv_path)
    assert "Unused Administrative Notes" not in raw.columns
    visits = collapse_to_visits(raw)
    assert len(visits) == 1
    assert visits.iloc[0]["guest_id"] == "G-100"
    assert visits.iloc[0]["total_individuals"] == 4
