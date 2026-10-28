from pathlib import Path

from webapp import app


def test_home_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Recipient Forecast" in response.data


def test_forecast_with_synthetic_data():
    client = app.test_client()
    sample = Path("sample_data/synthetic_guest_visits.csv")
    with sample.open("rb") as handle:
        response = client.post(
            "/forecast",
            data={
                "data_file": (handle, "synthetic_guest_visits.csv"),
                "drive_date": "2026-10-10",
                "drive_position": "1",
            },
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b"Predicted returning households" in response.data
    assert b"October 10, 2026" in response.data
