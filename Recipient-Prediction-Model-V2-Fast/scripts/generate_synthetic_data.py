"""Generate a privacy-safe synthetic Seva-style export for examples and tests."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import random
import pandas as pd

random.seed(42)

OUT = Path("sample_data/synthetic_guest_visits.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def saturdays(year, month):
    d = date(year, month, 1)
    while d.weekday() != 5:
        d += timedelta(days=1)
    vals = []
    while d.month == month:
        vals.append(d)
        d += timedelta(days=7)
    return vals

recipients = []
for gid in range(1001, 1181):
    recipients.append({
        "gid": str(gid),
        "base": random.betavariate(2.2, 4.5),
        "d2_pref": random.betavariate(2, 2),
        "kids": random.choices([0,1,2,3], weights=[40,28,22,10])[0],
        "seniors": random.choices([0,1,2], weights=[72,22,6])[0],
    })

rows = []
visit_id = 1
for year in [2024, 2025, 2026]:
    max_month = 9 if year == 2026 else 12
    for month in range(1, max_month + 1):
        sats = saturdays(year, month)
        if len(sats) < 2:
            continue
        # Use second and fourth Saturday where possible.
        core = [sats[1] if len(sats) > 2 else sats[0], sats[3] if len(sats) > 3 else sats[-1]]
        for pos, d in enumerate(core, start=1):
            special = (
                (month == 4 and pos == 1)
                or (month == 7 and pos == 2)
                or (month == 11 and pos == 2)
                or (month == 12 and pos == 2)
            )
            for r in recipients:
                p = r["base"] * 0.55
                if pos == 2:
                    p *= 0.75 + 1.0 * r["d2_pref"]
                if special:
                    p += 0.10
                if month == 4 and pos == 1 and r["kids"]:
                    p += 0.12
                if month == 7 and pos == 2 and r["kids"]:
                    p += 0.15
                p = min(0.92, p)
                if random.random() < p:
                    total = 1 + r["kids"] + r["seniors"] + random.choice([0,1,1,2])
                    base_row = {
                        "Visit ID": str(visit_id),
                        "Visit on": f"{d.isoformat()} 10:00:00",
                        "Visit Location": "The Seva Collective",
                        "Guest Number": "0",
                        "Tracking Method": "Pantry Food",
                        "Tracking Result": str(total),
                        "Visit Status": "Active",
                        "Volunteer": "Synthetic",
                        "Volunteer Overide Allow": "",
                        "Volunteer Overide Ident": "",
                        "Volunteer Overide Cancel": "",
                        "Volunteer Overide Conditions": "",
                        "Outreach ID": "1",
                        "Outreach Name": "Daily Distribution",
                        "Outreach on": f"{d.isoformat()} 10:00:00",
                        "Guest ID": r["gid"],
                        "First Name": "Synthetic",
                        "Middle Name": "",
                        "Last Name": f"Guest{r['gid']}",
                        "Date of Birth": "",
                        "Address": "",
                        "Apartment/Suite": "",
                        "City/Town": "",
                        "State/Province": "CA",
                        "Zip/Postal Code": "00000",
                        "County": "",
                        "Homeless": "No",
                        "Total Individuals": total,
                        "0 to 17": r["kids"],
                        "18 to 61": max(1, total-r["kids"]-r["seniors"]),
                        "62+": r["seniors"],
                    }
                    rows.append(base_row)
                    if pos == 2:
                        dup = dict(base_row)
                        dup["Tracking Method"] = "TEFAP Food"
                        rows.append(dup)
                    visit_id += 1

# A few small Tuesday supplemental distributions; these update recency but are not core targets.
for d in [date(2025,3,18), date(2025,5,13), date(2026,3,17), date(2026,5,12)]:
    for r in random.sample(recipients, 20):
        total = 2 + r["kids"]
        rows.append({
            "Visit ID": str(visit_id), "Visit on": f"{d.isoformat()} 10:00:00",
            "Visit Location": "The Seva Collective", "Guest Number": "0",
            "Tracking Method": "Pantry Food", "Tracking Result": str(total),
            "Visit Status": "Active", "Volunteer": "Synthetic",
            "Volunteer Overide Allow": "", "Volunteer Overide Ident": "",
            "Volunteer Overide Cancel": "", "Volunteer Overide Conditions": "",
            "Outreach ID": "2", "Outreach Name": "Supplemental Distribution",
            "Outreach on": f"{d.isoformat()} 10:00:00", "Guest ID": r["gid"],
            "First Name": "Synthetic", "Middle Name": "", "Last Name": f"Guest{r['gid']}",
            "Date of Birth": "", "Address": "", "Apartment/Suite": "",
            "City/Town": "", "State/Province": "CA", "Zip/Postal Code": "00000",
            "County": "", "Homeless": "No", "Total Individuals": total,
            "0 to 17": r["kids"], "18 to 61": max(1, total-r["kids"]), "62+": "0",
        })
        visit_id += 1

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows):,} rows to {OUT}")
