"""audit_country.py — the position of every selected photometer against the country named in its archive header.

The position audit (audit_positions.py) compares each month's header position with the position used, so a header that
is mistyped in every month passes it: stars397 (Pfälzerwald, Germany) was filed at 9.34 N instead of 49.34 N in all 25
of its months, which places it in Nigeria; this check also finds stars403 and stars411
(La Palma) at 1.8 W instead of 17.96 W, the decimal-point slip already corrected at stars425. This check reads the country, the last
element of the header's location string, and tests the position against generous bounding boxes of that country
(several boxes where a country has distant territories: the Canaries, Réunion, the Azores and Madeira, Hawaii and
Alaska). A position outside every box of its country is listed; a country with no box is listed as unchecked.

Reads data/tess_archive/selected.csv (positions after data/station_fixes.csv). Writes results/audit_country.csv
(id, country, lat, lon, status) and prints the positions outside their country.

Usage: python audit_country.py"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT = ROOT / "results" / "audit_country.csv"
# country -> boxes (lat_min, lat_max, lon_min, lon_max), generous
BOXES = {
    "Spain": [(35.5, 44.0, -9.5, 4.5), (27.4, 29.6, -18.5, -13.3)],
    "Portugal": [(36.8, 42.3, -9.6, -6.1), (36.8, 39.9, -31.5, -24.9), (32.3, 33.2, -17.4, -16.2)],
    "France": [(41.2, 51.2, -5.5, 9.8), (-21.5, -20.8, 55.1, 55.9), (41.2, 43.1, 8.4, 9.7)],
    "Germany": [(47.2, 55.2, 5.8, 15.1)], "Switzerland": [(45.7, 47.9, 5.9, 10.6)], "Austria": [(46.3, 49.1, 9.4, 17.2)],
    "Italy": [(35.4, 47.2, 6.5, 18.6)], "Italia": [(35.4, 47.2, 6.5, 18.6)], "Belgium": [(49.4, 51.6, 2.5, 6.5)],
    "Netherlands": [(50.7, 53.7, 3.2, 7.3), (11.9, 12.5, -69.2, -68.1), (17.4, 17.7, -63.3, -62.9)],
    "United Kingdom": [(49.8, 61.0, -8.7, 2.0)], "Ireland": [(51.3, 55.5, -10.7, -5.9)], "Norway": [(57.9, 71.3, 4.4, 31.2), (76.0, 81.0, 10.0, 35.0)],
    "Sweden": [(55.3, 69.1, 10.9, 24.2)], "Slovenia": [(45.4, 46.9, 13.3, 16.6)], "Bulgaria": [(41.2, 44.3, 22.3, 28.7)],
    "Malta": [(35.7, 36.1, 14.1, 14.6)], "Andorra": [(42.4, 42.7, 1.4, 1.8)], "Russia": [(41.1, 82.0, 19.5, 180.0)],
    "Canada": [(41.6, 83.2, -141.1, -52.5)], "United States": [(24.4, 49.5, -125.0, -66.9), (51.0, 71.5, -179.9, -129.9), (18.8, 22.3, -160.3, -154.7)],
    "USA": [(24.4, 49.5, -125.0, -66.9), (51.0, 71.5, -179.9, -129.9), (18.8, 22.3, -160.3, -154.7)],
    "Mexico": [(14.5, 32.8, -118.5, -86.6)], "Panama": [(7.1, 9.7, -83.1, -77.1)], "Argentina": [(-55.2, -21.7, -73.7, -53.5)],
    "Chile": [(-56.0, -17.4, -75.8, -66.3), (-27.3, -27.0, -109.5, -109.2)], "Australia": [(-44.0, -10.0, 112.8, 154.0)],
    "New Zealand": [(-47.4, -34.3, 166.3, 178.7)], "Israel": [(29.4, 33.4, 34.2, 35.9)], "India": [(6.6, 35.6, 68.0, 97.5)],
    "China": [(18.1, 53.6, 73.5, 135.1)], "Vietnam": [(8.4, 23.4, 102.1, 109.6)], "Namibia": [(-29.0, -16.9, 11.6, 25.3)],
    "South Africa": [(-35.0, -22.1, 16.4, 33.0)],
}


def main():
    rows, outside, unchecked = [], [], []
    for r in csv.DictReader(SELECTED.open()):
        country = r["location"].rsplit("/", 1)[-1].strip()
        lat, lon = float(r["lat"]), float(r["lon"])
        boxes = BOXES.get(country)
        if boxes is None:
            status = "unchecked"
            unchecked.append((r["id"], country))
        elif any(a <= lat <= b and c <= lon <= d for a, b, c, d in boxes):
            status = "inside"
        else:
            status = "outside"
            outside.append((r["id"], country, lat, lon))
        rows.append([r["id"], country, lat, lon, status])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "country", "lat", "lon", "status"])
        w.writerows(rows)
    print(f"{len(rows)} photometers; outside their country: {len(outside)}; unchecked (no box for the country): {len(unchecked)}")
    for x in outside:
        print("  outside:", *x)
    for x in unchecked:
        print("  unchecked:", *x)


if __name__ == "__main__":
    main()
