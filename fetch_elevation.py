"""fetch_elevation.py — ground elevation of every photometer from a digital elevation model, because 340 of the 406
archive headers give 0 m and the header rule of the freeze (mountain if header elevation >= 1500 m) therefore misses
mountain sites (stars393, Parador de Canadas del Teide, about 2150 m).

Source: OpenTopoData public API (https://www.opentopodata.org), dataset DATASET (ASTER GDEM 30 m, global), at
most 100 locations per request and one request a second. Only the published station coordinates are sent.

Writes data/elevation.csv (id, lat, lon, dem_m, header_m).

Usage: python fetch_elevation.py"""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import sat_score as ss

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "elevation.csv"
API = "https://api.opentopodata.org/v1/"
DATASET = "aster30m"
BATCH = 100


def main():
    rows = list(csv.DictReader(ss.SELECTED.open()))
    header = ss.elevations()
    out = []
    for i in range(0, len(rows), BATCH):
        part = rows[i:i + BATCH]
        locs = "|".join(f"{float(r['lat']):.6f},{float(r['lon']):.6f}" for r in part)
        req = urllib.request.Request(f"{API}{DATASET}?locations={locs}", headers={"User-Agent": "tess-clouds research script"})
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                res = json.load(fh)["results"]
        except urllib.error.HTTPError as e:
            raise SystemExit(f"batch {i}: {e.code} {e.read().decode(errors='replace')[:300]}")
        for r, e in zip(part, res):
            z = e.get("elevation")
            out.append([r["id"], r["lat"], r["lon"], "" if z is None else round(z, 1), header.get(r["id"], 0.0)])
        time.sleep(1.1)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon", "dem_m", "header_m"])
        w.writerows(out)
    print(len(out), "elevations;", sum(1 for r in out if r[3] != "" and r[3] >= 1500), ">= 1500 m")


if __name__ == "__main__":
    main()
