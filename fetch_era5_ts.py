"""fetch_era5_ts.py — hourly ERA5-Land 2 m air temperature, 2 m dew point and snow depth at each photometer,
from the CDS point time-series dataset (reanalysis-era5-land-timeseries, 0.1 deg; the nearest grid point).

The TESS-W "ambient" temperature is the heated sensor die, not the air, so the clear-sky baseline needs the
air temperature and humidity from elsewhere; snow depth is kept for the snow screen. One request per
ERA5-Land cell (photometers whose positions round to the same 0.1 deg cell share it; the point dataset takes
one location per request, so this, with all variables and the whole period in each request, is the most the
requests can be grouped), 2019-01-01 to 2026-09-30. Where the nearest cell is water (all values missing) the nearest land
cell within 0.2 deg is used and recorded. Writes data/era5_ts/<id>.csv.gz (time UTC, t2m_C, d2m_C, sde_m) and
data/era5_ts/<id>.json (grid point and offset); keeps the raw download in data/era5_ts_raw/. Up to THREADS
requests at once (CDS accepts three); needs cdsapi, pandas and ~/.cdsapirc.

Usage: python3 fetch_era5_ts.py [--only stars645 ...]"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import pandas as pd

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
RAW = ROOT / "data" / "era5_ts_raw"
OUT = ROOT / "data" / "era5_ts"
DATES = "2019-01-01/2026-09-30"
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature", "snow_depth"]
THREADS = 3


def fetch(lat: float, lon: float, raw: Path) -> pd.DataFrame:
    if not raw.exists():
        cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-land-timeseries", {
            "variable": VARIABLES, "location": {"longitude": lon, "latitude": lat},
            "date": [DATES], "data_format": "csv"}, str(raw.with_suffix(".part")))
        raw.with_suffix(".part").replace(raw)
    b = raw.read_bytes()
    if b[:2] != b"PK":
        return pd.read_csv(io.StringIO(b.decode()), parse_dates=["valid_time"])
    z = zipfile.ZipFile(io.BytesIO(b))                  # one CSV per variable, or one with all of them
    frames = [pd.read_csv(io.StringIO(z.read(n).decode()), parse_dates=["valid_time"]) for n in z.namelist()]
    t = frames[0]
    for f in frames[1:]:
        t = t.merge(f[["valid_time"] + [c for c in f.columns if c not in t.columns]], on="valid_time")
    return t


def one(group: list[dict]) -> str:
    """Fetch one cell and write the series for every photometer in it."""
    ids = [r["id"] for r in group]
    todo = [i for i in ids if not (OUT / f"{i}.csv.gz").exists()]
    if not todo:
        return f"{','.join(ids)}: exists"
    lat, lon = round(float(group[0]["lat"]), 1), round(float(group[0]["lon"]), 1)
    tries = [(0.0, 0.0)] + [(r * dy, r * dx) for r in (0.1, 0.2)
                            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
    try:
        for k, (dy, dx) in enumerate(tries):
            raw = RAW / f"cell_{lat + dy:+.1f}_{lon + dx:+.1f}.zip"
            t = fetch(lat + dy, lon + dx, raw)
            if t["t2m"].isna().mean() <= 0.05:
                break
        else:
            return f"{','.join(ids)}: SKIPPED, no land cell within 0.2 deg"
        s = pd.DataFrame({"time": t["valid_time"].dt.strftime("%Y-%m-%dT%H:%M"),
                          "t2m_C": (t["t2m"] - 273.15).round(2), "d2m_C": (t["d2m"] - 273.15).round(2),
                          "sde_m": t["sde"].clip(lower=0).round(4)})
        g = (round(float(t["latitude"].iloc[0]), 3), round(float(t["longitude"].iloc[0]), 3))
        for i in todo:
            s.to_csv(OUT / f"{i}.csv.gz", index=False, compression="gzip")
            (OUT / f"{i}.json").write_text(json.dumps({"grid": list(g), "offset_deg": [dy, dx],
                                                       "shared_with": [j for j in ids if j != i]}))
        return (f"{','.join(ids)}: {len(s)} hours, grid {g[0]:.2f},{g[1]:.2f}"
                + (f" (land cell {dy:+.1f},{dx:+.1f} deg away)" if (dy, dx) != (0.0, 0.0) else ""))
    except Exception as e:
        return f"{','.join(ids)}: ERROR {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rows = list(csv.DictReader(SELECTED.open()))
    if a.only:
        rows = [r for r in rows if r["id"] in a.only]
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((round(float(r["lat"]), 1), round(float(r["lon"]), 1)), []).append(r)
    print(f"{len(rows)} photometers in {len(cells)} cells", flush=True)
    with ThreadPoolExecutor(THREADS) as ex:
        for msg in ex.map(one, list(cells.values())):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
