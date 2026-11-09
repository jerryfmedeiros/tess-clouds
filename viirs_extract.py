"""viirs_extract.py — VIIRS cloud mask (NASA CLDMSK_L2_VIIRS, MVCM) at every photometer, night overpasses.

For each platform (SNPP, NOAA20, NOAA21) and month, the night granules (CMR day_night_flag=night) are listed
once from CMR with their footprint polygons; a granule is downloaded only if its footprint contains at least
one photometer with the Sun below SUN_MAX deg at the granule's start, every such photometer is sampled from
it, and the file is deleted. One download serves all the photometers under a swath, which matters where the
network is dense (Spain). Per photometer and granule: distance to the nearest swath pixel (rows kept only
within COVERAGE_MAX_KM, as in Fernandes Medeiros 2026), the Integer_Cloud_Mask code there (0 confident
cloudy, 1 probably cloudy, 2 probably clear, 3 confident clear), and the cloudy fraction (codes 0-1) over
3x3, 5x5 and 7x7 pixel boxes with their valid-pixel counts. Rows go to data/viirs/viirs_<platform>_<YYYY-MM>.csv.gz;
a finished month is not redone. Needs an Earthdata token in EARTHDATA_TOKEN or ~/.edl_token.

Usage: python viirs_extract.py [--months 2024-01 ...] [--platforms SNPP NOAA20 NOAA21] [--only stars645 ...] [--keep DIR]"""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import netCDF4
import numpy as np

from audit_ir import sun_altitude

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT = ROOT / "data" / "viirs"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = {"SNPP": "CLDMSK_L2_VIIRS_SNPP", "NOAA20": "CLDMSK_L2_VIIRS_NOAA20", "NOAA21": "CLDMSK_L2_VIIRS_NOAA21"}
START, END = dt.date(2019, 1, 1), dt.date(2026, 10, 1)
SUN_MAX = -12.0
COVERAGE_MAX_KM = 2.0
FIELDS = ["granule_start", "platform", "id", "dist_km", "code_center", "cloudy_3", "n_3", "cloudy_5", "n_5",
          "cloudy_7", "n_7", "granule"]


def token() -> str:
    t = os.environ.get("EARTHDATA_TOKEN", "").strip()
    return t or (Path.home() / ".edl_token").read_text().strip()


def cmr_month(platform: str, month: str) -> list[dict]:
    """Night granules of the month: start time, data URL, footprint polygons [(lat, lon), ...]."""
    y, m = map(int, month.split("-"))
    t0 = dt.datetime(y, m, 1)
    t1 = dt.datetime(y + (m == 12), m % 12 + 1, 1)
    q = {"short_name": SHORT_NAME[platform], "day_night_flag": "night", "page_size": 2000,
         "temporal": f"{t0:%Y-%m-%dT%H:%M:%SZ},{t1:%Y-%m-%dT%H:%M:%SZ}", "sort_key": "start_date"}
    out, after = [], None
    while True:
        req = urllib.request.Request(CMR + "?" + urllib.parse.urlencode(q))
        if after:
            req.add_header("CMR-Search-After", after)
        with urllib.request.urlopen(req, timeout=120) as r:
            after = r.headers.get("CMR-Search-After")
            entries = json.loads(r.read())["feed"]["entry"]
        for e in entries:
            url = next((l["href"] for l in e.get("links", []) if l.get("href", "").endswith(".nc")), None)
            polys = []
            for p in e.get("polygons", []):
                v = list(map(float, p[0].split()))
                polys.append(list(zip(v[0::2], v[1::2])))
            if url and polys:
                out.append({"start": dt.datetime.strptime(e["time_start"][:19], "%Y-%m-%dT%H:%M:%S"),
                            "url": url, "polys": polys})
        if not entries or not after:
            return out


def inside(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    """Point in a lat/lon polygon (ray casting; longitudes unwrapped about the point)."""
    pts = [(la, lo + 360 * round((lon - lo) / 360)) for la, lo in poly]
    hit = False
    for (a1, o1), (a2, o2) in zip(pts, pts[1:] + pts[:1]):
        if (o1 > lon) != (o2 > lon) and lat < a1 + (lon - o1) * (a2 - a1) / (o2 - o1):
            hit = not hit
    return hit


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = np.sin(p(lat - lat0) / 2) ** 2 + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def sample(path: Path, units: list[dict], g: dict, platform: str) -> list[list]:
    rows = []
    with netCDF4.Dataset(path) as nc:
        geo, phys = nc.groups["geolocation_data"], nc.groups["geophysical_data"]
        glat = np.asarray(geo.variables["latitude"][:], dtype=float)
        glon = np.asarray(geo.variables["longitude"][:], dtype=float)
        cm = phys.variables["Integer_Cloud_Mask"]
        ny, nx = glat.shape
        coarse_lat, coarse_lon = glat[::16, ::16], glon[::16, ::16]
        for u in units:
            # nearest pixel: on a 1-in-16 grid first, then the full grid in a window around it
            dc = km(u["lat"], u["lon"], coarse_lat, coarse_lon)
            cy0, cx0 = np.unravel_index(int(np.nanargmin(dc)), dc.shape)
            y0, x0 = max(0, cy0 * 16 - 32), max(0, cx0 * 16 - 32)
            d = km(u["lat"], u["lon"], glat[y0:y0 + 64, x0:x0 + 64], glon[y0:y0 + 64, x0:x0 + 64])
            iy, ix = np.unravel_index(int(np.nanargmin(d)), d.shape)
            d = d[iy, ix]
            iy, ix = iy + y0, ix + x0
            if d > COVERAGE_MAX_KM:
                continue
            win = np.ma.asarray(cm[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]).astype(float).filled(np.nan)
            cy, cx = iy - max(0, iy - 3), ix - max(0, ix - 3)
            c = win[cy, cx]
            row = [g["start"].isoformat(), platform, u["id"], round(float(d), 2),
                   "" if np.isnan(c) else int(c)]
            for h in (1, 2, 3):
                box = win[max(0, cy - h):cy + h + 1, max(0, cx - h):cx + h + 1]
                n = int(np.isfinite(box).sum())
                row += [round(float(np.nansum(box <= 1) / n), 3) if n else "", n]
            rows.append(row + [g["url"].rsplit("/", 1)[-1]])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--platforms", nargs="*", default=list(SHORT_NAME))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep", type=Path, help="move each downloaded granule here instead of deleting it")
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the monthly files (default data/viirs)")
    a = ap.parse_args()
    out_dir = a.out
    units = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(SELECTED.open())
             if not a.only or r["id"] in a.only]
    months = a.months or [f"{y}-{m:02d}" for y in range(START.year, END.year + 1) for m in range(1, 13)
                          if START <= dt.date(y, m, 1) < END]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="viirs_"))
    tok = token()
    for platform in a.platforms:
        for month in months:
            out = out_dir / f"viirs_{platform}_{month}.csv.gz"
            if out.exists():
                continue
            rows, n_dl = [], 0
            for g in cmr_month(platform, month):
                dark = [u for u in units if sun_altitude(g["start"], u["lat"], u["lon"]) < SUN_MAX
                        and any(inside(u["lat"], u["lon"], p) for p in g["polys"])]
                if not dark:
                    continue
                kept = a.keep / "viirs" / platform / month / g["url"].rsplit("/", 1)[-1] if a.keep else None
                if kept is not None and kept.exists() and kept.stat().st_size > 0:
                    rows += sample(kept, dark, g, platform)          # already downloaded by an earlier run
                    continue
                f = tmp / "granule.nc"
                r = subprocess.run(["curl", "-sS", "--fail", "-L", "--retry", "4", "--max-time", "300",
                                    "-H", f"Authorization: Bearer {tok}", "-o", str(f), g["url"]],
                                   capture_output=True, text=True)
                if r.returncode:
                    print(f"{g['start']}: ERROR {r.stderr.strip()[:200]}", flush=True)
                    continue
                n_dl += 1
                rows += sample(f, dark, g, platform)
                if a.keep:
                    dest = a.keep / "viirs" / platform / month / g["url"].rsplit("/", 1)[-1]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(f, dest)
                else:
                    f.unlink()
            part = out.with_suffix(".part")
            with gzip.open(part, "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(FIELDS)
                w.writerows(rows)
            part.replace(out)
            print(f"{platform} {month}: {n_dl} granules, {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
