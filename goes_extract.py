"""goes_extract.py — GOES-East ABI clear-sky mask at every photometer under the GOES-East disk, at night.

Product: ABI-L2-ACMF (full disk, 10-min scans), variable BCM (binary cloud mask: 0 clear, 1 cloudy) and
DQF. Only scans from START on are used: the Enterprise cloud-mask algorithm took over on 2021-11-29 19 UTC,
and the earlier algorithm differs (Fernandes Medeiros 2026, ESS Open Archive preprint, sec. 4c). Bucket by date: noaa-goes16 until the
GOES-19 handover, noaa-goes19 after (the nearest file in either is used when one has no scan).

For each target time (every STEP_MIN minutes) the scan starting nearest to it is downloaded once, every
photometer under the disk with the Sun below SUN_MAX deg is sampled from it, and the file is deleted. Per
photometer and scan: the nearest pixel, and the mean BCM over 3x3, 5x5 and 7x7 pixel boxes (valid pixels
only), with the number of valid pixels and the scan's view zenith angle at the site. Rows go to
data/goes/goes_<YYYY-MM>.csv.gz (one file per month; a month is rewritten only if missing, so runs resume).
The fixed-grid navigation is the ABI Product User Guide's, as in the code of Fernandes Medeiros (2026).

Usage: python goes_extract.py [--months 2024-01 2024-02 ...] [--only stars645 ...] [--step 30] [--keep DIR]"""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import math
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import netCDF4
import numpy as np

from audit_ir import sun_altitude

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT = ROOT / "data" / "goes"
START = dt.datetime(2021, 11, 29, 19)
END = dt.datetime(2026, 10, 1)
HANDOVER = dt.datetime(2025, 4, 7)          # GOES-19 operational as GOES-East (NOAA)
STEP_MIN = 30
SUN_MAX = -12.0
SUB_LON = -75.2
MAX_VZA = 75.0
FIELDS = ["time", "scan_start", "satellite", "id", "vza", "bcm_center", "bcm_3", "n_3", "bcm_5", "n_5",
          "bcm_7", "n_7"]


def s3_keys(bucket: str, prefix: str) -> list[str]:
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    for attempt in range(6):                    # S3 answers an occasional 500/503; back off and retry
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return re.findall(r"<Key>([^<]+)</Key>", r.read().decode())
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 5 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise
            time.sleep(10 * 2 ** attempt)


def nearest_scan(t: dt.datetime) -> tuple[str, str, dt.datetime] | None:
    first, second = ("noaa-goes16", "noaa-goes19") if t < HANDOVER else ("noaa-goes19", "noaa-goes16")
    for bucket in (first, second):
        best = None
        for hour in {t.hour, (t - dt.timedelta(minutes=10)).hour}:
            day = t if hour == t.hour else t - dt.timedelta(minutes=10)
            prefix = f"ABI-L2-ACMF/{day.year}/{day.timetuple().tm_yday:03d}/{hour:02d}/"
            for k in s3_keys(bucket, prefix):
                m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", k)
                if not m:
                    continue
                y, d, hh, mm, ss = map(int, m.groups())
                s = dt.datetime(y, 1, 1, hh, mm, ss) + dt.timedelta(days=d - 1)
                if best is None or abs((s - t).total_seconds()) < abs((best[2] - t).total_seconds()):
                    best = (bucket, k, s)
        if best and abs((best[2] - t).total_seconds()) <= 600:
            return best
    return None


def geos_xy(lat: float, lon: float, p) -> tuple[float, float, float] | None:
    """Fixed-grid scan angles (rad) and view zenith angle (deg); None if beyond the limb."""
    H = p.perspective_point_height + p.semi_major_axis
    req, rpol = p.semi_major_axis, p.semi_minor_axis
    lon0 = math.radians(p.longitude_of_projection_origin)
    e2 = 1 - (rpol / req) ** 2
    la, lo = math.radians(lat), math.radians(lon)
    phic = math.atan((rpol ** 2 / req ** 2) * math.tan(la))
    rc = rpol / math.sqrt(1 - e2 * math.cos(phic) ** 2)
    sx = H - rc * math.cos(phic) * math.cos(lo - lon0)
    sy = -rc * math.cos(phic) * math.sin(lo - lon0)
    sz = rc * math.sin(phic)
    if H * (H - sx) < sy ** 2 + (req / rpol) ** 2 * sz ** 2:
        return None
    # view zenith angle from Earth-centred coordinates: satellite on the equator at lon0, site on the ellipsoid
    N = req / math.sqrt(1 - e2 * math.sin(la) ** 2)
    site = (N * math.cos(la) * math.cos(lo), N * math.cos(la) * math.sin(lo), N * (1 - e2) * math.sin(la))
    sat = (H * math.cos(lon0), H * math.sin(lon0), 0.0)
    v = [sat[k] - site[k] for k in range(3)]
    n = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    cosv = sum(v[k] * n[k] for k in range(3)) / math.sqrt(sum(c * c for c in v))
    return (math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz)), math.atan(sz / sx),
            math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))


def sample(path: Path, units: list[dict], t: dt.datetime, scan: dt.datetime, sat: str) -> list[list]:
    rows = []
    with netCDF4.Dataset(path) as nc:
        p = nc.variables["goes_imager_projection"]
        x, y = nc.variables["x"][:], nc.variables["y"][:]
        bcm, dqf = nc.variables["BCM"], nc.variables["DQF"]
        for u in units:
            g = geos_xy(u["lat"], u["lon"], p)
            if g is None or g[2] > MAX_VZA:
                continue
            i, j = int(np.abs(y - g[1]).argmin()), int(np.abs(x - g[0]).argmin())
            win = np.ma.filled(bcm[max(0, i - 3):i + 4, max(0, j - 3):j + 4].astype(float), np.nan)
            q = np.ma.filled(dqf[max(0, i - 3):i + 4, max(0, j - 3):j + 4].astype(float), np.nan)
            win[(q != 0) | (win > 1)] = np.nan               # good-quality pixels only
            c = win.shape[0] // 2, win.shape[1] // 2
            row = [t.isoformat(timespec="minutes"), scan.isoformat(timespec="seconds"), sat, u["id"],
                   round(g[2], 1), "" if np.isnan(win[c]) else int(win[c])]
            for h in (1, 2, 3):
                box = win[c[0] - h:c[0] + h + 1, c[1] - h:c[1] + h + 1]
                n = int(np.isfinite(box).sum())
                row += [round(float(np.nanmean(box)), 3) if n else "", n]
            rows.append(row)
    return rows


def month_times(month: str, step: int = STEP_MIN) -> list[dt.datetime]:
    y, m = map(int, month.split("-"))
    t = max(dt.datetime(y, m, 1), START)
    end = dt.datetime(y + (m == 12), m % 12 + 1, 1)
    out = []
    while t < min(end, END):
        out.append(t)
        t += dt.timedelta(minutes=step)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--step", type=int, default=STEP_MIN)
    ap.add_argument("--keep", type=Path, help="move each downloaded scan here instead of deleting it")
    a = ap.parse_args()
    units = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(SELECTED.open())
             if -165 < float(r["lon"]) < -10 and (not a.only or r["id"] in a.only)]
    months = a.months or [f"{y}-{m:02d}" for y in range(2021, 2027) for m in range(1, 13)
                          if dt.datetime(y, m, 1) >= START.replace(day=1, hour=0) and dt.datetime(y, m, 1) < END]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="goes_"))
    for month in months:
        out = OUT / f"goes_{month}.csv.gz"
        if out.exists():
            continue
        rows = []
        for t in month_times(month, a.step):
            dark = [u for u in units if sun_altitude(t, u["lat"], u["lon"]) < SUN_MAX]
            if not dark:
                continue
            found = nearest_scan(t)
            if not found:
                print(f"{t}: no scan", flush=True)
                continue
            bucket, key, scan = found
            f = tmp / "scan.nc"
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "4", "--max-time", "300", "-o", str(f),
                                f"https://{bucket}.s3.amazonaws.com/{key}"], capture_output=True, text=True)
            if r.returncode:
                print(f"{t}: ERROR {r.stderr.strip()[:200]}", flush=True)
                continue
            rows += sample(f, dark, t, scan, bucket.replace("noaa-", ""))
            if a.keep:
                dest = a.keep / "goes" / month / Path(key).name
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
        print(f"{month}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
