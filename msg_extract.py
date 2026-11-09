"""msg_extract.py — EUMETSAT Meteosat (MSG, 0 deg service) cloud mask at every photometer under the disk, at night.

Product: EO:EUM:DAT:MSG:CLM ("Cloud Mask - MSG - 0 degree", 15-min repeat cycles, GRIB2 on a 3712 x 3712
space-view grid, about 3 km at the sub-satellite point; values 0 clear over water, 1 clear over land,
2 cloud, 3 not processed). Licence: NoConditions. Access through EUMETSAT's eumdac client with the
credentials in ~/.eumdac/credentials ("key,secret").

For each target time (every STEP_MIN minutes, the same grid as goes_extract.py) the product whose sensing
end is nearest to it is downloaded once, every photometer under the disk with the Sun below SUN_MAX deg and a
view zenith angle below MAX_VZA is sampled from it, and the file is deleted. Pixel positions come from the standard
geostationary navigation (scan angles from latitude and longitude on the ellipsoid, as in goes_extract.py)
with the grid constants in each GRIB (Nr, dx, dy, Xp, Yp; columns run east to west, rows south to north). The
GRIB's own latitude and longitude arrays are not used: eccodes returns (0, 0) off the disk and in reversed
order to the values. The navigation was checked on 2024-01-15 05:00: clear pixels over the Sahara and Arabia
are class 1 (land) and over the South Atlantic class 0 (water), never the other way. Per photometer and product: the nearest pixel's class and the cloudy fraction over
3x3, 5x5 and 7x7 pixel boxes (class 3 excluded) with their valid-pixel counts. Rows go to
data/msg/msg_<YYYY-MM>.csv.gz; a finished month is not redone.

Usage: python msg_extract.py [--months 2024-01 ...] [--only stars51 ...] [--step 30] [--keep DIR] [--threads 4]"""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import io
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import zipfile
from pathlib import Path

import eccodes
import eumdac
import numpy as np

from audit_ir import sun_altitude

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT = ROOT / "data" / "msg"
COLLECTION = "EO:EUM:DAT:MSG:CLM"
START, END = dt.datetime(2019, 1, 1), dt.datetime(2026, 10, 1)
STEP_MIN = 30
SUN_MAX = -12.0
MAX_VZA = 75.0
FIELDS = ["time", "sensing_end", "product", "id", "vza", "class_center", "cloudy_3", "n_3", "cloudy_5", "n_5",
          "cloudy_7", "n_7"]


def datastore() -> eumdac.DataStore:
    key, secret = (Path.home() / ".eumdac" / "credentials").read_text().strip().split(",")
    return eumdac.DataStore(eumdac.AccessToken((key, secret)))


def vza(lat: float, lon: float, sublon: float) -> float:
    """View zenith angle (deg) of a geostationary satellite over the equator at sublon (spherical Earth)."""
    la, dlo = math.radians(lat), math.radians(lon - sublon)
    cosg = math.cos(la) * math.cos(dlo)                       # cosine of the central angle to the sub-point
    r, h = 6371.0, 42164.0
    d = math.sqrt(r * r + h * h - 2 * r * h * cosg)
    return math.degrees(math.asin(min(1.0, h * math.sqrt(max(0.0, 1 - cosg * cosg)) / d)))


def read_grib(zbytes: bytes) -> tuple[np.ndarray, dict]:
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    name = next(n for n in z.namelist() if n.endswith(".grb"))
    with tempfile.NamedTemporaryFile(suffix=".grb") as t:
        t.write(z.read(name))
        t.flush()
        with open(t.name, "rb") as f:
            h = eccodes.codes_grib_new_from_file(f)
            try:
                g = {k: eccodes.codes_get(h, k) for k in ("Nx", "Ny", "dx", "dy", "Xp", "Yp", "Nr",
                                                          "longitudeOfSubSatellitePointInDegrees")}
                return eccodes.codes_get_values(h).reshape(g["Ny"], g["Nx"]), g
            finally:
                eccodes.codes_release(h)


def pixel(lat: float, lon: float, g: dict) -> tuple[int, int] | None:
    """Row and column of the grid cell containing (lat, lon); None beyond the limb."""
    req, rpol = 6378.169, 6356.5838
    nr = g["Nr"] / 1e6
    H = nr * req
    e2 = 1 - (rpol / req) ** 2
    la, dl = math.radians(lat), math.radians(lon - g["longitudeOfSubSatellitePointInDegrees"])
    phic = math.atan((rpol ** 2 / req ** 2) * math.tan(la))
    rc = rpol / math.sqrt(1 - e2 * math.cos(phic) ** 2)
    sx = H - rc * math.cos(phic) * math.cos(dl)
    sy = -rc * math.cos(phic) * math.sin(dl)
    sz = rc * math.sin(phic)
    if H * (H - sx) < sy * sy + (req / rpol) ** 2 * sz * sz:
        return None
    x = math.degrees(math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz)))
    y = math.degrees(math.atan(sz / sx))
    ang = math.degrees(2 * math.asin(1 / nr))           # apparent Earth diameter, deg
    return round(g["Yp"] / 1000 + y / (ang / g["dy"])), round(g["Xp"] / 1000 - x / (ang / g["dx"]))


def sample(v: np.ndarray, g: dict, units: list[dict], t: dt.datetime, p) -> list[list]:
    rows = []
    sub = g["longitudeOfSubSatellitePointInDegrees"]
    for u in units:
        ij = pixel(u["lat"], u["lon"], g)
        if ij is None:
            continue
        i, j = ij
        win = v[max(0, i - 3):i + 4, max(0, j - 3):j + 4].astype(float)
        win[(win > 2) | (win < 0)] = np.nan            # class 3: not processed
        ci, cj = i - max(0, i - 3), j - max(0, j - 3)
        c = win[ci, cj]
        row = [t.isoformat(timespec="minutes"), p.sensing_end.strftime("%Y-%m-%dT%H:%M:%S"), str(p), u["id"],
               round(vza(u["lat"], u["lon"], sub), 1), "" if np.isnan(c) else int(c)]
        for h in (1, 2, 3):
            box = win[max(0, ci - h):ci + h + 1, max(0, cj - h):cj + h + 1]
            n = int(np.isfinite(box).sum())
            row += [round(float(np.nansum(box == 2) / n), 3) if n else "", n]
        rows.append(row)
    return rows


def with_timeout():
    """The Data Store client sends requests with no timeout, so one stalled download can hang a run for good;
    give every request a connect and read timeout, which turns a stall into an error the retry loop handles."""
    import requests
    send = requests.Session.request

    def request(self, *args, **kw):
        if kw.get("timeout") is None:
            kw["timeout"] = (30, 180)
        return send(self, *args, **kw)
    requests.Session.request = request


def main():
    with_timeout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--step", type=int, default=STEP_MIN)
    ap.add_argument("--keep", type=Path, help="also save each downloaded product here")
    ap.add_argument("--threads", type=int, default=4, help="concurrent downloads")
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the monthly files (default data/msg)")
    a = ap.parse_args()
    out_dir = a.out
    units = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(SELECTED.open())
             if vza(float(r["lat"]), float(r["lon"]), 0.0) < MAX_VZA and (not a.only or r["id"] in a.only)]
    months = a.months or [f"{y}-{m:02d}" for y in range(START.year, END.year + 1) for m in range(1, 13)
                          if START <= dt.datetime(y, m, 1) < END]
    out_dir.mkdir(parents=True, exist_ok=True)
    coll = datastore().get_collection(COLLECTION)
    for month in months:
        out = out_dir / f"msg_{month}.csv.gz"
        if out.exists():
            continue
        y, m = map(int, month.split("-"))
        t0, t1 = dt.datetime(y, m, 1), dt.datetime(y + (m == 12), m % 12 + 1, 1)
        prods = {p.sensing_end.replace(second=0, microsecond=0, tzinfo=None): p
                 for p in coll.search(dtstart=t0, dtend=t1)}
        jobs = []
        t = t0
        while t < min(t1, END):
            dark = [u for u in units if sun_altitude(t, u["lat"], u["lon"]) < SUN_MAX]
            near = min(prods, key=lambda s: abs((s - t).total_seconds()), default=None) if dark else None
            if near is not None and abs((near - t).total_seconds()) <= 600:
                jobs.append((t, prods[near], dark))
            t += dt.timedelta(minutes=a.step)

        def one(job):
            t, p, dark = job
            dest = a.keep / "msg" / month / f"{p}.zip" if a.keep else None
            if dest is not None and dest.exists() and dest.stat().st_size > 0:
                v, g = read_grib(dest.read_bytes())               # already downloaded by an earlier run
                return sample(v, g, dark, t, p)
            for attempt in range(4):
                try:
                    with p.open() as fh:
                        zb = fh.read()
                    break
                except Exception as e:                     # transient Data Store errors: retry
                    err = e
                    time.sleep(20 * (attempt + 1))
            else:
                print(f"{t}: ERROR {err}", flush=True)
                return []
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_suffix(".part")
                part.write_bytes(zb)
                part.replace(dest)
            v, g = read_grib(zb)
            return sample(v, g, dark, t, p)

        with ThreadPoolExecutor(a.threads) as ex:
            results = list(ex.map(one, jobs))
        rows = [r for res in results for r in res]
        n = sum(1 for res in results if res)
        part = out.with_suffix(".part")
        with gzip.open(part, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            w.writerows(rows)
        part.replace(out)
        print(f"{month}: {n} products, {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
