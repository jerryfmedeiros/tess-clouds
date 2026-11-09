"""arco_check.py — does ARCO-ERA5 (Google's Zarr copy of ERA5, public bucket gcp-public-data-arco-era5) give the same
pressure-level temperature and relative humidity as the CDS files of fetch_mountain_met.py --era5-sites?

ARCO stores temperature and specific humidity, not relative humidity, so relative humidity is computed as in the IFS
(ECMWF IFS documentation Cy41r2, Part IV, Sect. 7.2.1, Eq. 7.5, and Sect. 12.7, Eqs. 12.8-12.13): saturation over water above 0 C, over ice below
-23 C, and a quadratic blend between. Each ARCO chunk is one hour of the globe at all 37 levels, read over HTTPS and
decoded with numcodecs; the CDS box and levels are cut from it. A seeded sample of hours from one CDS file is compared
point by point. Needs the raw CDS file (not a verify target).

Usage: python arco_check.py RAW/era5pl/europe_2019-01_2019-04.nc [--hours 24]"""
from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.request
from pathlib import Path

import netCDF4
import numpy as np
from numcodecs import Blosc

from mountain_baseline import dew_from_rh

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EPOCH_HOURS = 0                # time[0] of the store is 1900-01-01 (hours since 1900-01-01; empty before 1940)
OUT = Path(__file__).parent / "results" / "arco_check.txt"


def get(path: str, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{ARCO}/{path}", timeout=300) as r:
                return r.read()
        except (OSError, http.client.HTTPException):          # dropped connections end as IncompleteRead
            if k == tries - 1:
                raise
            time.sleep(10 * 2 ** k)


def meta():
    m = json.loads(get(".zmetadata"))["metadata"]
    arr = {}
    for k in ("level", "latitude", "longitude", "time"):
        z = m[f"{k}/.zarray"]
        assert z["chunks"][0] >= z["shape"][0] or k == "time"
        if k != "time":
            arr[k] = np.frombuffer(Blosc().decode(get(f"{k}/0")), dtype=z["dtype"])
    return m, arr


def field(var: str, i: int, shape) -> np.ndarray:
    return np.frombuffer(Blosc().decode(get(f"{var}/{i}.0.0.0")), dtype="<f4").reshape(shape[1:])


def rh_ifs(t: np.ndarray, q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature (K), specific humidity (kg/kg) and pressure (Pa), IFS mixed phase."""
    t0, tice, eps = 273.16, 250.16, 287.0597 / 461.5250
    esw = 611.21 * np.exp(17.502 * (t - t0) / (t - 32.19))
    esi = 611.21 * np.exp(22.587 * (t - t0) / (t + 0.7))
    a = np.clip((t - tice) / (t0 - tice), 0, 1) ** 2
    es = a * esw + (1 - a) * esi
    e = p_pa * q / (eps + (1 - eps) * q)
    return 100 * e / es


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cds", type=Path)
    ap.add_argument("--hours", type=int, default=24)
    a = ap.parse_args()
    d = netCDF4.Dataset(a.cds)
    vt = d["valid_time"][:].astype("int64")
    lev, lat, lon = d["pressure_level"][:], d["latitude"][:], d["longitude"][:]
    m, ax = meta()
    shape = m["temperature/.zarray"]["shape"]
    li = [int(np.where(ax["level"] == L)[0][0]) for L in lev]
    yi = [int(np.where(np.isclose(ax["latitude"], y))[0][0]) for y in lat]
    xi = [int(np.where(np.isclose(ax["longitude"], x % 360))[0][0]) for x in lon]
    rng = np.random.default_rng(20261102)
    pick = np.sort(rng.choice(len(vt), size=min(a.hours, len(vt)), replace=False))
    dt, dr, dd = [], [], []
    for k in pick:
        i = int(vt[k] // 3600 + 613608 - EPOCH_HOURS)       # 613608 = 1970-01-01 in hours since 1900-01-01
        t = field("temperature", i, shape)[np.ix_(li, yi, xi)]
        q = field("specific_humidity", i, shape)[np.ix_(li, yi, xi)]
        r = rh_ifs(t, q, (np.asarray(lev, float) * 100)[:, None, None])
        dt.append((t - d["t"][k]).ravel())
        dr.append((r - d["r"][k]).ravel())
        dd.append((dew_from_rh(t - 273.15, r) - dew_from_rh(np.asarray(d["t"][k]) - 273.15, np.asarray(d["r"][k]))).ravel())
    dt, dr, dd = np.concatenate(dt), np.concatenate(dr), np.concatenate(dd)
    lines = [f"{a.cds.name}: {len(pick)} hours, {len(lev)} levels, {len(lat)}x{len(lon)} points, {dt.size} values",
             f"temperature ARCO-CDS (K): mean {dt.mean():.6f} max|d| {np.abs(dt).max():.6f}",
             f"relative humidity ARCO(IFS formula)-CDS (%): mean {dr.mean():.6f} SD {dr.std():.6f} "
             f"p99|d| {np.percentile(np.abs(dr), 99):.6f} max|d| {np.abs(dr).max():.6f}",
             f"dew point as mountain_baseline.py computes it, ARCO-CDS (C): mean {dd.mean():.6f} SD {dd.std():.6f} "
             f"p95|d| {np.percentile(np.abs(dd), 95):.6f} p99|d| {np.percentile(np.abs(dd), 99):.6f}"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
