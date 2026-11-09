"""mountain_pl.py — air temperature and dew point at the height of every mountain photometer, from ERA5 pressure
levels, for the clear-sky baseline (features.py uses them in place of ERA5-Land at these photometers).

A mountain photometer is one at or above MOUNTAIN_M on the DEM (data/elevation.csv; the same rule as the download,
fetch_mountain_met.py). For each, the nearest 0.25 deg point of its group's files (fetch_mountain_met.py --arco,
00/06/12/18 UTC) is read; temperature and relative humidity are interpolated in log pressure to the
standard-atmosphere pressure of the DEM elevation, and the dew point is computed from them as in mountain_baseline.py
(Magnus). Needs the raw files under TESS_CLOUDS_RAW (not a verify target); the per-photometer series it writes are small and
shipped.
Writes data/mountain_pl/<id>.csv.gz (time UTC, t_C, td_C) and results/mountain_pl.txt (photometer, grid point,
distance, pressure, hours).

Usage: python mountain_pl.py [--keep RAW]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from fetch_mountain_met import GROUPS, MOUNTAIN_M, p_std
from mountain_baseline import dew_from_rh

RAW = Path(os.environ.get("TESS_CLOUDS_RAW", Path(__file__).parent / "data" / "raw"))   # raw files, outside git

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "mountain_pl"
TXT = ROOT / "results" / "mountain_pl.txt"


def extract(f: Path, units: pd.DataFrame) -> dict:
    """The series of every photometer inside one file's box (each file is read once: its chunks hold whole hours)."""
    out = {}
    with netCDF4.Dataset(f) as nc:
        la, lo = np.asarray(nc["latitude"][:]), np.asarray(nc["longitude"][:])
        inside = [r for r in units.itertuples() if la.min() - 0.125 <= r.lat <= la.max() + 0.125
                  and lo.min() - 0.125 <= r.lon <= lo.max() + 0.125]
        if not inside:
            return out
        lev = np.asarray(nc["pressure_level"][:], float)
        times = pd.to_datetime(np.asarray(nc["valid_time"][:], "int64"), unit="s")
        t_all, r_all = np.asarray(nc["t"][:]) - np.float32(273.15), np.asarray(nc["r"][:])
    lp, o = np.log(lev), np.argsort(np.log(lev))
    for r in inside:
        i, j = int(np.abs(la - r.lat).argmin()), int(np.abs(lo - r.lon).argmin())
        p = np.log(p_std(r.dem_m))
        ti = np.array([np.interp(p, lp[o], row[o]) for row in t_all[:, :, i, j]])
        ri = np.array([np.interp(p, lp[o], row[o]) for row in r_all[:, :, i, j]])
        out[r.id] = ((float(la[i]), float(lo[j])), pd.DataFrame({"time": times, "t_C": ti, "td_C": dew_from_rh(ti, ri)}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, default=RAW)
    a = ap.parse_args()
    e = pd.read_csv(ROOT / "data" / "elevation.csv")
    e = e[e["dem_m"] >= MOUNTAIN_M].sort_values("id")
    OUT.mkdir(parents=True, exist_ok=True)
    got = {uid: [] for uid in e["id"]}
    points = {}
    for g, (w, east) in GROUPS.items():
        units = e[(e["lon"] >= w) & (e["lon"] < east)]
        for f in sorted((a.keep / "era5pl_arco").glob(f"{g}_*.nc")):
            for uid, (pt, s) in extract(f, units).items():
                got[uid].append(s)
                points[uid] = pt
            print(f.name, flush=True)
    lines = ["id dem_m p_hPa grid_lat grid_lon km hours first last"]
    for r in e.itertuples():
        p = p_std(r.dem_m)
        if not got[r.id]:
            lines.append(f"{r.id} {r.dem_m:.0f} {p:.1f} none")
            continue
        s = pd.concat(got[r.id]).drop_duplicates("time").sort_values("time")
        point = points[r.id]
        km = 6371 * np.hypot(np.radians(point[0] - r.lat), np.radians(point[1] - r.lon) * np.cos(np.radians(r.lat)))
        s.round({"t_C": 3, "td_C": 3}).to_csv(OUT / f"{r.id}.csv.gz", index=False, compression="gzip")
        lines.append(f"{r.id} {r.dem_m:.0f} {p:.1f} {point[0]:.2f} {point[1]:.2f} {km:.1f} {len(s)} "
                     f"{s['time'].min():%Y-%m-%d} {s['time'].max():%Y-%m-%d}")
    TXT.write_text("\n".join(lines) + "\n")
    print(TXT.read_text())


if __name__ == "__main__":
    main()
