"""build_nights.py — night-time TESS-W infrared record in 5-minute bins, with ERA5-Land air temperature and dew
point, for every selected photometer.

Per photometer, every reading from data/tess_archive/<id>/ with the Sun below SUN_MAX deg is kept, then:
  - the no-read code (sky exactly 0.00 C) and the error code (-273.15 C) are dropped and counted;
  - a reading in the firmware floor band (-50, -49.7] C is flagged `floor`: the first firmware drops sky readings
    at or below -50 C and repeats the last value, so the true sky is at or below the floor (censored, not a value);
  - a reading equal to the one before it in a run of HELD_RUN or more is flagged `held` (a repeated value, not a
    new measurement); held readings are not used for medians or spreads;
  - readings are grouped into BIN_MIN-minute bins (UTC) and summarised: median sky and die temperature, the
    standard deviation of sky readings within the bin (the temporal test), median brightness (mag/arcsec2),
    reading counts, and the shares flagged floor and held.
Sun and Moon positions (PyEphem, at bin centres) and the hourly ERA5-Land 2 m air temperature and dew point
(data/era5_ts/<id>.csv.gz, linearly interpolated to the bin centre) are added. Photometers listed in
data/excluded_ir.csv are skipped, and so are the archive months listed in data/excluded_months.csv (two devices
under one identifier, audit_sequence.py) and data/excluded_positions.csv (recorded away from the position used,
audit_positions.py; its "all" rows skip the photometer). Positions: data/tess_archive/selected.csv (station fixes applied).

Writes data/nights/<id>.csv.gz and results/build_nights.txt (counts per photometer and in total). Reads the raw
archive, so it is not a verify target.

Usage: python build_nights.py [--only stars645 ...] [--workers 6]"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ephem
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
ERA5 = ROOT / "data" / "era5_ts"
OUT = ROOT / "data" / "nights"
EXCLUDED = ROOT / "data" / "excluded_ir.csv"
EXCLUDED_MONTHS = ROOT / "data" / "excluded_months.csv"
EXCLUDED_POSITIONS = ROOT / "data" / "excluded_positions.csv"
REPORT = ROOT / "results" / "build_nights.txt"
SUN_MAX = -12.0
BIN_MIN = 5
HELD_RUN = 5
FLOOR = (-50.0, -49.7)


def exclusion_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(l for l in path.open() if not l.startswith("#"))) if path.exists() else []


def excluded_months(uid: str) -> set[str]:
    rows = exclusion_rows(EXCLUDED_MONTHS) + exclusion_rows(EXCLUDED_POSITIONS)
    return {r["month"] for r in rows if r["id"] == uid}


def read_unit(uid: str) -> pd.DataFrame:
    frames = []
    skip = excluded_months(uid)
    for f in sorted((ARCHIVE / uid).glob("*.dat")):
        if f.stem.split("_")[1] in skip:
            continue
        try:
            d = pd.read_csv(f, sep=";", comment="#", header=None, usecols=[0, 2, 3, 5],
                            names=["t", "die", "sky", "mag"], on_bad_lines="skip", engine="c")
        except (pd.errors.EmptyDataError, ValueError):
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["t", "die", "sky", "mag"])
    d = pd.concat(frames, ignore_index=True)
    d["t"] = pd.to_datetime(d["t"].astype(str).str[:19], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    for c in ("die", "sky", "mag"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["t", "sky"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def sun_moon(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    o = ephem.Observer()
    o.lat, o.lon, o.pressure = str(lat), str(lon), 0
    sun, moon = ephem.Sun(), ephem.Moon()
    rows = []
    for t in times:
        o.date = t.to_pydatetime()
        sun.compute(o)
        moon.compute(o)
        rows.append((math.degrees(sun.alt), math.degrees(moon.alt), moon.moon_phase))
    return pd.DataFrame(rows, index=times, columns=["sun_alt", "moon_alt", "moon_illum"])


def one(args) -> str:
    uid, lat, lon = args
    out = OUT / f"{uid}.csv.gz"
    if out.exists():
        return f"{uid}: exists"
    d = read_unit(uid)
    n_all = len(d)
    if not n_all:
        return f"{uid}: no readings"
    noread, err = d["sky"] == 0.0, (d["sky"] <= -273) | (d["die"] <= -273)
    d = d[~noread & ~err].copy()
    # night only: Sun position at each reading (vectorised low-precision formula is enough for the cut;
    # the exact positions are recomputed per bin below)
    from audit_ir import sun_altitude
    hours = d["t"].dt.floor("15min")
    alt = {h: sun_altitude(h.to_pydatetime(), lat, lon) for h in hours.unique()}
    d = d[hours.map(alt) < SUN_MAX + 1].copy()
    d["floor"] = (d["sky"] > FLOOR[0]) & (d["sky"] <= FLOOR[1])
    same = d["sky"].eq(d["sky"].shift()) & (d["t"].diff() < pd.Timedelta(minutes=3))
    run_id = (~same).cumsum()
    run_len = d.groupby(run_id)["sky"].transform("size")
    d["held"] = same & (run_len >= HELD_RUN)
    d["bin"] = d["t"].dt.floor(f"{BIN_MIN}min")
    fresh = d[~d["held"]]
    g = fresh.groupby("bin")
    b = pd.DataFrame({
        "sky_med": g["sky"].median(), "sky_std": g["sky"].std(), "n": g["sky"].size(),
        "die_med": g["die"].median(), "mag_med": g["mag"].median(),
    })
    ga = d.groupby("bin")
    b = b.join(pd.DataFrame({"n_all": ga["sky"].size(), "floor_share": ga["floor"].mean(),
                             "held_share": ga["held"].mean()}), how="outer")
    b.index = b.index + pd.Timedelta(minutes=BIN_MIN / 2)          # bin centre
    b = b.join(sun_moon(b.index, lat, lon))
    b = b[b["sun_alt"] < SUN_MAX]
    e = pd.read_csv(ERA5 / f"{uid}.csv.gz", parse_dates=["time"]).set_index("time") if (ERA5 / f"{uid}.csv.gz").exists() else None
    if e is not None:
        e = e[~e.index.duplicated()].dropna(subset=["t2m_C"])
        et = e.index.asi8 / 1e9
        bt = b.index.asi8 / 1e9
        near = np.abs(et[np.clip(np.searchsorted(et, bt), 1, len(et) - 1) - 1] - bt)   # to the hour before
        for c in ("t2m_C", "d2m_C", "sde_m"):
            v = np.interp(bt, et, e[c].to_numpy(float))
            v[(near > 3600 * 1.5) | (bt < et[0]) | (bt > et[-1])] = np.nan              # no ERA5 hour within reach
            b[c] = v
    else:
        b[["t2m_C", "d2m_C", "sde_m"]] = np.nan
    b.index.name = "time"
    b = b.round({"sky_med": 2, "sky_std": 3, "die_med": 2, "mag_med": 2, "floor_share": 3, "held_share": 3,
                 "sun_alt": 2, "moon_alt": 2, "moon_illum": 3, "t2m_C": 2, "d2m_C": 2})
    OUT.mkdir(parents=True, exist_ok=True)
    b.to_csv(out.with_suffix(".part"), compression="gzip")
    out.with_suffix(".part").replace(out)
    return (f"{uid}: {n_all} readings, no-read {int(noread.sum())}, error {int(err.sum())}, "
            f"night bins {len(b)}, floor bins {int((b['floor_share'] > 0.5).sum())}, "
            f"held readings {int(d['held'].sum())}, ERA5 {'yes' if e is not None else 'MISSING'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    skip = {r["id"] for r in exclusion_rows(EXCLUDED)} | {r["id"] for r in exclusion_rows(EXCLUDED_POSITIONS) if r["month"] == "all"}
    jobs = [(r["id"], float(r["lat"]), float(r["lon"])) for r in csv.DictReader((ARCHIVE / "selected.csv").open())
            if r["id"] not in skip and (not a.only or r["id"] in a.only)]
    with ProcessPoolExecutor(a.workers) as ex:
        msgs = list(ex.map(one, jobs))
    for m in msgs:
        print(m, flush=True)
    REPORT.parent.mkdir(exist_ok=True)
    if not a.only:
        REPORT.write_text(f"photometers: {len(jobs)} (excluded for broken infrared: {len(skip)})\n" + "\n".join(sorted(msgs)) + "\n")


if __name__ == "__main__":
    main()
