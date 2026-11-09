"""cloud_record.py — the released cloud record: the two-test call for every 5-minute night bin of every photometer of
the record, with the flags a user needs, and one summary row per photometer.

For each photometer of the record (qc_baseline.with_baseline) the bins of data/features/<id>.csv.gz are scored with
the frozen thresholds (data/frozen_thresholds.csv, as sat_score.py): cloudy = 1 when z_brunt > kz or s5_excess > kv,
0 otherwise; a bin mostly at the firmware floor (floor >= 0.5) is clear; a bin where neither test is defined (no
baseline, no reading) has no call (empty). Flags per bin: floor (mostly at the -50 C floor), snow (ERA5-Land snow
depth > SNOW_M, where the night satellite masks are unreliable), moon_up, dark (Sun below -18 deg), obstructed (the bin's calendar month fails the obstruction rule of qc_ir.py
applied month by month, results/qc_ir_months.csv). Flags per
photometer, repeated on every bin for convenience: mountain (1500 m and above on the terrain model, or
sat_score.MOUNTAIN; the baseline there uses ERA5 pressure levels) and weak (clear-sky baseline above the network's
Tukey fence, qc_baseline.py: calls close to chance against the satellites).

Writes data/record/<id>.csv.gz (time UTC, sky_C, z, s5_excess_C, cloudy, floor, snow, moon_up, dark,
obstructed) and
results/cloud_record.csv, one row per photometer: id, lat, lon, dem_m, mountain, weak, sd_brunt, first, last, bins,
bins_called, cloudy_share, obstructed_months, and the per-photometer two-test balanced accuracy against each satellite mask on
snow-free lowland scans where at least 100 clear and 100 cloudy scans exist (results/sat_score_sites_dem.csv),
plus results/cloud_record.txt (the column key and the totals).

Usage: python cloud_record.py [--workers 12]"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import sat_score as ss
from qc_baseline import weak, with_baseline
from qc_ir import obstructed_months

ROOT = Path(__file__).parent
FEAT = ROOT / "data" / "features"
OUT_DIR = ROOT / "data" / "record"
OUT_CSV = ROOT / "results" / "cloud_record.csv"
OUT_TXT = ROOT / "results" / "cloud_record.txt"
SNOW_M = ss.SNOW_M

KEY = """Columns of data/record/<id>.csv.gz (one row per 5-minute night bin, Sun below -12 deg):
  time          bin centre, UTC
  sky_C         median sky temperature of the bin (C)
  z             baseline residual over the photometer's held-out clear-sky SD (Eq. 2 of the paper); -inf at the floor
  s5_excess_C   within-bin scatter minus the clear-sky scatter at that sky temperature (C)
  cloudy        1 cloudy, 0 clear (the two-test classifier, frozen thresholds), empty where neither test is defined;
                a bin mostly at the floor is clear even when the scatter test fires, as Sect. 3 of the paper states
                (the scoring scripts let the scatter test fire there: 585 of 43.9 million bins, 1.3e-5, differ)
  floor         1 when the bin is mostly at the -50 C firmware floor (called clear; colder cloud cannot be seen)
  snow          1 when ERA5-Land snow depth > 0.01 m (night satellite masks unreliable; the record itself is unaffected)
  moon_up       1 when the Moon is above the horizon
  dark          1 when the Sun is below -18 deg
  obstructed    1 when the bin's calendar month (UTC) fails the obstruction test month by month: the 95th percentile
                of ERA5-Land air minus sky temperature over the month's dark bins below 8 C (the sensor does not see
                the sky; the calls of that month are unreliable)
Per-photometer flags (results/cloud_record.csv): mountain (1500 m and above; ERA5 pressure-level baseline; satellite
masks may report cloud below the site), weak (clear-sky baseline too loose: scores near chance against the satellites),
and the two-test balanced accuracy against GOES, Meteosat and VIIRS on snow-free lowland scans (empty where the
photometer has fewer than 100 clear or 100 cloudy matched scans, or is a mountain site)."""


def one(args) -> list:
    uid, th, is_mountain, is_weak, bad_months = args
    f = pd.read_csv(FEAT / f"{uid}.csv.gz", parse_dates=["time"])
    z, s5, floor = f["z_brunt"].to_numpy(float), f["s5_excess"].to_numpy(float), f["floor"].to_numpy(float)
    at_floor = floor >= 0.5
    defined = np.isfinite(z) | at_floor | np.isfinite(s5)
    call = ((z > th["kx"]) | (s5 > th["kv"])) & ~at_floor
    out = pd.DataFrame({"time": f["time"].dt.strftime("%Y-%m-%d %H:%M"), "sky_C": f["sky_med"].round(2),
                        "z": np.where(at_floor, -np.inf, z.round(3)), "s5_excess_C": s5.round(3),
                        "cloudy": pd.array(np.where(defined, call.astype(int), -1), dtype="Int64").to_numpy(),
                        "floor": at_floor.astype(int), "snow": (f["sde_m"] > SNOW_M).astype(int),
                        "moon_up": f["moon_up"].astype(int), "dark": f["dark"].astype(int),
                        "obstructed": f["time"].dt.strftime("%Y-%m").isin(bad_months).astype(int)})
    out["cloudy"] = out["cloudy"].where(out["cloudy"] >= 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / f"{uid}.csv.gz", index=False, compression="gzip")
    n_called = int(defined.sum())
    return [uid, int(is_mountain), int(is_weak), f["time"].min().strftime("%Y-%m-%d"), f["time"].max().strftime("%Y-%m-%d"),
            len(f), n_called, round(float(call[defined].mean()), 4) if n_called else "", len(bad_months)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    th = ss.thresholds()
    ids = sorted(with_baseline() & {f.name.split(".")[0] for f in FEAT.glob("*.csv.gz")})
    dem = pd.read_csv(ROOT / "data" / "elevation.csv").set_index("id")
    mountain = ss.MOUNTAIN | {u for u, zm in dem["dem_m"].fillna(0.0).items() if zm >= ss.MOUNTAIN_M}
    weak_ids = weak()
    bad = obstructed_months()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(u, th, u in mountain, u in weak_ids, sorted(m for i, m in bad if i == u)) for u in ids]))
    t = pd.DataFrame(rows, columns=["id", "mountain", "weak", "first", "last", "bins", "bins_called", "cloudy_share",
                                    "obstructed_months"])
    t.insert(1, "lat", t["id"].map(dem["lat"]))
    t.insert(2, "lon", t["id"].map(dem["lon"]))
    t.insert(3, "dem_m", t["id"].map(dem["dem_m"]))
    units = pd.read_csv(ROOT / "results" / "features_units.csv").set_index("id")["sd_brunt"]
    t.insert(6, "sd_brunt", t["id"].map(units))
    sites = pd.read_csv(ROOT / "results" / "sat_score_sites_dem.csv")
    for ref in ss.REFS:
        s = sites[sites["reference"] == ref].set_index("id")["two_test"]
        t[f"ba_{ref}"] = t["id"].map(s)
    t.to_csv(OUT_CSV, index=False)
    lines = [f"cloud record: {len(t)} photometers, {int(t['bins'].sum())} night bins, {int(t['bins_called'].sum())} with a call; "
             f"mountain {int(t['mountain'].sum())}, weak baseline {int(t['weak'].sum())}; obstructed months "
             f"{int(t['obstructed_months'].sum())} at {int((t['obstructed_months'] > 0).sum())} photometers; thresholds z > {th['kx']} or "
             f"s5_excess > {th['kv']} C (frozen)", "", KEY]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    main()
