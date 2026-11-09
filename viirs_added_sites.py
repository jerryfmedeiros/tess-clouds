"""viirs_added_sites.py — the VIIRS cloud mask (version 2) before August 2023 at the photometers added to the
extraction list after the first run.

The first extraction (viirs_extract.py, 2019-01 to 2023-07) and its version 2 backfill (viirs_v2_backfill.py)
sampled the photometers selected at the time; the selection was extended in 2023-08 and the earlier months were
not redone for the added sites, so a photometer added then has no VIIRS row before 2023-08. The CMR catalogue no
longer lists any cloud-mask granule before 2023-08 (checked 2026-11-04), so the footprints come from the NOAA-20
geolocation product VJ103MOD (collection 2.1, listed with footprints from 2018), which shares the 6-minute granule
boundaries of the cloud mask. For each month this script lists the night VJ103MOD granules, keeps those whose
footprint holds one of the given photometers with the Sun below viirs_extract.SUN_MAX and with readings in that
month (data/features), finds the version 2 cloud-mask granule with the same start stamp in the LAADS day listing (viirs_v2_backfill.listing), downloads it
to the keep folder beside the other granules unless it is already there, and samples the given photometers from it
(viirs_extract.sample).

Writes data/viirs/viirs_NOAA20_<month>_added.csv.gz (viirs_extract.py format), which sat_score.load_ref reads with
the monthly files; a finished month is not redone. Needs an Earthdata token (viirs_extract.token).

Usage: python viirs_added_sites.py --only stars247 ... --keep RAW
                                   [--months 2019-01 ...] [--workers 6]"""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import viirs_extract as ve
import viirs_v2_backfill as bf
from audit_ir import sun_altitude

PLATFORM = "NOAA20"
GEO = "VJ103MOD"           # NOAA-20 geolocation granules, listed in CMR with footprints for every year
ve.SHORT_NAME["GEO"] = GEO  # so that viirs_extract.cmr_month lists them
LAST = "2023-07"           # the first month with every selected photometer in the monthly files is 2023-08
TOKEN = None
UNITS: list[dict] = []


def init(tok: str, units: list[dict]):
    global TOKEN, UNITS
    TOKEN, UNITS = tok, units


def one(job) -> list[list]:
    url, dest, start, ids = job
    dark = [u for u in UNITS if u["id"] in ids]
    if not (dest.exists() and dest.stat().st_size > 0):
        part = dest.with_suffix(".part")
        r = subprocess.run(["curl", "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "20", "--max-time", "600",
                            "-H", f"Authorization: Bearer {TOKEN}", "-o", str(part), url], capture_output=True, text=True)
        if r.returncode:
            print(f"{dest.name}: ERROR {r.stderr.strip()[:200]}", flush=True)
            part.unlink(missing_ok=True)
            return []
        part.replace(dest)
    try:
        return ve.sample(dest, dark, {"start": start, "url": dest.name}, PLATFORM)
    except OSError as e:
        print(f"{dest.name}: ERROR {e}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", required=True, help="photometers added to the list after the first run")
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    units = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(ve.SELECTED.open())
             if r["id"] in a.only]
    missing_ids = set(a.only) - {u["id"] for u in units}
    if missing_ids:
        raise SystemExit(f"not in {ve.SELECTED.name}: {sorted(missing_ids)}")
    months = a.months or [f"{y}-{m:02d}" for y in range(ve.START.year, 2024) for m in range(1, 13)
                          if ve.START <= dt.date(y, m, 1) and f"{y}-{m:02d}" <= LAST]
    tok = ve.token()
    base = a.keep / "viirs" / PLATFORM
    active = {u["id"]: set(pd.read_csv(ve.ROOT / "data" / "features" / f"{u['id']}.csv.gz", usecols=["time"])["time"]
                           .str.slice(0, 7)) for u in units}          # months with readings, per photometer
    ve.OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(tok, units)) as ex:
        for month in months:
            out = ve.OUT / f"viirs_{PLATFORM}_{month}_added.csv.gz"
            if out.exists():
                continue
            days, jobs, listed, missing = {}, [], 0, 0
            here = [u for u in units if month in active[u["id"]]]
            for g in (ve.cmr_month("GEO", month) if here else []):
                listed += 1
                dark = [u["id"] for u in here if sun_altitude(g["start"], u["lat"], u["lon"]) < ve.SUN_MAX
                        and any(ve.inside(u["lat"], u["lon"], p) for p in g["polys"])]
                if not dark:
                    continue
                key = (g["start"].year, g["start"].timetuple().tm_yday)
                if key not in days:
                    days[key] = bf.listing(*key, tok)
                url = days[key].get(f"{g['start']:%H%M}")
                if url is None:
                    missing += 1
                    continue
                dest = base / month / url.rsplit("/", 1)[-1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((url, dest, g["start"], dark))
            rows = [r for res in ex.map(one, jobs) for r in res]
            part = out.with_suffix(".part")
            with gzip.open(part, "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(ve.FIELDS)
                w.writerows(rows)
            part.replace(out)
            print(f"{PLATFORM} {month}: {len(here)} photometers with readings, {listed} night {GEO} granules listed, "
                  f"{len(jobs)} over them "
                  f"({missing} without a version 2 cloud-mask twin), {len(rows)} rows, {len({r[2] for r in rows})} photometers",
                  flush=True)


if __name__ == "__main__":
    main()
