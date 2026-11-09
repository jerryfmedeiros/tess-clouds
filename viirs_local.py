"""viirs_local.py — the VIIRS cloud mask at every photometer from the granules already kept on disk, for the months
NASA no longer lists.

The first VIIRS extraction (viirs_extract.py) sampled only the METAR-paired photometers and kept every granule it
downloaded (--keep). Since then CMR lists no CLDMSK_L2_VIIRS_NOAA20 granules before August 2023 (version 1 withdrawn,
version 2 not yet back-filled; checked 2026-10-31), so those months cannot be downloaded again. A VIIRS swath is
about 3000 km wide, so the kept granules also cover many of the other photometers. This samples every photometer
in every kept granule of the given months with viirs_extract.sample (same rules: nearest pixel within 2 km, Sun
below SUN_MAX at the granule start, 3x3/5x5/7x7 boxes). Photometers under no kept granule stay uncovered in those
months.

Granule start time from the file name (A<year><day of year>.<HHMM>). Writes data/viirs_local/viirs_NOAA20_<month>.csv
(viirs_extract.py format); a finished month is not redone.

Usage: python viirs_local.py --keep RAW [--months 2019-01 ...] [--workers 8]"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import viirs_extract as ve
from audit_ir import sun_altitude

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "viirs_local"
PLATFORM = "NOAA20"
LAST = "2023-08"
NAME = re.compile(r"\.A(\d{4})(\d{3})\.(\d{2})(\d{2})\.")


def units() -> list[dict]:
    return [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(ve.SELECTED.open())]


def one(path: Path) -> list[list]:
    m = NAME.search(path.name)
    if not m:
        return []
    y, doy, hh, mm = map(int, m.groups())
    start = dt.datetime(y, 1, 1) + dt.timedelta(days=doy - 1, hours=hh, minutes=mm)
    dark = [u for u in UNITS if sun_altitude(start, u["lat"], u["lon"]) < ve.SUN_MAX]
    if not dark:
        return []
    try:
        return ve.sample(path, dark, {"start": start, "url": path.name}, PLATFORM)
    except OSError as e:
        print(f"{path.name}: ERROR {e}", flush=True)
        return []


UNITS = units()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    base = a.keep / "viirs" / PLATFORM
    months = a.months or sorted(p.name for p in base.iterdir() if p.is_dir() and p.name <= LAST)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for month in months:
            out = OUT / f"viirs_{PLATFORM}_{month}.csv"
            if out.exists():
                continue
            files = sorted((base / month).glob("*.nc"))
            rows = [r for res in ex.map(one, files, chunksize=4) for r in res]
            part = out.with_suffix(".part")
            with part.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(ve.FIELDS)
                w.writerows(rows)
            part.replace(out)
            print(f"{PLATFORM} {month}: {len(files)} kept granules, {len(rows)} rows, "
                  f"{len({r[2] for r in rows})} photometers", flush=True)


if __name__ == "__main__":
    main()
