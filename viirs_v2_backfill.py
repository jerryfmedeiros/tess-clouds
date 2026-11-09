"""viirs_v2_backfill.py — the VIIRS cloud mask in version 2 (LAADS ArchiveSet 5200, file version .002.) for the months
the CMR catalogue lists only version 1 or nothing, at every photometer.

Fernandes Medeiros (2026) used CLDMSK_L2_VIIRS v2.0 (ArchiveSet 5200) throughout. The first extraction here got version 1
(ArchiveSet 5110, .001.) for 2019 to July 2023, because that is what CMR returned for those dates; CMR now lists
version 2 only from August 2023, but the LAADS archive holds version 2 for every year (checked 2026-10-31: 240 files
a day in 5200/CLDMSK_L2_VIIRS_NOAA20/<year>/<day of year>/ for 2020, 2023 and 2024). The version 2 granules have the
same start times as version 1, so the granules to fetch are the version 2 twins of the version 1 night granules
kept by the first run (--keep): same overpasses, current processing. Each is found in the LAADS day listing by its
A<year><doy>.<HHMM> stamp, downloaded to the keep folder beside its version 1 twin, and every photometer is sampled
from it with viirs_extract.sample (nearest pixel within 2 km, Sun below SUN_MAX at the granule start).
Photometers under no kept granule stay uncovered in these months.

Writes data/viirs_v2/viirs_NOAA20_<month>.csv (viirs_extract.py format); a finished month is not redone. Needs an
Earthdata token (viirs_extract.token).

Usage: python viirs_v2_backfill.py --keep RAW [--months 2019-01 ...] [--workers 6]"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import viirs_extract as ve
from audit_ir import sun_altitude

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "viirs_v2"
PLATFORM = "NOAA20"
ARCHIVE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/CLDMSK_L2_VIIRS_NOAA20"
LAST = "2023-08"
STAMP = re.compile(r"\.A(\d{4})(\d{3})\.(\d{4})\.")
UNITS = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(ve.SELECTED.open())]
TOKEN = None


def listing(year: int, doy: int, tok: str) -> dict[str, str]:
    """Version 2 file URL by HHMM stamp for one day."""
    req = urllib.request.Request(f"{ARCHIVE}/{year}/{doy:03d}.json", headers={"Authorization": f"Bearer {tok}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                content = json.load(r)["content"]
            break
        except Exception:
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f"no listing for {year}/{doy:03d}")
    out = {}
    for c in content:
        m = STAMP.search(c["name"])
        if m and ".002." in c["name"] and c["name"].endswith(".nc"):
            out[m.group(3)] = c["downloadsLink"]
    return out


def one(job) -> list[list]:
    url, dest, start = job
    dark = [u for u in UNITS if sun_altitude(start, u["lat"], u["lon"]) < ve.SUN_MAX]
    if not dark:
        return []
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


def init(tok: str):
    global TOKEN
    TOKEN = tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    base = a.keep / "viirs" / PLATFORM
    months = a.months or sorted(p.name for p in base.iterdir() if p.is_dir() and p.name <= LAST)
    tok = ve.token()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(tok,)) as ex:
        for month in months:
            out = OUT / f"viirs_{PLATFORM}_{month}.csv"
            if out.exists():
                continue
            v1 = sorted({STAMP.search(f.name).groups() for f in (base / month).glob("*.001.*.nc") if STAMP.search(f.name)})
            days, jobs, missing = {}, [], 0
            for y, doy, hhmm in v1:
                key = (int(y), int(doy))
                if key not in days:
                    days[key] = listing(*key, tok)
                url = days[key].get(hhmm)
                if url is None:
                    missing += 1
                    continue
                start = dt.datetime(int(y), 1, 1) + dt.timedelta(days=int(doy) - 1, hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
                jobs.append((url, base / month / url.rsplit("/", 1)[-1], start))
            rows = [r for res in ex.map(one, jobs) for r in res]
            part = out.with_suffix(".part")
            with part.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(ve.FIELDS)
                w.writerows(rows)
            part.replace(out)
            print(f"{PLATFORM} {month}: {len(v1)} version 1 granules, {len(jobs)} version 2 twins "
                  f"({missing} without a twin), {len(rows)} rows, {len({r[2] for r in rows})} photometers", flush=True)


if __name__ == "__main__":
    main()
