"""goes_local.py — the GOES ABI cloud mask at given photometers from the scans already kept on disk (goes_extract.py
--keep), for photometers whose position was corrected after the extraction: the same half-hourly times, dark units,
nearest kept scan within 10 min (GOES-16 preferred before the GOES-19 handover, GOES-19 after), and the same sampling
(goes_extract.sample). A time with no kept scan (the first extraction downloaded scans only for times when some
photometer of the Americas was dark, and a La Palma evening is an American afternoon) is fetched from NOAA's open
data as goes_extract.py does, sampled and deleted (--keep-new keeps them). A finished month (its output file exists) is not redone.

Writes <out>/goes_<month>.csv.gz (goes_extract.py format) for the given photometers only, to be merged into
data/goes/ by replacing those photometers' rows (used for stars403 and stars411, whose positions were corrected).

Usage: python goes_local.py --only stars403 stars411 --keep RAW --out DIR [--months 2021-11 ...]"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import datetime as dt
import gzip
import re
from pathlib import Path

import goes_extract as ge
from audit_ir import sun_altitude


def kept_scans(keep: Path, month: str) -> list[tuple[dt.datetime, str, Path]]:
    out = []
    for f in sorted((keep / "goes" / month).glob("*.nc")) if (keep / "goes" / month).exists() else []:
        m = re.search(r"_G(\d\d)_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", f.name)
        if not m:
            continue
        sat, y, d, hh, mm, ss = m.groups()
        out.append((dt.datetime(int(y), 1, 1, int(hh), int(mm), int(ss)) + dt.timedelta(days=int(d) - 1), f"goes{sat}", f))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", required=True)
    ap.add_argument("--keep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--step", type=int, default=ge.STEP_MIN)
    ap.add_argument("--keep-new", action="store_true", help="also move fetched scans into the keep folder")
    a = ap.parse_args()
    units = [{"id": r["id"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(ge.SELECTED.open())
             if r["id"] in a.only]
    months = a.months or [f"{y}-{m:02d}" for y in range(ge.START.year, ge.END.year + 1) for m in range(1, 13)
                          if ge.START <= dt.datetime(y, m, 28) <= ge.END]
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="goes_local_"))
    for month in months:
        if (a.out / f"goes_{month}.csv.gz").exists():
            continue
        scans = kept_scans(a.keep, month)
        rows, missing, fetched = [], 0, 0
        for t in ge.month_times(month, a.step):
            dark = [u for u in units if sun_altitude(t, u["lat"], u["lon"]) < ge.SUN_MAX]
            if not dark or not scans:
                continue
            order = ("goes16", "goes19") if t < ge.HANDOVER else ("goes19", "goes16")
            best = None
            for sat in order:
                near = [(abs((s - t).total_seconds()), s, f) for s, ss, f in scans if ss == sat]
                if near and min(near)[0] <= 600:
                    best = (sat, *min(near)[1:])
                    break
            if best is None:
                found = ge.nearest_scan(t)
                if not found:
                    missing += 1
                    continue
                bucket, key, scan = found
                f = tmp / "scan.nc"
                r = subprocess.run(["curl", "-sS", "--fail", "--retry", "4", "--max-time", "300", "-o", str(f),
                                    f"https://{bucket}.s3.amazonaws.com/{key}"], capture_output=True, text=True)
                if r.returncode:
                    missing += 1
                    continue
                rows += ge.sample(f, dark, t, scan, bucket.replace("noaa-", ""))
                if a.keep_new:
                    dest = a.keep / "goes" / month / Path(key).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(f, dest)
                else:
                    f.unlink()
                fetched += 1
                continue
            sat, scan, f = best
            try:
                rows += ge.sample(f, dark, t, scan, sat)
                continue
            except OSError:                              # a damaged kept scan: drop it and fetch it again
                f.unlink(missing_ok=True)
                scans = [x for x in scans if x[2] != f]
            found = ge.nearest_scan(t)
            if not found:
                missing += 1
                continue
            bucket, key, scan = found
            g = tmp / "scan.nc"
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "4", "--max-time", "300", "-o", str(g),
                                f"https://{bucket}.s3.amazonaws.com/{key}"], capture_output=True, text=True)
            if r.returncode:
                missing += 1
                continue
            rows += ge.sample(g, dark, t, scan, bucket.replace("noaa-", ""))
            g.unlink()
            fetched += 1
        out = a.out / f"goes_{month}.csv.gz"
        part = out.with_suffix(".part")
        with gzip.open(part, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(ge.FIELDS)
            w.writerows(rows)
        part.replace(out)
        print(f"{month}: {len(rows)} rows, {fetched} scans fetched, {missing} times with no scan", flush=True)


if __name__ == "__main__":
    main()
