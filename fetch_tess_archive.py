"""fetch_tess_archive.py — list the STARS4ALL TESS archive, select stations, download their monthly files.

The archive holds every photometer's monthly files on a public Nextcloud share. Four resumable steps:
  --scan      list every photometer folder and its monthly files, read the latest header (position,
              location) and write data/tess_archive/stations.json;
  --select    choose the stations (rule below) and write data/tess_archive/selected.csv;
  --download  fetch every selected monthly file from FIRST to LAST into data/tess_archive/<id>/;
  --splits    fetch the parts of split months. When a photometer's location or zero point changes
              within a month the archive files that month as <id>_YYYY-MM_N.dat, one part per location
              record; parts with a negative N have no location set and are not fetched. The parts
              fetched are listed in data/tess_archive/split_files.csv.

Selection: every photometer with a position and at least MIN_MONTHS monthly files from FIRST
to LAST. Files are stored unchanged (STARS4ALL TESS network, ODbL 1.0). Standard library only.

Usage: python3 fetch_tess_archive.py --scan | --select | --download [--threads 2] [--pause 0.5] | --splits"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DAV = "https://fta-cloud.fis.ucm.es/public.php/webdav/"
SHARE = "Gr9DAbfiX8Pdm86"
AUTH = "Basic " + base64.b64encode(f"{SHARE}:".encode()).decode()

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "tess_archive"
STATIONS = OUT / "stations.json"
SELECTED = OUT / "selected.csv"
LOG = OUT / "download.log"

FIRST, LAST = "2019-01", "2026-09"          # 2026-09 is the last complete month at the freeze
MIN_MONTHS = 12
MONTH = re.compile(r"_(\d{4}-\d{2})\.dat$")
PART = re.compile(r"_(\d{4}-\d{2})_(\d+)\.dat$")      # a part of a split month with a location set
SPLITS = OUT / "split_files.csv"


def request(path: str, method: str = "GET", headers: dict | None = None, tries: int = 5) -> bytes:
    h = {"Authorization": AUTH, **(headers or {})}
    for i in range(tries):
        try:
            req = urllib.request.Request(DAV + path, method=method, headers=h)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"gave up on {path}")


def listing(path: str = "") -> list[str]:
    txt = request(path, "PROPFIND", {"Depth": "1"}).decode("utf-8", "replace")
    hrefs = re.findall(r"<d:href>([^<]*)</d:href>", txt)
    return [urllib.request.unquote(h.split("/public.php/webdav/")[1]) for h in hrefs][1:]


def scan_one(folder: str) -> dict:
    sid = folder.strip("/")
    try:
        files = sorted(f for f in listing(folder) if MONTH.search(f))
        months = [MONTH.search(f).group(1) for f in files]
        row = {"id": sid, "months": months, "lat": None, "lon": None, "location": ""}
        if files:
            head = request(files[-1], headers={"Range": "bytes=0-2500"}).decode("utf-8", "replace")
            pos = re.search(r"# Position:\s*([-\d.]+),\s*([-\d.]+)", head)
            loc = re.search(r"# Location name:\s*(.*)", head)
            if pos:
                row["lat"], row["lon"] = float(pos.group(1)), float(pos.group(2))
            row["location"] = loc.group(1).strip() if loc else ""
        return row
    except Exception as e:                                       # keep going; record the failure
        return {"id": sid, "months": [], "lat": None, "lon": None, "location": f"ERROR {e}"}


def scan():
    OUT.mkdir(parents=True, exist_ok=True)
    folders = [f for f in listing() if f.endswith("/")]
    print(f"{len(folders)} folders; reading listings...", file=sys.stderr)
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(scan_one, folders))
    STATIONS.write_text(json.dumps({"scanned": time.strftime("%Y-%m-%d"), "stations": rows}, indent=1))
    print(f"{len(rows)} stations -> {STATIONS}")


FIXES = OUT.parent / "station_fixes.csv"


def apply_fixes(rows: list[dict]) -> list[dict]:
    """Replace archive positions known to be wrong (data/station_fixes.csv, with the evidence)."""
    if not FIXES.exists():
        return rows
    fix = {r["id"]: r for r in csv.DictReader(l for l in FIXES.open() if not l.startswith("#"))}
    for r in rows:
        if r["id"] in fix:
            f = fix[r["id"]]
            r["lat"], r["lon"], r["location"] = float(f["lat"]), float(f["lon"]), f["location"]
    return rows


def select():
    rows = apply_fixes(json.loads(STATIONS.read_text())["stations"])
    keep = []
    for r in rows:
        n = sum(FIRST <= x <= LAST for x in r["months"])
        if r["lat"] is not None and n >= MIN_MONTHS:
            keep.append([r["id"], r["lat"], r["lon"], r["location"], n])
    with SELECTED.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "location", "n_months"])
        w.writerows(keep)
    print(f"{len(keep)} selected; {sum(k[4] for k in keep)} monthly files -> {SELECTED}")


def download(threads: int, pause: float):
    rows = json.loads(STATIONS.read_text())["stations"]
    months = {r["id"]: r["months"] for r in rows}
    ids = [r["id"] for r in csv.DictReader(SELECTED.open())]
    todo = []
    for sid in ids:
        for m in months[sid]:
            if FIRST <= m <= LAST:
                dest = OUT / sid / f"{sid}_{m}.dat"
                if not dest.exists():
                    todo.append((sid, m, dest))
    print(f"{len(todo)} files to fetch for {len(ids)} stations", file=sys.stderr)

    def one(job):
        sid, m, dest = job
        try:
            data = request(f"{sid}/{sid}_{m}.dat")
            if not data.startswith(b"#"):
                return f"BAD     {sid}_{m} (not an NSBM file, {len(data)} bytes)"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)                                   # never leave a half file
            time.sleep(pause)
            return f"ok      {sid}_{m} {len(data)}"
        except Exception as e:
            return f"ERROR   {sid}_{m} {e}"

    done = 0
    with LOG.open("a") as log, ThreadPoolExecutor(threads) as ex:
        for msg in ex.map(one, todo):
            done += 1
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            if done % 100 == 0 or not msg.startswith("ok"):
                log.flush()
                print(f"[{done}/{len(todo)}] {msg}", file=sys.stderr, flush=True)
    print(f"finished: {done} attempted", file=sys.stderr)


def splits(pause: float):
    ids = [r["id"] for r in csv.DictReader(SELECTED.open())]
    rows = []
    for sid in ids:
        for f in sorted(listing(f"{sid}/")):
            m = PART.search(f)
            if not m or not FIRST <= m.group(1) <= LAST:
                continue
            dest = OUT / sid / Path(f).name
            if not dest.exists():
                data = request(f)
                if not data.startswith(b"#"):
                    print(f"BAD     {f} (not an NSBM file, {len(data)} bytes)", file=sys.stderr)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                time.sleep(pause)
            rows.append([sid, m.group(1), m.group(2), Path(f).name])
    with SPLITS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "month", "part", "file"])
        w.writerows(rows)
    print(f"{len(rows)} parts of {len({(r[0], r[1]) for r in rows})} split months -> {SPLITS}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between requests per thread")
    a = ap.parse_args()
    if a.scan:
        scan()
    if a.select:
        select()
    if a.download:
        download(a.threads, a.pause)
    if a.splits:
        splits(a.pause)
    if not (a.scan or a.select or a.download or a.splits):
        ap.print_help()


if __name__ == "__main__":
    main()
