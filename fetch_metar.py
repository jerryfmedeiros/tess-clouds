"""fetch_metar.py — METAR reports from airports near the photometers, and which of them report automatically
at night.

Three resumable steps:
  --pairs     pair each selected photometer (data/tess_archive/selected.csv) with every airport within MAX_KM
              that has a four-letter ICAO code (OurAirports, public domain; cached in data/ourairports/).
              Writes data/metar/pairs.csv (photometer, airport, distance).
  --download  raw METAR text for each paired airport, FIRST to LAST, from the Iowa Environmental Mesonet ASOS
              archive (public; one request per airport and year, at most one request per second as IEM asks).
              Writes data/metar/<ICAO>_<year>.csv.gz; a year with no reports is recorded as an empty file.
  --auto      per airport, the share of reports with Sun below -6 deg that carry the AUTO flag, and the
              share of those that report clouds as NCD/NSC/CLR or SKC, to tell automated night reports from
              human ones. Writes results/metar_auto.csv.

Usage: python3 fetch_metar.py --pairs | --download | --auto"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from audit_ir import sun_altitude

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OA = ROOT / "data" / "ourairports" / "airports.csv"
OA_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
METAR = ROOT / "data" / "metar"
PAIRS = METAR / "pairs.csv"
AUTO = ROOT / "results" / "metar_auto.csv"
MAX_KM = 20.0
FIRST, LAST = 2019, 2026
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}&data=metar"
       "&year1={y}&month1=1&day1=1&year2={y2}&month2=1&day2=1&tz=Etc/UTC&format=onlycomma"
       "&latlon=yes&missing=M&trace=T&direct=yes")
TYPES = {"large_airport", "medium_airport", "small_airport"}


def km(a, b, c, d) -> float:
    p = math.radians
    return 2 * 6371 * math.asin(math.sqrt(math.sin(p(c - a) / 2) ** 2
                                          + math.cos(p(a)) * math.cos(p(c)) * math.sin(p(d - b) / 2) ** 2))


def pairs():
    if not OA.exists():
        OA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OA_URL, OA)
    ap = []
    for r in csv.DictReader(OA.open(encoding="utf-8")):
        code = r["icao_code"] or r["gps_code"]
        if r["type"] in TYPES and re.fullmatch(r"[A-Z]{4}", code or ""):
            ap.append((code, r["name"], r["iso_country"], float(r["latitude_deg"]), float(r["longitude_deg"])))
    rows = []
    for t in csv.DictReader(SELECTED.open()):
        la, lo = float(t["lat"]), float(t["lon"])
        for code, name, cc, a_la, a_lo in ap:
            if abs(a_la - la) < 0.3 and abs(a_lo - lo) < 0.5:
                d = km(la, lo, a_la, a_lo)
                if d <= MAX_KM:
                    rows.append([t["id"], code, name, cc, round(d, 2)])
    METAR.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r[0], r[4]))
    with PAIRS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "icao", "airport", "country", "km"])
        w.writerows(rows)
    print(f"{len(rows)} pairs, {len({r[0] for r in rows})} photometers, {len({r[1] for r in rows})} airports -> {PAIRS}")


def download():
    icaos = sorted({r["icao"] for r in csv.DictReader(PAIRS.open())})
    for s in icaos:
        for y in range(FIRST, LAST + 1):
            out = METAR / f"{s}_{y}.csv.gz"
            if out.exists():
                continue
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "3", "--max-time", "300",
                                IEM.format(s=s, y=y, y2=y + 1)], capture_output=True, text=True)
            if r.returncode:
                print(f"{s} {y}: ERROR {r.stderr.strip()[:200]}", flush=True)
                time.sleep(5)
                continue
            body = r.stdout if r.stdout.count("\n") > 1 else ""
            with gzip.open(out.with_suffix(".part"), "wt") as f:
                f.write(body)
            out.with_suffix(".part").replace(out)
            print(f"{s} {y}: {max(0, body.count(chr(10)) - 1)} reports", flush=True)
            time.sleep(1.2)


NIL_CLOUD = re.compile(r"\b(NCD|NSC|CLR|SKC)\b")


def auto():
    pos = {}
    rows = []
    for f in sorted(METAR.glob("*_*.csv.gz")):
        s = f.name.split("_")[0]
        n = n_auto = n_auto_nil = 0
        for r in csv.DictReader(io.StringIO(gzip.open(f, "rt").read())):
            try:
                t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
                la, lo = float(r["lat"]), float(r["lon"])
            except (ValueError, KeyError):
                continue
            if sun_altitude(t, la, lo) >= -6:
                continue
            n += 1
            m = r["metar"]
            if " AUTO " in f" {m} ":
                n_auto += 1
                n_auto_nil += bool(NIL_CLOUD.search(m))
            pos[s] = (la, lo)
        rows.append([s, f.name.split("_")[1][:4], n, n_auto, n_auto_nil])
    by = {}
    for s, y, n, a, nil in rows:
        t = by.setdefault(s, [0, 0, 0, {}])
        t[0] += n; t[1] += a; t[2] += nil
        t[3][y] = round(a / n, 3) if n else None
    AUTO.parent.mkdir(exist_ok=True)
    with AUTO.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["icao", "night_reports", "auto_share", "auto_nil_cloud_share"]
                   + [f"auto_{y}" for y in range(FIRST, LAST + 1)])
        for s, (n, a, nil, yr) in sorted(by.items()):
            w.writerow([s, n, round(a / n, 4) if n else "", round(nil / a, 4) if a else ""]
                       + [yr.get(str(y), "") for y in range(FIRST, LAST + 1)])
    print(f"{len(by)} airports -> {AUTO}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", action="store_true")
    g.add_argument("--download", action="store_true")
    g.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    (pairs if a.pairs else download if a.download else auto)()


if __name__ == "__main__":
    main()
