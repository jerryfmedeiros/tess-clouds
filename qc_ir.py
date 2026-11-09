"""qc_ir.py — photometers whose infrared sensor does not see the sky (obstructed, covered, or behind a window).

A sensor that sees the sky reads well below the air temperature when the sky is clear, even in humid tropical air
(water vapour in the 5.5-14 um band warms the clear sky, but the clearest moments still read 14-23 C below the
air at Reunion, Panama and Hanoi). A sensor that sees a cover reads close to the air, or warmer. The test uses no
cloud reference: over every astronomical-night bin (Sun below -18 deg) with an ERA5-Land air temperature, the
95th percentile of air minus sky (the clearest moments) must reach DEPRESSION_MIN; photometers below it are
flagged obstructed. Over the network the values fall into two groups with a gap from 6.1 to 10.5 C, and the
threshold sits in that gap.

Writes results/qc_ir.csv (id, bins, median and 95th percentile of air minus sky, sky 5th percentile, obstructed).
Photometers flagged obstructed are left out of every scoring stage (metar_labels.py and the satellite matches).

The same rule month by month: a sensor can see a cover for part of its
life and still pass over the whole record. Every calendar month (UTC) with at least MIN_MONTH_BINS astronomical-night
bins is tested against DEPRESSION_MIN; results/qc_ir_months.csv lists each tested month with its 95th percentile and
the flag. The monthly values have no gap at DEPRESSION_MIN as the whole-record values do; the flagged months are
checked against the satellite masks in sat_score.py --dem (no cloud reference enters the rule). Flagged months stay
in the primary scores; the record marks them (cloud_record.py) and sat_score.py --dem scores the primary rows
without them.

Usage: python qc_ir.py"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
NIGHTS = ROOT / "data" / "nights"
OUT = ROOT / "results" / "qc_ir.csv"
OUT_MONTHS = ROOT / "results" / "qc_ir_months.csv"
DEPRESSION_MIN = 8.0
MIN_BINS = 1000
MIN_MONTH_BINS = 500


def obstructed() -> set[str]:
    """Identifiers flagged by the last run (read by the scoring stages)."""
    if not OUT.exists():
        return set()
    return {r["id"] for r in csv.DictReader(OUT.open()) if r["obstructed"] == "1"}


def obstructed_months() -> set[tuple[str, str]]:
    """(id, YYYY-MM) of the months flagged by the last run."""
    if not OUT_MONTHS.exists():
        return set()
    return {(r["id"], r["month"]) for r in csv.DictReader(OUT_MONTHS.open()) if r["obstructed_month"] == "1"}


def excluded_ids() -> set[str]:
    """Photometers left out of the record: infrared never usable (data/excluded_ir.csv), obstructed field of view
    (this script) and placeholder positions (data/excluded_positions.csv, month "all"). Read by qc_baseline.py,
    whose with_baseline() is the set of photometers the record covers."""
    def column(name: str, only_all: bool) -> set[str]:
        rows = [l.split(",") for l in (ROOT / "data" / name).read_text().splitlines()
                if l.strip() and not l.startswith("#") and not l.startswith("id,")]
        return {r[0].strip() for r in rows if not only_all or r[1].strip() == "all"}
    return obstructed() | column("excluded_ir.csv", False) | column("excluded_positions.csv", True)


def main():
    rows, months = [], []
    for f in sorted(NIGHTS.glob("*.csv.gz")):
        d = pd.read_csv(f, usecols=["time", "sky_med", "t2m_C", "sun_alt"]).dropna()
        d = d[d["sun_alt"] < -18]
        if len(d) < MIN_BINS:
            continue
        dep = d["t2m_C"] - d["sky_med"]
        uid = f.name.split(".")[0]
        g = dep.groupby(d["time"].str[:7])
        for month, n, q in zip(g.size().index, g.size(), g.quantile(0.95)):
            if n >= MIN_MONTH_BINS:
                months.append([uid, month, int(n), round(q, 4), int(q < DEPRESSION_MIN)])
        p95 = dep.quantile(0.95)
        rows.append([f.name.split(".")[0], len(d), round(dep.median(), 2), round(p95, 2),
                     round(d["sky_med"].quantile(0.05), 2), int(p95 < DEPRESSION_MIN)])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "bins", "depression_median_C", "depression_p95_C", "sky_p05_C", "obstructed"])
        w.writerows(rows)
    with OUT_MONTHS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "month", "bins", "depression_p95_C", "obstructed_month"])
        w.writerows(months)
    flagged = [r[0] for r in rows if r[-1]]
    print(f"{len(rows)} photometers; obstructed {len(flagged)}: {', '.join(flagged)}")
    fm = [m for m in months if m[-1] and m[0] not in flagged]
    print(f"months tested {len(months)}; flagged {sum(m[-1] for m in months)}, of which {len(fm)} at "
          f"{len({m[0] for m in fm})} photometers not flagged over the whole record")


if __name__ == "__main__":
    main()
