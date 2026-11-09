"""audit_ir.py — audit the infrared columns of the TESS-W archive before any fitting.

Per photometer identifier, over every monthly file in data/tess_archive/<id>/, it counts:
  - readings, and the share with the no-read code (sky exactly 0.00 C) and the error code (-273.15 C);
  - the share of valid sky readings in the firmware floor band (-50, -49.7] C and the coldest valid reading
    (the first firmware drops sky readings at or below -50 C and repeats the last value);
  - the share at or above 49.9 C (the matching upper cut);
  - the longest run of identical consecutive sky values (stuck sensor or held value), in readings;
  - at night (Sun below -18 deg): the median and 5th percentile of the sky temperature, the median die
    ("Enclosure") temperature, and the share of night readings with the die between 4 and 12 C, where the
    heater holds it on cold nights.
Positions come from data/tess_archive/selected.csv (archive scan, with data/station_fixes.csv applied),
falling back to each file's "# Position:" header; the Sun's altitude from a standard low-precision
formula (accurate to a few hundredths of a degree, enough for a -18 deg cut).

Writes results/audit_ir.csv (one row per identifier) and results/audit_ir.txt (network summary).
Reads the raw archive, so it is not a verify target. Standard library only.

Usage: python3 audit_ir.py"""
from __future__ import annotations

import csv
import datetime as dt
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
OUT_CSV = ROOT / "results" / "audit_ir.csv"
OUT_TXT = ROOT / "results" / "audit_ir.txt"

NO_READ, ERROR = 0.0, -273.15
FLOOR = (-50.0, -49.7)
CEIL = 49.9
NIGHT_SUN_ALT = -18.0
HEATER_BAND = (4.0, 12.0)
POS = re.compile(r"# Position:\s*([-\d.]+),\s*([-\d.]+)")
SELECTED = ARCHIVE / "selected.csv"


def sun_altitude(t: dt.datetime, lat: float, lon: float) -> float:
    """Solar altitude in degrees (low-precision ephemeris; t in UTC)."""
    d = (t - dt.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    lam = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))


def pct(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else float("nan")


def audit_unit(folder: Path, known: tuple[float, float] | None) -> dict:
    n = n_noread = n_err = n_floor = n_ceil = 0
    coldest = math.inf
    run = best_run = 0
    run_value = float("nan")
    prev = None
    night_sky, night_die = [], []
    lat, lon = known if known else (None, None)
    for f in sorted(folder.glob("*.dat")):
        for line in f.open(errors="ignore"):
            if line.startswith("#"):
                m = POS.match(line)
                if m and not known:
                    lat, lon = float(m.group(1)), float(m.group(2))
                continue
            p = line.split(";")
            if len(p) < 5:
                continue
            try:
                t = dt.datetime.fromisoformat(p[0][:19])
                die, sky = float(p[2]), float(p[3])
            except ValueError:
                continue
            n += 1
            if sky == NO_READ:
                n_noread += 1
                continue
            if sky == ERROR or die == ERROR:
                n_err += 1
                continue
            if FLOOR[0] < sky <= FLOOR[1]:
                n_floor += 1
            if sky >= CEIL:
                n_ceil += 1
            coldest = min(coldest, sky)
            run = run + 1 if sky == prev else 1
            if run > best_run:
                best_run, run_value = run, sky
            prev = sky
            if lat is not None and sun_altitude(t, lat, lon) < NIGHT_SUN_ALT:
                night_sky.append(sky)
                night_die.append(die)
    valid = n - n_noread - n_err
    share = lambda k, of: round(k / of, 5) if of else float("nan")
    return {
        "id": folder.name, "lat": lat, "lon": lon, "readings": n,
        "noread_share": share(n_noread, n), "error_share": share(n_err, n),
        "floor_share": share(n_floor, valid), "ceil_share": share(n_ceil, valid),
        "coldest_valid": coldest if valid else float("nan"), "longest_identical_run": best_run, "longest_run_value": run_value,
        "night_readings": len(night_sky),
        "night_sky_median": round(st.median(night_sky), 2) if night_sky else float("nan"),
        "night_sky_p05": round(pct(night_sky, 5), 2),
        "night_die_median": round(st.median(night_die), 2) if night_die else float("nan"),
        "night_die_heater_band_share": share(sum(HEATER_BAND[0] <= x <= HEATER_BAND[1] for x in night_die),
                                             len(night_die)),
    }


def main():
    known = {}
    if SELECTED.exists():
        known = {r["id"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(SELECTED.open())}
    rows = [audit_unit(d, known.get(d.name)) for d in sorted(p for p in ARCHIVE.iterdir() if p.is_dir())]
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["readings"]]
    tot = sum(r["readings"] for r in ok)
    lines = [
        f"identifiers audited: {len(ok)}; readings: {tot}",
        f"no-read share (sky 0.00 C), network: {sum(r['noread_share'] * r['readings'] for r in ok) / tot:.4f}",
        f"error share (-273.15 C), network: {sum(r['error_share'] * r['readings'] for r in ok) / tot:.4f}",
        f"identifiers with any valid sky reading below -50.0 C: {sum(r['coldest_valid'] < FLOOR[0] for r in ok)}",
        f"identifiers with floor share > 1%: {sum(r['floor_share'] > 0.01 for r in ok)}",
        f"identifiers whose IR never reads (no-read share > 99%): {sum(r['noread_share'] > 0.99 for r in ok)}",
        f"identifiers with an identical-value run over 600 readings: {sum(r['longest_identical_run'] > 600 for r in ok)}",
        f"median of per-identifier night die median: {st.median(r['night_die_median'] for r in ok if r['night_readings']):.2f} C",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
