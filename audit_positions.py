"""audit_positions.py — whether each archive month of each photometer was recorded at the position the pipeline uses.

Every stage places a photometer at one position (data/tess_archive/selected.csv, the latest header position with
data/station_fixes.csv applied): the ERA5-Land cell, the satellite pixels and the Sun and Moon all come from it.
A photometer that moved, or a month whose header gives no position, is then matched to the wrong place. Each
monthly archive file (and each part of a split month) is classed by its header position and MAC tail:
  same               within MOVE_KM of the position used (header typos corrected in station_fixes.csv count as same);
  other              further than MOVE_KM (the photometer was elsewhere that month);
  none_same_device   no position, but the same device states this position in another month (kept: a gap in the
                     registration, not a move);
  none_other_device  no position, and the device never states this position under this identifier.
Months with a file classed other or none_other_device are excluded.

Placeholder positions: several identifiers at the very same coordinates that do not see the same sky. PLACEHOLDERS
lists them, with the evidence (from the night files, no cloud reference): the seven TESS4C photometers filed at
"The Hebrew University of Jerusalem" (31.7758334, 35.2; no data supplier, no elevation) differ by up to 2.65 mag in
median dark-sky brightness and their sky temperatures correlate at 0.54-0.89, against 0.97-1.00 for photometers
that really share a site (stars75/76, stars40/41, stars312/313, stars2/489/717). They are excluded altogether.

Reads the raw archive headers, so it is not a verify target. Writes
results/audit_positions.csv (id, file, month, mac, lat, lon, km, status) and data/excluded_positions.csv (id,
month or "all", reason), read by build_nights.py and sat_score.py.

Usage: python audit_positions.py"""
from __future__ import annotations

import collections
import csv
import math
from pathlib import Path

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
SELECTED = ARCHIVE / "selected.csv"
FIXES = ROOT / "data" / "station_fixes.csv"
OUT = ROOT / "results" / "audit_positions.csv"
EXCLUDED = ROOT / "data" / "excluded_positions.csv"
MOVE_KM = 5.0
PLACEHOLDERS = {
    "stars705": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars706": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars707": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars854": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars855": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars856": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
    "stars857": "placeholder position shared by 7 TESS4C photometers (Hebrew University of Jerusalem) whose skies differ",
}


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    p = math.radians
    x = math.sin(p(a[0])) * math.sin(p(b[0])) + math.cos(p(a[0])) * math.cos(p(b[0])) * math.cos(p(a[1] - b[1]))
    return 6371 * math.acos(max(-1.0, min(1.0, x)))


def header(f: Path) -> tuple[tuple[float, float] | None, str | None]:
    pos = mac = None
    with f.open(errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("# Position:"):
                p = [x.strip() for x in line.split(":", 1)[1].split(",")]
                try:
                    pos = (float(p[0]), float(p[1]))
                except (ValueError, IndexError):
                    pos = None
            elif line.startswith("# TESS MAC address:"):
                mac = line.split(":", 1)[1].strip().upper()[-8:] or None
    return pos, mac


def main():
    used = {r["id"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(SELECTED.open())}
    fixed = {r["id"] for r in csv.DictReader(l for l in FIXES.open() if not l.startswith("#"))}
    rows = []
    for uid, ref in used.items():
        for f in sorted((ARCHIVE / uid).glob("*.dat")):
            pos, mac = header(f)
            month = f.stem.split("_")[1]
            if pos is None:
                rows.append([uid, f.name, month, mac, "", "", "", "none"])
                continue
            d = km(pos, ref) if abs(pos[0]) <= 90 else math.inf
            same = d <= MOVE_KM or uid in fixed
            rows.append([uid, f.name, month, mac, pos[0], pos[1], round(d, 1) if math.isfinite(d) else "", "same" if same else "other"])
    stated = collections.defaultdict(set)
    for r in rows:
        if r[7] == "same":
            stated[r[0]].add(r[3])
    for r in rows:
        if r[7] == "none":
            r[7] = "none_same_device" if r[3] in stated[r[0]] else "none_other_device"
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "file", "month", "mac", "lat", "lon", "km", "status"])
        w.writerows(rows)
    bad = collections.defaultdict(set)
    for r in rows:
        if r[7] in ("other", "none_other_device") and r[0] not in PLACEHOLDERS:
            bad[(r[0], r[2])].add(r[7])
    with EXCLUDED.open("w", newline="") as fh:
        fh.write("# Written by audit_positions.py: archive months recorded away from the position used, and photometers with\n"
                 "# a placeholder position. Skipped by build_nights.py (and the 'all' rows by qc_ir.excluded_ids, read by the baseline check and the satellite scoring).\n")
        w = csv.writer(fh)
        w.writerow(["id", "month", "reason"])
        for uid, why in sorted(PLACEHOLDERS.items()):
            w.writerow([uid, "all", why])
        for (uid, month), st in sorted(bad.items()):
            why = "header position more than 5 km from the position used" if "other" in st else \
                  "no header position, and the device never states this position"
            w.writerow([uid, month, why])
    c = collections.Counter(r[7] for r in rows)
    ids = collections.Counter(u for u, _ in bad)
    print(f"{len(rows)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print(f"excluded: {len(PLACEHOLDERS)} placeholder photometers; {len(bad)} months at {len(ids)} photometers: "
          + ", ".join(f"{u} {n}" for u, n in ids.most_common()))


if __name__ == "__main__":
    main()
