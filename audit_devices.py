"""audit_devices.py — which hardware and which position stand behind each photometer identifier, month by month.

The infrared sensor belongs to the device (MAC address), not to the identifier: a repaired photometer keeps its
name and gets new hardware, a device can move between identifiers, and pool devices run under loaner identifiers
at UCM before installation. Each monthly archive file header gives the MAC and the position; consecutive months
with the same MAC tail (last 8 characters, upper case) and a position within MOVE_KM form one segment.

Cross-matched with the STARS4ALL (name, MAC) history lists provided by R. Gonzalez Fuentetaja (archive database
snapshot 2026-03-11; not redistributed, available from STARS4ALL on request; read from data/stars4all_histories/
when present, otherwise the category column is left empty): category easy (never repaired or renamed),
repaired (name kept, MAC changed), renamed (MAC kept, name changed), complicated (both), or absent (registered
after the snapshot). LOANER lists the UCM pool identifiers.

Standard library only. Reads the raw archive, so it is not a verify target. Writes results/audit_devices.csv
(one row per segment: id, segment, mac, lat, lon, first, last, months) and results/audit_devices_ids.csv (per
identifier: segments, devices, positions, STARS4ALL category, loaner flag).

Usage: python audit_devices.py"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
HIST = ROOT / "data" / "stars4all_histories"
OUT_SEG = ROOT / "results" / "audit_devices.csv"
OUT_IDS = ROOT / "results" / "audit_devices_ids.csv"
MOVE_KM = 1.0
LOANER = {"stars381", "stars395", "stars410", *(f"stars{i}" for i in range(421, 435)),
          *(f"stars{i}" for i in range(502, 507)), "stars682", "stars683", "stars712"}


def km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def header(f: Path) -> tuple[str | None, tuple[float, float] | None]:
    mac = pos = None
    with f.open(errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "MAC address" in line:
                mac = line.split(":", 1)[1].strip().upper()[-8:] or None
            elif line.startswith("# Position:"):
                p = re.findall(r"-?\d+\.\d+", line)
                pos = (float(p[0]), float(p[1])) if len(p) >= 2 else None
    return mac, pos


def categories() -> tuple[dict, set, set]:
    if not (HIST / "rafael_entries.csv").exists():
        return {}, set(), set()
    cat = {}
    for r in csv.DictReader((HIST / "rafael_entries.csv").open()):
        cat.setdefault(r["name"].strip(), r["cat"])
    rx = re.compile(r"\] (\S+) => \d+ entries")
    repaired = {m.group(1) for m in map(rx.search, (HIST / "repaired.log").read_text(errors="replace").splitlines()) if m}
    renamed = {m.group(1).upper()[-8:] for m in map(rx.search, (HIST / "renamed.log").read_text(errors="replace").splitlines()) if m}
    return cat, repaired, renamed


def main():
    cat, repaired, renamed = categories()
    selected = [r["id"] for r in csv.DictReader((ARCHIVE / "selected.csv").open())]
    segs, ids = [], []
    for uid in selected:
        months = []
        for f in sorted((ARCHIVE / uid).glob("*.dat")):
            m = re.search(r"_(\d{4}-\d{2})\.dat$", f.name)
            if m:
                months.append((m.group(1), *header(f)))
        cur = None
        mine = []
        for ym, mac, pos in months:
            if cur and mac == cur["mac"] and (pos is None or cur["pos"] is None or km(pos, cur["pos"]) <= MOVE_KM):
                cur["last"], cur["months"] = ym, cur["months"] + 1
                cur["pos"] = cur["pos"] or pos
            else:
                cur = {"id": uid, "segment": len(mine) + 1, "mac": mac, "pos": pos, "first": ym, "last": ym, "months": 1}
                mine.append(cur)
        segs += mine
        c = "repaired" if uid in repaired else cat.get(uid, "absent" if cat else "")
        if c == "easy" and any(s["mac"] in renamed for s in mine):
            c = "renamed"
        ids.append([uid, len(mine), len({s["mac"] for s in mine}), len({s["pos"] for s in mine if s["pos"]}), c,
                    int(uid in LOANER), max((s["months"] for s in mine), default=0)])
    OUT_SEG.parent.mkdir(exist_ok=True)
    with OUT_SEG.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "segment", "mac", "lat", "lon", "first", "last", "months"])
        for s in segs:
            p = s["pos"] or ("", "")
            w.writerow([s["id"], s["segment"], s["mac"] or "", p[0], p[1], s["first"], s["last"], s["months"]])
    with OUT_IDS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "segments", "devices", "positions", "stars4all_category", "loaner", "longest_segment_months"])
        w.writerows(ids)
    n = len(ids)
    print(f"identifiers {n}; one segment {sum(r[1] == 1 for r in ids)}; >1 device {sum(r[2] > 1 for r in ids)}; "
          f">1 position {sum(r[3] > 1 for r in ids)}; loaner {sum(r[5] for r in ids)}")
    for k in ("easy", "repaired", "renamed", "complicated", "absent"):
        print(f"  {k:12s} {sum(r[4] == k for r in ids)}")


if __name__ == "__main__":
    main()
