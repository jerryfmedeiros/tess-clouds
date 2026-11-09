"""audit_sequence.py — the message sequence number of every archive reading, as a check that one identifier's
readings come from one device at a time.

Each TESS-W numbers its messages from 0 at power-up and counts up by one per transmission (last column of the
archive files, "Multiple of Tx period"; column 8 of 8 in TESS-W files, 17 of 17 in TESS4C files). As suggested by
R. Gonzalez Fuentetaja (STARS4ALL), two devices publishing under the same name show as two sequences intermingled. In one identifier's readings, in time order, a backward step is either
  reset   the next value is below RESET_MAX, or the step follows a gap of more than GAP_MAX in the readings (a
          reboot, a power cut, or a new device taking over the name; photometers on a daily power timer restart
          during the day and their first counts are not in the archive), or
  jump    the next value is larger and follows within GAP_MAX (another counter running at the same time: a second
          device under the name).
A file (identifier, month, part) with at least JUMP_MIN jumps is flagged.

Reads the raw archive (data/tess_archive/<id>/*.dat), so it is not a verify target. Writes
results/audit_sequence.csv (id, file, readings, resets, jumps, flagged), results/audit_sequence_resets.csv
(id, time UTC, before, after: every reset, to compare with device-change dates) and
results/audit_sequence_jumps.csv (id, time UTC: every jump in a flagged file, so the days a second device ran can
be counted).

Usage: python audit_sequence.py [--workers 8]"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
OUT = ROOT / "results" / "audit_sequence.csv"
OUT_RESETS = ROOT / "results" / "audit_sequence_resets.csv"
OUT_JUMPS = ROOT / "results" / "audit_sequence_jumps.csv"
RESET_MAX = 120          # first counts after power-up (two hours of one message a minute)
GAP_MAX = pd.Timedelta(minutes=10)
JUMP_MIN = 3


def seq_column(f: Path) -> int:
    """The sequence number is the last field: 8 fields a line in TESS-W files, 17 in TESS4C (four channels)."""
    with f.open(errors="replace") as fh:
        for line in fh:
            if line.startswith("# Number of fields per line:"):
                return int(line.split(":")[1]) - 1
            if not line.startswith("#"):
                break
    return 7


def one(uid: str) -> tuple[list, list, list]:
    frames = []
    for f in sorted((ARCHIVE / uid).glob("*.dat")):
        try:
            d = pd.read_csv(f, sep=";", comment="#", header=None, usecols=[0, seq_column(f)], names=["time", "seq"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        d["seq"] = pd.to_numeric(d["seq"], errors="coerce")
        d["file"] = f.name
        frames.append(d.dropna())
    if not frames:
        return [], [], []
    d = pd.concat(frames)
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna().sort_values("time", kind="stable").reset_index(drop=True)
    s = d["seq"].to_numpy(np.int64)
    names = d["file"].to_numpy()
    back = np.flatnonzero(np.diff(s) < 0) + 1
    gap = (d["time"].diff() > GAP_MAX).to_numpy()
    restart = (s[back] < RESET_MAX) | gap[back]
    reset, jump = back[restart], back[~restart]
    rows = []
    for name, g in d.groupby("file", sort=True):
        idx = g.index
        nr = int(np.isin(reset, idx).sum())
        nj = int(np.isin(jump, idx).sum())
        rows.append([uid, name, len(g), nr, nj, int(nj >= JUMP_MIN)])
    resets = [[uid, d.at[i, "time"].isoformat(), int(s[i - 1]), int(s[i])] for i in reset]
    flagged = {r[1] for r in rows if r[5]}
    jumps = [[uid, d.at[i, "time"].isoformat()] for i in jump if names[i] in flagged] if flagged else []
    return rows, resets, jumps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = sorted(p.name for p in ARCHIVE.iterdir() if p.is_dir())
    rows, resets, jumps = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, z, j in ex.map(one, ids):
            rows += r
            resets += z
            jumps += j
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "file", "readings", "resets", "jumps", "flagged"])
        w.writerows(rows)
    with OUT_RESETS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "time", "before", "after"])
        w.writerows(resets)
    with OUT_JUMPS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "time"])
        w.writerows(jumps)
    t = pd.DataFrame(rows, columns=["id", "file", "readings", "resets", "jumps", "flagged"])
    f = t[t["flagged"] == 1]
    if jumps:
        j = pd.DataFrame(jumps, columns=["id", "time"])
        j["day"] = pd.to_datetime(j["time"]).dt.date
        print("days with a second counter, per flagged identifier: " + "; ".join(
            f"{u} {g['day'].nunique()} days ({g['day'].min()} to {g['day'].max()})" for u, g in j.groupby("id")))
    print(f"{t['id'].nunique()} identifiers, {len(t)} files, {int(t['readings'].sum())} readings; "
          f"{int(t['resets'].sum())} resets, {int(t['jumps'].sum())} jumps; flagged {len(f)} files at {f['id'].nunique()} identifiers")
    if len(f):
        print(f.groupby("id").agg(files=("file", "size"), jumps=("jumps", "sum"), readings=("readings", "sum"))
              .sort_values("jumps", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
