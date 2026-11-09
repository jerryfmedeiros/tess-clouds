"""sat_vs_metar.py — how far the satellite references agree with automated METAR, and how the classifier scores
where the two agree. Exploratory (after the freeze-satellite run): it changes no rule and no threshold; it asks how
much of the classifier's lower score against a satellite mask is the mask's own error.

Each automated METAR report of data/metar/scored.csv.gz (held-out classifier calls of classify_metar.py) is
matched to the rows of each satellite reference (data/sat/<ref>.csv.gz, from sat_score.py, so every frozen rule
on boxes, view angle and timing applies) whose time falls in the report's 30-minute window. Satellite state: cloudy
if any matched row is cloudy, clear if all are clear, otherwise partial. METAR state: clear (CLR, SKC, NCD) or
cloudy (BKN, OVC, VV). Reports with both states clear or cloudy, snow-free (ERA5-Land snow depth <= SNOW_M) and not at
a mountain site (sat_score.MOUNTAIN or 1500 m and above on the terrain model, data/elevation.csv, as sat_score.py
--dem and the paper's Table 4; the archive header elevations of the frozen rule would keep the two Jackson Hole
photometers, at 2125 and 1874 m, in the GOES row), are scored three ways on the same reports:
  satellite vs METAR       the satellite mask taken as a classifier of METAR
  classifier vs METAR      and classifier vs satellite
  classifier vs consensus  on the reports where METAR and satellite agree
Balanced accuracy with a 95% night-block bootstrap within photometers (B, SEED) and, in the columns ci_lo_phot and
ci_hi_phot, a 95% interval from a bootstrap that resamples whole photometers (common.cluster_ci).

Writes results/sat_vs_metar.csv and results/sat_vs_metar.txt.

Usage: python sat_vs_metar.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sat_score as ss
from common import cluster_ci

ROOT = Path(__file__).parent
OUT_CSV = ROOT / "results" / "sat_vs_metar.csv"
OUT_TXT = ROOT / "results" / "sat_vs_metar.txt"
SEED, B = 20261030, 1000
WINDOW = pd.Timedelta(minutes=30)
SNOW_M = 0.01


def boot(y: np.ndarray, c: np.ndarray, uid: np.ndarray, night: np.ndarray, rng) -> tuple[float, float, float, float, float, float, float]:
    key = pd.Series(uid).astype(str) + "|" + pd.Series(night).astype(str)
    blk, inv = np.unique(key.to_numpy(), return_inverse=True)
    unit = pd.Series(blk).str.split("|").str[0].to_numpy()
    order = np.argsort(unit, kind="stable")
    _, start, size = np.unique(unit[order], return_index=True, return_counts=True)
    s0, n0 = np.repeat(start, size), np.repeat(size, size)
    k = np.zeros((len(blk), 4))
    np.add.at(k, (inv, 0), c & y)
    np.add.at(k, (inv, 1), y)
    np.add.at(k, (inv, 2), c & ~y)
    np.add.at(k, (inv, 3), ~y)
    k = k[order]
    bs = np.empty(B)
    for b in range(B):
        s = k[s0 + (rng.random(len(blk)) * n0).astype(int)].sum(axis=0)
        bs[b] = (s[0] / s[1] + 1 - s[2] / s[3]) / 2
    ba, hit, fa = ss.rates(y, c)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    plo, phi = cluster_ci(y, c, uid, rng, B)
    return ba, lo, hi, plo, phi, hit, fa


def main():
    sc = pd.read_csv(ROOT / "data" / "metar" / "scored.csv.gz", parse_dates=["time"])
    dem = pd.read_csv(ROOT / "data" / "elevation.csv")
    mountain = ss.MOUNTAIN | {u for u, z in zip(dem["id"], dem["dem_m"].fillna(0.0)) if z >= ss.MOUNTAIN_M}
    sc = sc[sc["code"].isin(["CLR", "SKC", "NCD", "LAYERS"]) & sc["label"].isin(["clear", "cloudy"])
            & (sc["sde_m"] <= SNOW_M) & ~sc["id"].isin(mountain)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows, lines = [], ["Satellite references against automated METAR on the same reports (snow-free, no mountain sites on "
                       "the terrain model); exploratory, after the freeze; intervals: night-block within photometers [..], "
                       "photometer-cluster (..)"]
    for ref in ss.REFS:
        s = pd.read_csv(ss.SAT / f"{ref}.csv.gz", usecols=["id", "time", "label"], parse_dates=["time"])
        state = np.full(len(sc), "", dtype=object)
        for uid, h in sc.groupby("id"):
            g = s[s["id"] == uid].sort_values("time")
            if g.empty:
                continue
            t, lab = g["time"].to_numpy(), g["label"].to_numpy()
            lo = np.searchsorted(t, (h["time"] - WINDOW).to_numpy(), side="right")
            hi = np.searchsorted(t, h["time"].to_numpy(), side="right")
            for i, a, b in zip(h.index, lo, hi):
                if b > a:
                    L = lab[a:b]
                    state[i] = "cloudy" if (L == "cloudy").any() else "clear" if (L == "clear").all() else "partial"
        d = sc.assign(sat=state)
        d = d[d["sat"].isin(["clear", "cloudy"])].reset_index(drop=True)
        if d.empty:
            continue
        ym = (d["label"] == "cloudy").to_numpy()
        ys = (d["sat"] == "cloudy").to_numpy()
        cl = d["call_two_test"].to_numpy(bool)
        u, n = d["id"].to_numpy(), d["night"].to_numpy()
        agree = ym == ys
        lines.append(f"\n{ref}: {len(d)} reports at {d['id'].nunique()} photometers; METAR and satellite agree on "
                     f"{agree.mean():.3f} (METAR clear, satellite cloudy {(~ym & ys).mean():.3f}; METAR cloudy, "
                     f"satellite clear {(ym & ~ys).mean():.3f})")
        for name, y, c, sel in (("satellite vs METAR", ym, ys, slice(None)),
                                ("classifier vs METAR", ym, cl, slice(None)),
                                ("classifier vs satellite", ys, cl, slice(None)),
                                ("classifier vs consensus", ym, cl, agree)):
            r = boot(y[sel], c[sel], u[sel], n[sel], rng)
            rows.append([ref, name, int(np.size(y[sel])), int(y[sel].sum()), *[round(x, 6) for x in r]])
            lines.append(f"  {name:24s} n={np.size(y[sel]):7d}  {r[0]:.3f} [{r[1]:.3f}-{r[2]:.3f}] ({r[3]:.3f}-{r[4]:.3f})  "
                         f"hit {r[5]:.3f}  false alarm {r[6]:.3f}")
    pd.DataFrame(rows, columns=["reference", "comparison", "reports", "cloudy", "balanced_accuracy", "ci_lo", "ci_hi",
                                "ci_lo_phot", "ci_hi_phot", "hit", "false_alarm"]).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
