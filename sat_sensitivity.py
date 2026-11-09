"""sat_sensitivity.py — how much the satellite scores depend on the fixed matching rules of sat_score.py.

The primary rows of sat_score.py --dem (snow-free, not mountain on the terrain model) are rescored with one rule
changed at a time; the frozen classifier and thresholds are unchanged:
  box rule      a box is cloudy at 5 of 9 pixels (30 of 49 for VIIRS) or at all of them, instead of 7 of 9 (40 of
                49); relabelled from the stored cloudy fraction of each matched box
  window        photometer bins within 2.5 or 10 min of the scan, instead of 5 min; the bins are matched again
                (sat_score.match with MATCH changed inside each worker)
  view angle    GOES and Meteosat boxes viewed below 60 deg from nadir, instead of 70 deg; the stored matches are
                kept when the raw box (data/goes, data/msg) passes the stricter limit
Each variant gives the two-test balanced accuracy, hit and false-alarm rates on the clear and cloudy boxes, with a
95 % photometer-cluster interval (common.cluster_ci, B draws, seeded), beside the primary score on the same rules.

Writes results/sat_sensitivity.csv and results/sat_sensitivity.txt.

Usage: python sat_sensitivity.py [--workers 12]"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import sat_score as ss
from common import cluster_ci
from qc_baseline import with_baseline

ROOT = Path(__file__).parent
OUT_CSV = ROOT / "results" / "sat_sensitivity.csv"
OUT_TXT = ROOT / "results" / "sat_sensitivity.txt"
SEED, B = 20261106, 1000
BOX = {"goes": (9, (5, 9)), "msg": (9, (5, 9)), "viirs": (49, (30, 49))}
WINDOWS = (2.5, 10.0)
VZA = 60.0


def mountain_ids() -> set[str]:
    dem = pd.read_csv(ROOT / "data" / "elevation.csv")
    return ss.MOUNTAIN | {u for u, z in zip(dem["id"], dem["dem_m"].fillna(0.0)) if z >= ss.MOUNTAIN_M}


def primary(m: pd.DataFrame, mountain: set[str]) -> pd.DataFrame:
    return m[m["sde_m"].notna() & (m["sde_m"] <= ss.SNOW_M) & ~m["id"].isin(mountain)]


def relabel(frac: pd.Series, n: int, k_cloudy: int) -> np.ndarray:
    k = np.rint(frac.to_numpy(float) * n)
    return np.where(k == 0, "clear", np.where(k >= k_cloudy, "cloudy", "partial"))


def score(g: pd.DataFrame, rng: np.random.Generator) -> list:
    g = g[g["label"].isin(["clear", "cloudy"])]
    y = (g["label"] == "cloudy").to_numpy()
    c = g["call_two_test"].to_numpy(bool)
    ba, hit, fa = ss.rates(y, c)
    lo, hi = cluster_ci(y, c, g["id"].to_numpy(), rng, B)
    return [g["id"].nunique(), len(g), round(ba, 6), round(lo, 6), round(hi, 6), round(hit, 6), round(fa, 6)]


def rematch(args) -> pd.DataFrame:
    uid, ref, th, minutes = args
    ss.MATCH = pd.Timedelta(minutes=minutes)
    return ss.match((uid, ref, th))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    th = ss.thresholds()
    mountain = mountain_ids()
    ids = sorted(with_baseline() & {f.name.split(".")[0] for f in ss.FEAT.glob("*.csv.gz")})
    rows = []
    for ref in ss.REFS:
        m = primary(pd.read_csv(ss.SAT / f"{ref}.csv.gz", parse_dates=["time"]), mountain)
        rows.append([ref, "primary (7 of 9, 40 of 49; 5 min; 70 deg)", *score(m, rng)])
        n, ks = BOX[ref]
        for k in ks:
            rows.append([ref, f"box cloudy at {k} of {n}", *score(m.assign(label=relabel(m["frac"], n, k)), rng)])
        for w in WINDOWS:
            with ProcessPoolExecutor(a.workers) as ex:
                parts = [x for x in ex.map(rematch, [(u, ref, th, w) for u in ids]) if len(x)]
            r = primary(pd.concat(parts, ignore_index=True), mountain)
            rows.append([ref, f"window {w:g} min", *score(r, rng)])
        if ref in ("goes", "msg"):
            ss.MAX_VZA = VZA
            keep = ss.load_ref(ref)[["id", "time"]]
            ss.MAX_VZA = 70.0
            r = m.merge(keep, on=["id", "time"], how="inner")
            rows.append([ref, f"view angle below {VZA:g} deg", *score(r, rng)])
        print(ref, "done", flush=True)
    t = pd.DataFrame(rows, columns=["reference", "variant", "photometers", "boxes", "balanced_accuracy", "ci_lo_phot",
                                    "ci_hi_phot", "hit", "false_alarm"])
    t.to_csv(OUT_CSV, index=False)
    lines = ["Two-test classifier on snow-free lowland boxes (terrain-model mountain rule), one matching rule changed at "
             "a time; balanced accuracy (95% photometer-cluster interval), hit, false alarm"]
    for r in t.itertuples():
        lines.append(f"  {r.reference:6s} {r.variant:44s} photometers {r.photometers:4d} boxes {r.boxes:8d}  "
                     f"{r.balanced_accuracy:.3f} ({r.ci_lo_phot:.3f}-{r.ci_hi_phot:.3f})  hit {r.hit:.3f}  "
                     f"false alarm {r.false_alarm:.3f}")
    rng_note = "ranges over variants: " + "; ".join(
        f"{ref} {g['balanced_accuracy'].min():.3f}-{g['balanced_accuracy'].max():.3f}" for ref, g in t.groupby("reference", sort=False))
    lines.append(rng_note)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
