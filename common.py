"""common.py — helpers shared by the scoring stages: the night key and the two bootstraps.

Night key: a night is the local solar date at its start (UTC shifted by the photometer's longitude at 15 degrees
per hour, minus 12 hours), as in nsb_selection.py. UTC noon to noon would split 15 % of the local nights (13
Australian and New Zealand photometers on almost every night).

Night-block bootstrap: whole nights resampled within each photometer (each keeps its number of nights), B draws.
It says how noisy a score is at these photometers and conditions on the set of photometers and airports sampled.

Cluster bootstrap: whole clusters resampled with replacement (airports for the ceilometer scores, photometers for
the satellite scores), B draws. It gives the between-site uncertainty that applies to a site not in the sample.
Both are 95 % percentile intervals of the balanced accuracy (or the hit rate) recomputed from summed counts.

Paired cluster bootstrap: the difference of two methods' pooled balanced
accuracies, with both methods recomputed on the same resampled clusters in each draw."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
B = 1000


def longitudes() -> dict[str, float]:
    """Longitude of every photometer of the selection (the position the pipeline uses)."""
    return {r["id"]: float(r["lon"]) for r in csv.DictReader(SELECTED.open())}


def local_night(time: pd.Series, lon: float) -> pd.Series:
    """The local solar date at the start of the night, as a string key."""
    return (time + pd.to_timedelta(lon / 15.0, unit="h") - pd.Timedelta(hours=12)).dt.date.astype(str)


def counts_by(y: np.ndarray, c: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per group (sorted unique): hits, cloudy, false alarms, clear."""
    blk, inv = np.unique(group, return_inverse=True)
    k = np.zeros((len(blk), 4))
    np.add.at(k, (inv, 0), c & y)
    np.add.at(k, (inv, 1), y)
    np.add.at(k, (inv, 2), c & ~y)
    np.add.at(k, (inv, 3), ~y)
    return blk, k


def _stat(s: np.ndarray, stat: str) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "hit":
            return s[0] / s[1]
        if stat == "fa":
            return s[2] / s[3]
        return (s[0] / s[1] + 1 - s[2] / s[3]) / 2


def cluster_ci(y: np.ndarray, c: np.ndarray, cluster: np.ndarray, rng: np.random.Generator, b: int = B,
               stat: str = "ba") -> tuple[float, float]:
    """95 % interval resampling whole clusters with replacement."""
    _, k = counts_by(y, c, np.asarray(cluster))
    n = len(k)
    bs = np.array([_stat(k[rng.integers(0, n, n)].sum(axis=0), stat) for _ in range(b)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def night_block_ci(y: np.ndarray, c: np.ndarray, uid: np.ndarray, night: np.ndarray, rng: np.random.Generator,
                   b: int = B, stat: str = "ba") -> tuple[float, float]:
    """95 % interval resampling whole nights within each photometer."""
    uid = np.asarray(uid).astype(str)
    key = np.char.add(np.char.add(uid, "|"), np.asarray(night).astype(str))
    blk, k = counts_by(y, c, key)
    unit = np.array([x.split("|", 1)[0] for x in blk])
    order = np.argsort(unit, kind="stable")
    _, start, size = np.unique(unit[order], return_index=True, return_counts=True)
    s0, n0 = np.repeat(start, size), np.repeat(size, size)
    k = k[order]
    bs = np.array([_stat(k[s0 + (rng.random(len(blk)) * n0).astype(int)].sum(axis=0), stat) for _ in range(b)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def paired_cluster_ci(y: np.ndarray, c1: np.ndarray, c2: np.ndarray, cluster: np.ndarray, rng: np.random.Generator,
                      b: int = B) -> tuple[float, float, float]:
    """Pooled balanced accuracy of c1 minus that of c2, with a 95 % interval from one set of cluster draws."""
    _, k1 = counts_by(y, c1, np.asarray(cluster))
    _, k2 = counts_by(y, c2, np.asarray(cluster))
    n = len(k1)
    diff = _stat(k1.sum(axis=0), "ba") - _stat(k2.sum(axis=0), "ba")
    bs = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        bs[i] = _stat(k1[idx].sum(axis=0), "ba") - _stat(k2[idx].sum(axis=0), "ba")
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(diff), float(lo), float(hi)
