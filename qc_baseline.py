"""qc_baseline.py — photometers whose clear-sky baseline is too loose for the spectral test to see cloud.

The spectral test calls cloud when the sky reads warmer than the ERA5 baseline by more than kz held-out robust SDs
of the photometer's clear-sky residual (sd_brunt, features.py). Where that SD is as large as the warming a cloud
causes (5-20 C), the test has no power, whatever the threshold. The cut uses no cloud reference: a photometer is
flagged when its sd_brunt lies beyond the upper Tukey fence of the network distribution (Q3 + 1.5 IQR over every
photometer with a baseline that the sensor and position checks keep, qc_ir.excluded_ids). That set, with_baseline(),
is the photometers the cloud record covers; the satellite scoring and the tables read it.

This flag was added after the first satellite scoring (Sect. 3.5 of the paper), where the pattern was seen, so
scores are reported with and without the flagged photometers.

Writes results/qc_baseline.csv (id, sd_brunt, fence, weak), one row per photometer of the record.

Usage: python qc_baseline.py"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from qc_ir import excluded_ids

ROOT = Path(__file__).parent
UNITS = ROOT / "results" / "features_units.csv"
OUT = ROOT / "results" / "qc_baseline.csv"
FENCE_K = 1.5


def weak() -> set[str]:
    """Identifiers flagged by the last run (read by the scoring stages)."""
    if not OUT.exists():
        return set()
    return {r["id"] for r in csv.DictReader(OUT.open()) if r["weak"] == "1"}


def with_baseline() -> set[str]:
    """The photometers of the record: a clear-sky baseline was fitted (features.py) and no check excludes them."""
    f = pd.read_csv(UNITS)[["id", "sd_brunt"]].dropna()
    return set(f["id"]) - excluded_ids()


def main():
    f = pd.read_csv(UNITS)[["id", "sd_brunt"]].dropna()
    f = f[f["id"].isin(with_baseline())]
    q1, q3 = f["sd_brunt"].quantile([0.25, 0.75])
    fence = q3 + FENCE_K * (q3 - q1)
    f["fence"] = round(fence, 6)
    f["weak"] = (f["sd_brunt"] > fence).astype(int)
    f.to_csv(OUT, index=False)
    flagged = f.loc[f["weak"] == 1, "id"].tolist()
    print(f"{len(f)} photometers with a baseline and no exclusion; fence {fence:.2f} C; weak {len(flagged)}: {', '.join(flagged)}")


if __name__ == "__main__":
    main()
