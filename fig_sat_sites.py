"""fig_sat_sites.py — the classifier's balanced accuracy per photometer against each satellite mask.

From results/sat_score_sites_dem.csv (sat_score.py --dem: snow-free lowland scans, photometers with at least 100 clear
and 100 cloudy scans) and the weak-baseline flag of results/qc_baseline.csv. One row per satellite: every photometer
as a point (jittered vertically by a seeded draw, so that points do not hide each other), the weak-baseline
photometers marked apart, and the median of the others; the photometers per row in the axis label.

Writes figures/fig_sat_sites.png and results/fig_sat_sites.txt (the values plotted).

Usage: python fig_sat_sites.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

ROOT = Path(__file__).parent
SITES = ROOT / "results" / "sat_score_sites_dem.csv"
QC = ROOT / "results" / "qc_baseline.csv"
OUT_PNG = ROOT / "figures" / "fig_sat_sites.png"
OUT_TXT = ROOT / "results" / "fig_sat_sites.txt"
REFS = [("goes", "GOES ABI"), ("msg", "Meteosat"), ("viirs", "VIIRS")]
SEED = 20261103


def main():
    s = pd.read_csv(SITES)
    q = pd.read_csv(QC)
    weak = set(q.loc[q["weak"] == 1, "id"])
    rng = np.random.default_rng(SEED)
    plt.rcParams.update(fs.RC)
    fig, ax = plt.subplots(figsize=(fs.COL, 0.62 * fs.COL), layout="constrained")
    lines = ["per photometer two-test balanced accuracy (snow-free lowland, >= 100 clear and cloudy scans)"]
    for k, (ref, name) in enumerate(REFS):
        g = s[s["reference"] == ref].copy()
        g["weak"] = g["id"].isin(weak)
        y = len(REFS) - 1 - k
        jit = rng.uniform(-0.22, 0.22, len(g))
        ok, wk = ~g["weak"].to_numpy(), g["weak"].to_numpy()
        ax.scatter(g["two_test"][ok], y + jit[ok], s=7, facecolors="none", edgecolors=fs.G2, linewidths=0.5, zorder=3)
        ax.scatter(g["two_test"][wk], y + jit[wk], s=11, marker="x", color=fs.INK, linewidths=0.7, zorder=4)
        med = float(g.loc[ok, "two_test"].median())
        ax.plot([med, med], [y - 0.32, y + 0.32], color=fs.ACCENT, lw=1.6, zorder=5)
        ax.text(med, y + 0.36, f"median {med:.3f}", ha="center", va="bottom", fontsize=6, color=fs.INK)
        low = g[g["two_test"] < g["two_test"].quantile(0.1)]
        lines.append(f"{ref}: photometers {len(g)}, weak {int(wk.sum())}; median without weak {med:.3f}, "
                     f"median of weak {g.loc[wk, 'two_test'].median():.3f}, min {g['two_test'].min():.3f}; "
                     f"lowest tenth {len(low)}, of which weak {int(low['weak'].sum())}")
        lines += [f"  {r.id} {r.two_test:.4f}{' weak' if r.weak else ''}" for r in g.sort_values("two_test").itertuples()]
    counts = {ref: int((s["reference"] == ref).sum()) for ref, _ in REFS}
    ax.set_yticks(range(len(REFS)), [f"{n}\n({counts[r]})" for r, n in REFS][::-1])   # photometers per row
    ax.set_ylim(-0.6, len(REFS) - 0.25)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Balanced accuracy per photometer")
    ax.axvline(0.5, color=fs.G3, lw=0.6, zorder=1)
    ax.text(0.505, -0.55, "chance", fontsize=6, color=fs.MUTED, va="bottom")
    ax.scatter([], [], s=7, facecolors="none", edgecolors=fs.G2, linewidths=0.5, label="photometer")
    ax.scatter([], [], s=11, marker="x", color=fs.INK, linewidths=0.7, label="weak clear-sky baseline")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, borderaxespad=0, handletextpad=0.3)
    fs.style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG)                      # full figure width: 8.3 cm at 300 dpi (AMT minimum 8 cm)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:1] + [l for l in lines if not l.startswith("  ")][1:]))


if __name__ == "__main__":
    main()
