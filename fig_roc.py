"""fig_roc.py — ROC curves of the spectral scores against automated METAR (the baseline residual, die minus sky, the
raw sky temperature), with the operating points of the two-test classifier, the network dashboard rule and the
Alarcon et al. 2021 cut. The variability and brightness scores are in Table 3 only.

Reports and scores from data/metar/scored.csv.gz (classify_metar.py): clear (CLR, SKC, NCD) against cloudy (BKN, OVC,
VV), the reports classify_metar.py scores (NSC and CAVOK, which do not say the sky is clear, are left out); a report
mostly at the firmware floor has every spectral score at -inf, as in classify_metar.py. A curve is drawn for each continuous score (no threshold needed); AUC by classify_metar.auc, so the values match
results/classify_metar.txt. Operating points: hit and false-alarm rates of the held-out calls (call_*).

Writes figures/fig_roc.png and results/fig_roc.txt (AUC and the operating points plotted).

Usage: python fig_roc.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs
from classify_metar import SPECTRAL, auc

ROOT = Path(__file__).parent
SCORED = ROOT / "data" / "metar" / "scored.csv.gz"
OUT_PNG = ROOT / "figures" / "fig_roc.png"
OUT_TXT = ROOT / "results" / "fig_roc.txt"
CURVES = ("z_brunt", "neg_dT", "sky_med")             # the spectral scores; the others' AUCs are in Table 3
POINTS = {"two_test": "Two-test classifier", "dashboard": "Dashboard rule", "alarcon": "Alarcon et al. (2021)"}


def roc(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-alarm and hit rates for every threshold (score above the threshold = cloudy)."""
    ok = np.isfinite(s) | np.isneginf(s)
    y, s = y[ok], s[ok]
    o = np.argsort(-s, kind="stable")
    y, s = y[o], s[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]       # end of each run of equal scores
    tp, fp = np.cumsum(y)[last], np.cumsum(~y)[last]
    return np.r_[0, fp / (~y).sum()], np.r_[0, tp / y.sum()]


def main():
    d = pd.read_csv(SCORED)
    d = d[d["label"].isin(["clear", "cloudy"]) & d["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])].copy()   # as scored
    d.loc[d["floor"] >= 0.5, list(SPECTRAL)] = -np.inf
    y = (d["label"] == "cloudy").to_numpy()
    lines = [f"reports {len(d)} (cloudy {int(y.sum())}, clear {int((~y).sum())})", "AUC"]
    plt.rcParams.update(fs.RC)
    fig, ax = plt.subplots(figsize=(fs.COL, 0.92 * fs.COL), layout="constrained")
    ax.plot([0, 1], [0, 1], color=fs.GRID, lw=0.8, zorder=1)
    ax.text(0.60, 0.555, "chance", fontsize=6, color=fs.MUTED, ha="left", va="top", zorder=2)   # labelled in the figure (AMT)
    for m in CURVES:
        s = d[m].to_numpy(float)
        fa, hit = roc(y, s)
        a = auc(y, s)
        label, colour, ls, _ = fs.METHODS[m]
        name = {"z_brunt": "Baseline residual", "neg_dT": "Die minus sky", "sky_med": "Sky temperature"}[m]
        ax.plot(fa, hit, color=colour, ls=ls, lw=1.0, label=f"{name} (AUC {a:.3f})", zorder=3)
        lines.append(f"  {m:10s} {a:.3f}")
    lines.append("operating points (false alarm, hit)")
    style = {"two_test": dict(marker="o", ms=5, color=fs.ACCENT, mfc=fs.ACCENT),
             "dashboard": dict(marker="D", ms=4, color=fs.INK, mfc="white"),
             "alarcon": dict(marker="s", ms=4, color=fs.INK, mfc="white")}
    place = {"two_test": (0.30, 0.70, "left"), "dashboard": (0.22, 0.52, "left"), "alarcon": (0.97, 0.86, "right")}
    for m, label in POINTS.items():
        c = d[f"call_{m}"].astype(bool).to_numpy()
        hit, fa = c[y].mean(), c[~y].mean()
        ax.plot(fa, hit, ls="none", mew=0.8, zorder=5, clip_on=False, **style[m])
        tx, ty, ha = place[m]                       # labels in the empty space below the curves, with a leader
        ax.annotate(label, (fa, hit), (tx, ty), ha=ha, va="center", fontsize=6.5, color=fs.INK,
                    arrowprops=dict(arrowstyle="-", color=fs.G2, lw=0.5, shrinkA=1.5, shrinkB=3.5))
        lines.append(f"  {m:10s} {fa:.3f} {hit:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False-alarm rate (clear reports called cloudy)")
    ax.set_ylabel("Hit rate (cloudy reports called cloudy)")
    fs.style(ax, grid="both")
    ax.legend(loc="lower right", frameon=False, handlelength=2.8, borderaxespad=0.3)
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG)                      # full figure width: 8.3 cm at 300 dpi (AMT minimum 8 cm)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
