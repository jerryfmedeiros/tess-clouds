"""fig_breakdown.py — the classifier and three single-rule methods against automated METAR, by condition.

One figure, two panels, from results/breakdown_metar.csv (breakdown_metar.py; held-out calls, 95% airport-cluster
bootstrap intervals):
  (a) hit rate on cloudy reports by the ceilometer's lowest base height (ft, the METAR unit);
  (b) balanced accuracy by ERA5-Land air temperature (C).
Methods: the two-test classifier, the baseline test alone, die minus sky with a learned threshold, and the network
dashboard rule. The classes are ordered, so their points are joined. In (b) the two coldest classes, mostly one
airport, are shaded.

Writes figures/fig_conditions.png and results/fig_breakdown.txt (the values plotted).

Usage: python fig_breakdown.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

ROOT = Path(__file__).parent
SRC = ROOT / "results" / "breakdown_metar.csv"
OUT_TXT = ROOT / "results" / "fig_breakdown.txt"
OUT_PNG = ROOT / "figures" / "fig_conditions.png"
METHODS = ("two_test", "z_brunt", "neg_dT", "dashboard")
FIGS = {
    "cloud_base": ("cloud_base_ft", "Lowest cloud base, thousands of ft (km)", "Hit rate (cloudy reports called cloudy)",
                   ["< 2000", "2000-6500", "6500-12000", "12000-20000", ">= 20000"],
                   ["< 2\n(< 0.6)", "2–6.5\n(0.6–2.0)", "6.5–12\n(2.0–3.7)", "12–20\n(3.7–6.1)", "≥ 20\n(≥ 6.1)"]),
    "air_temp": ("air", "Air temperature (°C)", "Balanced accuracy",
                 ["< -10", "-10 to 0", "0 to 10", "10 to 20", "> 20"],
                 ["< −10", "−10 to 0", "0 to 10", "10 to 20", "> 20"]),
}


def main():
    b = pd.read_csv(SRC)
    plt.rcParams.update(fs.RC)
    lines = []
    fig, axes = plt.subplots(2, 1, figsize=(fs.COL, 1.38 * fs.COL), layout="constrained")
    for ax, (panel, (name, (split, xlabel, ylabel, levels, ticks))) in zip(axes, zip("ab", FIGS.items())):
        t = b[b["split"] == split]
        x = np.arange(len(levels))
        lines.append(f"({panel}) {name}: {split}, {ylabel.lower()} [95% airport-cluster interval], reports per class")
        for k, m in enumerate(METHODS):
            r = t[t["method"] == m].set_index("level").loc[levels]
            label, colour, ls, mk = fs.METHODS[m]
            dx = (k - 1.5) * 0.06                                   # small offsets keep the intervals apart
            ax.errorbar(x + dx, r["value"], yerr=[r["value"] - r["ci_lo_air"], r["ci_hi_air"] - r["value"]], color=colour,
                        ls=ls, marker=mk, ms=3.2 if m != "two_test" else 3.8, mfc="white" if m != "two_test" else colour,
                        mew=0.8, lw=1.3 if m == "two_test" else 0.9, elinewidth=0.6, capsize=0, label=label,
                        zorder=4 if m == "two_test" else 3)
            lines.append(f"  {m:10s} " + "  ".join(f"{lv}: {v:.3f} [{lo:.3f}-{hi:.3f}] n={n}" for lv, v, lo, hi, n in
                                                   zip(levels, r["value"], r["ci_lo_air"], r["ci_hi_air"], r["reports"])))
        ax.set_xticks(x, ticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if split == "air":                                          # balanced accuracy: 0.5 is chance
            ax.set_ylim(0.5, 1)
            # the two coldest classes come mostly from one airport (Jackson Hole: 85 % and 71 % of their reports, Sect. 4.3)
            ax.axvspan(-0.5, 1.5, color="#f4f4f4", lw=0, zorder=-1)   # behind the grid
            ax.text(0.5, 0.985, "mostly one airport", ha="center", va="top", fontsize=6, color=fs.MUTED)
            ax.text(-0.22, 0.505, "chance", ha="left", va="bottom", fontsize=6, color=fs.MUTED)
        else:
            ax.set_ylim(0, 1)
        fs.style(ax)
        ax.set_title(f"({panel})", loc="left", fontsize=7, pad=3)
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, handlelength=2.8,
                   columnspacing=1.4, borderaxespad=0)
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG)                      # full figure width: 8.3 cm at 300 dpi (AMT minimum 8 cm)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
