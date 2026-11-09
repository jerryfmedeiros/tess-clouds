"""fig_example_night.py — one night of the classifier at work, beside the automated METAR reports and the Meteosat
cloud mask.

The night is chosen by a rule on the references only, never on the classifier's calls: among nights at METAR
photometers inside the Meteosat disc with the Moon down, no bin at the firmware floor, at least 12 clear or cloudy
automated reports (at least six of each, about 3 h each at 30 min reports), at least one change between clear and
cloudy, and at least 20 Meteosat matches: the fewest changes, then the most reports in the smaller class, then the
most reports.

Panel (a): METAR reports over their 30 min windows (clear, FEW/SCT, cloudy), Meteosat boxes over +-15 min around
each scan (clear, partial, cloudy) and the classifier's call for each 5 min bin, as blocks in three grey levels.
Panel (b): the 5 min median sky temperature, the out-of-sample clear-sky baseline (Brunt form, features.py), the
ERA5-Land air temperature, and the range read as clear (up to baseline + kz times the photometer's held-out
clear-sky SD) as a shaded band, labelled in the figure. Panel (c): the excess within-bin scatter, with the range up to kv shaded. Thresholds: those of
the photometer's held-out fold (results/classify_metar_thresholds.csv), so that the strip shows the held-out calls the
scores use; the fold is the one whose thresholds reproduce every stored held-out call of the photometer
(call_two_test in data/metar/scored.csv.gz), so no threshold is typed. The released thresholds
(data/frozen_thresholds.csv) are written to the results file beside them.

Writes figures/fig_example_night.png and results/fig_example_night.txt (the chosen night and the values plotted).

Usage: python fig_example_night.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import figstyle as fs

ROOT = Path(__file__).parent
SCORED = ROOT / "data" / "metar" / "scored.csv.gz"
MSG = ROOT / "data" / "sat" / "msg.csv.gz"
FEAT = ROOT / "data" / "features"
UNITS = ROOT / "results" / "features_units.csv"
THR = ROOT / "data" / "frozen_thresholds.csv"
FOLDS = ROOT / "results" / "classify_metar_thresholds.csv"
OUT_PNG = ROOT / "figures" / "fig_example_night.png"
OUT_TXT = ROOT / "results" / "fig_example_night.txt"
STATE = {"clear": "#e4e4e4", "partial": "#9a9a9a", "cloudy": "#262626"}   # three lightness steps, read in greyscale
BAND = "#eeeeee"                                                          # the range read as clear


def choose(s: pd.DataFrame, msg: pd.DataFrame) -> pd.Series:
    s = s[s["label"].isin(["clear", "cloudy"]) & s["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])]
    rows = []
    for (uid, night), g in s.groupby(["id", "night"]):
        if len(g) < 12 or g["moon_up"].max() > 0 or g["floor"].max() > 0:
            continue
        g = g.sort_values("time")
        n_clear, n_cloudy = int((g["label"] == "clear").sum()), int((g["label"] == "cloudy").sum())
        if min(n_clear, n_cloudy) < 6:
            continue
        changes = int((g["label"] != g["label"].shift()).sum()) - 1
        n_msg = int(((msg["id"] == uid) & msg["time"].between(g["time"].min() - pd.Timedelta("1h"), g["time"].max())).sum())
        if changes < 1 or n_msg < 20:
            continue
        rows.append(dict(id=uid, night=night, reports=len(g), smaller=min(n_clear, n_cloudy), changes=changes, msg=n_msg))
    c = pd.DataFrame(rows).sort_values(["changes", "smaller", "reports"], ascending=[True, False, False])
    return c.iloc[0], len(c)


def main():
    s = pd.read_csv(SCORED, parse_dates=["time"])
    msg = pd.read_csv(MSG, usecols=["id", "time", "label", "night"], parse_dates=["time"])
    pick, n_cand = choose(s, msg)
    uid, night = pick["id"], pick["night"]
    final = pd.read_csv(THR, comment="#").iloc[0]
    kz_rel, kv_rel = float(final["kx"]), float(final["kv"])
    folds = pd.read_csv(FOLDS)
    folds = folds[folds["fold"] != "all"]
    r = s[(s["id"] == uid) & s["label"].isin(["clear", "cloudy"]) & s["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])]
    match = {f.fold: float(((((r["z_brunt"] > f.kx) | (r["s5_excess"] > f.kv)) & ~(r["floor"] >= 0.5))
                            == r["call_two_test"].astype(bool)).mean()) for f in folds.itertuples()}
    fold = max(match, key=match.get)
    assert match[fold] == 1.0, f"no fold reproduces the held-out calls of {uid}: {match}"
    kz, kv = (float(x) for x in folds.set_index("fold").loc[fold, ["kx", "kv"]])
    sd = float(pd.read_csv(UNITS).set_index("id").loc[uid, "sd_brunt"])
    f = pd.read_csv(FEAT / f"{uid}.csv.gz", parse_dates=["time"])
    rep = s[(s["id"] == uid) & (s["night"] == night)].sort_values("time")
    t0, t1 = rep["time"].min() - pd.Timedelta("30min"), rep["time"].max()
    f = f[f["time"].between(t0, t1)].copy()
    f["call"] = ((f["z_brunt"] > kz) | (f["s5_excess"] > kv)) & ~(f["floor"] >= 0.5)
    m = msg[(msg["id"] == uid) & msg["time"].between(t0, t1)]

    plt.rcParams.update(fs.RC)
    fig = plt.figure(figsize=(fs.WIDE, 0.66 * fs.WIDE))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.42, 2.2, 0.95], hspace=0.2)
    ax_s, ax_t, ax_v = (fig.add_subplot(gs[i]) for i in range(3))
    ax_t.sharex(ax_s)
    ax_v.sharex(ax_s)

    # strips: each reference as blocks over the time it covers, shaded by its state (three lightness steps)
    rows = {"METAR": 2, "Meteosat": 1, "Classifier": 0}

    def blocks(starts, ends, labels, y, gap=0.6):
        for t0_, t1_, lab in zip(starts, ends, labels):
            ax_s.broken_barh([(mdates.date2num(t0_), mdates.date2num(t1_) - mdates.date2num(t0_))], (y - 0.3, 0.6),
                             facecolor=STATE[lab], edgecolor="white" if gap else STATE[lab], lw=gap or 0.3)

    blocks(rep["time"] - pd.Timedelta("30min"), rep["time"], rep["label"], rows["METAR"], gap=0)   # the report's window;
    # no white edge: specials overlap the routine reports' windows, and edges there would leave slivers (later on top)
    blocks(m["time"] - pd.Timedelta("15min"), m["time"] + pd.Timedelta("15min"), m["label"], rows["Meteosat"])
    blocks(f["time"] - pd.Timedelta("150s"), f["time"] + pd.Timedelta("150s"),
           np.where(f["call"], "cloudy", "clear"), rows["Classifier"], gap=0)          # contiguous bins, no gaps
    ax_s.set_yticks(list(rows.values()), list(rows.keys()))
    ax_s.set_ylim(-0.45, 2.45)
    ax_s.tick_params(axis="x", labelbottom=False, length=0)
    ax_s.tick_params(axis="y", length=0)
    for side in ("top", "right", "bottom", "left"):
        ax_s.spines[side].set_visible(False)
    ax_s.set_title("(a)", loc="left", fontsize=7, pad=2)
    key = [Patch(facecolor=STATE[k], edgecolor="none", label=lab) for k, lab in
           (("clear", "clear"), ("partial", "few/scattered or partial"), ("cloudy", "cloudy"))]
    ax_s.legend(handles=key, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=3, frameon=False, fontsize=6.5,
                handlelength=1.4, handleheight=0.7, columnspacing=1.2, borderaxespad=0)

    end = f["time"].iloc[-1]
    thr = f["base_brunt"] + kz * sd

    def tag(ax, yv, text, colour, va="center"):
        ax.text(end + pd.Timedelta("6min"), yv, text, va=va, ha="left", fontsize=6.5, color=colour, clip_on=False)

    ax_t.fill_between(f["time"], f["base_brunt"].min() - 50, thr, color=BAND, lw=0)
    ax_t.plot(f["time"], thr, color=fs.G3, lw=0.6)
    ax_t.plot(f["time"], f["t2m_C"], color=fs.G2, lw=0.8, ls=(0, (5, 2)))
    ax_t.plot(f["time"], f["base_brunt"], color=fs.INK, lw=0.9)
    ax_t.plot(f["time"], f["sky_med"], color=fs.ACCENT, lw=1.2)
    tag(ax_t, f["t2m_C"].iloc[-1], "air (ERA5-Land)", fs.G1)
    tag(ax_t, f["sky_med"].iloc[-1], "sky (TESS-W)", fs.ACCENT)
    tag(ax_t, thr.iloc[-1], f"threshold, +{kz:.2f} SD\n(shaded: read as clear)", fs.G1)   # legend in the figure (AMT)
    tag(ax_t, f["base_brunt"].iloc[-1], "clear-sky baseline", fs.INK)
    lo = min(f["sky_med"].min(), f["base_brunt"].min()) - 2
    ax_t.set_ylim(lo, max(f["t2m_C"].max(), f["sky_med"].max()) + 2)
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.tick_params(axis="x", labelbottom=False)
    fs.style(ax_t)
    ax_t.set_title("(b)", loc="left", fontsize=7, pad=2)

    ax_v.axhspan(-10, kv, color=BAND, lw=0)
    ax_v.axhline(kv, color=fs.G3, lw=0.6)
    ax_v.plot(f["time"], f["s5_excess"], color=fs.ACCENT, lw=0.9)
    ax_v.set_ylim(min(-0.3, f["s5_excess"].min() - 0.2), f["s5_excess"].max() + 0.4)
    tag(ax_v, kv, f"threshold, {kv:.2f} K\n(shaded: read as clear)", fs.G1, va="bottom")   # above the line, clear of the axis
    ax_v.set_ylabel("Excess scatter (K)")
    ax_v.set_xlabel(f"Time (UTC), night of {pd.Timestamp(night):%-d %B %Y}")
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fs.style(ax_v)
    ax_v.set_title("(c)", loc="left", fontsize=7, pad=2)

    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    agree = rep.merge(f[["time", "call"]], how="left", on="time")
    lines = [f"candidates {n_cand}; chosen {uid} night {night}: reports {pick['reports']}, smaller class {pick['smaller']}, "
             f"changes {pick['changes']}, Meteosat matches {pick['msg']}",
             f"held-out fold {fold} (reproduces all {len(r)} stored calls of the photometer): kz {kz} kv {kv}; "
             f"released kz {kz_rel} kv {kv_rel}; held-out clear-sky SD {sd} C",
             f"bins {len(f)}, called cloudy {int(f['call'].sum())}",
             "METAR reports (time, label, held-out two-test call):"]
    lines += [f"  {r.time:%H:%M} {r.label:7s} {bool(r.call_two_test)}" for r in rep.itertuples()]
    lines += ["Meteosat (time, label):"] + [f"  {r.time:%H:%M} {r.label}" for r in m.itertuples()]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
