"""features.py — the inputs of the cloud classifier and of the comparison methods, per 5-minute night bin, for
every photometer in data/nights/.

Baseline test (spectral):
  device                   the device (MAC tail) behind the bin's monthly file (results/audit_devices.csv).
  base_brunt, base_linear  expected clear-sky sky temperature (C) from ERA5-Land air temperature and dew point, in
                           the Brunt and linear forms of baseline_explore.py, with one offset per device. At mountain
                           photometers (data/mountain_pl, from mountain_pl.py) the air temperature and dew point are
                           those of ERA5 pressure levels at the site's height instead, linear in time between the
                           6-hourly values (met_pl = 1); ERA5-Land's 2 m cell lies far below a summit. The output
                           columns t2m_C and d2m_C stay ERA5-Land everywhere (the conditions of breakdown_metar.py
                           and cloud_base.py); base_t_C and base_td_C are the values the baseline used. Out of
                           sample: the bins of each interleaved-month fold are predicted by a fit to the clear
                           candidates of the other folds, so no bin is predicted by a fit that saw it; a device
                           with no candidates in the other folds gets no baseline.
  res_brunt, res_linear    sky_med minus the baseline (C); clouds read warmer, so cloud gives a positive residual.
  z_brunt, z_linear        the residual over the photometer's held-out clear-sky robust SD (sd_* in units file).
Variability test (temporal), adapted to the archive cadence of about one reading a minute:
  s5                       standard deviation of the readings within the bin (sky_std, about 5 readings).
  s5_clear                 the clear-sky value of s5 at the bin's sky temperature: the median s5 of the
                           photometer's clear candidates in 2 C bins of sky_med, interpolated (sensor noise
                           grows as the sky gets colder; after the dynamic threshold of Ahn et al. 2015, Eq. 6).
  s5_excess                s5 minus s5_clear.
  d30                      scatter of the bin-to-bin changes of sky_med over a centred 30-minute window
                           (7 bins; 1.4826 x the median absolute change), for broken cloud that moves slower
                           than the within-bin scatter shows.
  dfa_alpha, dfa_f20       detrended fluctuation analysis of sky_med in a centred 2-hour window (24 bins, at least
                           18 present), box sizes 3, 4, 6 and 8 bins (15-40 min; a
                           2-bin box is fitted exactly by its line): the log-log slope and the
                           fluctuation at 20 min (C). After Kerber et al. 2016, who use 2-hour windows and boxes of
                           60-300 s at a faster cadence; at 5-minute bins the boxes here are longer.
Comparison methods:
  dT                       die_med minus sky_med (C), the "T_amb - T_sky" of the network and of Alarcon et al. 2021
                           (the TESS-W ambient is the sensor die).
  vis15                    range of mag_med over the centred 3 bins (15 min), the visible-band stability.
Flags carried from data/nights: floor (floor_share >= 0.5: the true sky is at or below -50 C, censored, read as
clear), moon_up (moon_alt > 0), dark (sun_alt < -18).

Writes data/features/<id>.csv.gz (one row per night bin) and results/features_units.csv (per photometer:
candidates, held-out robust SD and median residual of each form, number of bins with a baseline). Reads data/
nights, so it is a verify target once the release is assembled.

Usage: python features.py [--only stars645 ...] [--workers 12]"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_explore as be

ROOT = Path(__file__).parent
NIGHTS = ROOT / "data" / "nights"
MOUNT_PL = ROOT / "data" / "mountain_pl"
PL_GAP_H = 3.5                                      # the 6-hourly values are at most 3 h from any bin
OUT = ROOT / "data" / "features"
UNITS = ROOT / "results" / "features_units.csv"
FORMS = ("brunt", "linear")
DFA_BOXES = (3, 4, 6, 8)
DFA_WINDOW, DFA_MIN = 24, 18


def cv_baseline(d: pd.DataFrame, c: pd.DataFrame, form: str) -> tuple[np.ndarray, float, float]:
    """Out-of-sample baseline for every bin of d; held-out robust SD and median of the clear core."""
    pred = np.full(len(d), np.nan)
    have = d[["t2m_C", "d2m_C"]].notna().all(axis=1).to_numpy()
    for k in range(be.FOLDS):
        tr = c[c["fold"] != k]
        m = have & (d["fold"] == k).to_numpy()
        if len(tr) < be.MIN_CANDIDATES or not m.any():
            continue
        pred[m] = be.predict(form, be.fit(form, tr), d[m])
    res = be.cv_residuals(form, c)
    if not res:
        return pred, np.nan, np.nan
    sd, med = be.robust(np.concatenate(res))
    return pred, sd, med


def clear_noise(c: pd.DataFrame, sky: np.ndarray) -> np.ndarray:
    """Median within-bin scatter of clear candidates as a function of sky temperature, at each value of sky."""
    cc = c.dropna(subset=["sky_std"])
    edges = np.arange(np.floor(cc["sky_med"].min() / 2) * 2, cc["sky_med"].max() + 2, 2.0)
    g = cc.groupby(pd.cut(cc["sky_med"], edges), observed=True)
    tab = pd.DataFrame({"x": g["sky_med"].median(), "y": g["sky_std"].median(), "n": g.size()})
    tab = tab[tab["n"] >= 30].sort_values("x")
    if len(tab) < 2:
        return np.full(len(sky), cc["sky_std"].median() if len(cc) else np.nan)
    return np.interp(sky, tab["x"].to_numpy(float), tab["y"].to_numpy(float))


def dfa(x: np.ndarray) -> tuple[float, float]:
    """DFA-1 of one window (NaNs dropped): slope of log F against log box size, and F at the middle box."""
    x = x[np.isfinite(x)]
    if len(x) < DFA_MIN:
        return np.nan, np.nan
    y = np.cumsum(x - x.mean())
    f = []
    for s in DFA_BOXES:
        k = len(y) // s
        segs = y[:k * s].reshape(k, s)
        t = np.arange(s)
        A = np.column_stack([t, np.ones(s)])
        coef = np.linalg.lstsq(A, segs.T, rcond=None)[0]
        f.append(np.sqrt(np.mean((segs - (A @ coef).T) ** 2)))
    f = np.array(f)
    if np.any(f <= 0):
        return np.nan, float(f[1])
    return float(np.polyfit(np.log(DFA_BOXES), np.log(f), 1)[0]), float(f[1])


def windows(d: pd.DataFrame) -> pd.DataFrame:
    """d30, vis15 and DFA on a regular 5-minute grid, so windows never bridge a gap as if it were continuous."""
    s = d.set_index("time")[["sky_med", "mag_med"]]
    g = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="5min"))
    step = g["sky_med"].diff().abs()
    d30 = 1.4826 * step.rolling(7, center=True, min_periods=5).median()
    vis = g["mag_med"].rolling(3, center=True, min_periods=3)
    vis15 = vis.max() - vis.min()
    v = g["sky_med"].to_numpy()
    half = DFA_WINDOW // 2
    alpha, f20 = np.full(len(v), np.nan), np.full(len(v), np.nan)
    ok = np.isfinite(v)
    run = np.convolve(ok, np.ones(DFA_WINDOW), mode="same")       # values present in each window
    for i in np.flatnonzero(ok & (run >= DFA_MIN)):
        alpha[i], f20[i] = dfa(v[max(0, i - half):i + half])
    w = pd.DataFrame({"d30": d30, "vis15": vis15, "dfa_alpha": alpha, "dfa_f20": f20}, index=g.index)
    return w.reindex(s.index).reset_index(drop=True)


def mountain_met(d: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Air temperature and dew point from ERA5 pressure levels at a mountain photometer, where they cover the bin."""
    d["met_pl"] = 0
    f = MOUNT_PL / f"{uid}.csv.gz"
    if not f.exists():
        return d
    s = pd.read_csv(f, parse_dates=["time"])
    st = s["time"].to_numpy("datetime64[s]").astype("int64")
    bt = d["time"].to_numpy("datetime64[s]").astype("int64")
    k = np.clip(np.searchsorted(st, bt), 1, len(st) - 1)
    ok = (np.minimum(np.abs(st[k] - bt), np.abs(st[k - 1] - bt)) <= PL_GAP_H * 3600) & (bt >= st[0]) & (bt <= st[-1])
    for col, src in (("t2m_C", "t_C"), ("d2m_C", "td_C")):
        d.loc[ok, col] = np.interp(bt[ok], st, s[src].to_numpy(float))
    d.loc[ok, "met_pl"] = 1
    return d


def one(f: Path) -> list:
    uid = f.name.split(".")[0]
    d = be.attach_device(pd.read_csv(f, parse_dates=["time"]).sort_values("time").reset_index(drop=True), uid)
    land = d[["t2m_C", "d2m_C"]].copy()
    d = mountain_met(d, uid)
    out = OUT / f"{uid}.csv.gz"
    c = be.candidates(d, uid)
    o = pd.DataFrame({"time": d["time"], "device": d["device"], "sky_med": d["sky_med"]})
    row = [uid, len(c)]
    for form in FORMS:
        if len(c) >= be.MIN_CANDIDATES:
            pred, sd, med = cv_baseline(d, c, form)
        else:
            pred, sd, med = np.full(len(d), np.nan), np.nan, np.nan
        o[f"base_{form}"] = pred
        o[f"res_{form}"] = d["sky_med"] - pred
        o[f"z_{form}"] = o[f"res_{form}"] / sd if np.isfinite(sd) and sd > 0 else np.nan
        row += [round(sd, 3) if np.isfinite(sd) else "", round(med, 3) if np.isfinite(med) else "",
                int(np.isfinite(pred).sum())]
    o["s5"] = d["sky_std"]
    o["s5_clear"] = clear_noise(c, d["sky_med"].to_numpy()) if len(c) else np.nan
    o["s5_excess"] = o["s5"] - o["s5_clear"]
    o = pd.concat([o, windows(d)], axis=1)
    o["dT"] = d["die_med"] - d["sky_med"]
    o["floor"] = (d["floor_share"] >= 0.5).astype(int)
    o["moon_up"] = (d["moon_alt"] > 0).astype(int)
    o["dark"] = (d["sun_alt"] < -18).astype(int)
    o["base_t_C"], o["base_td_C"] = d["t2m_C"], d["d2m_C"]
    for col in ("moon_alt", "moon_illum", "mag_med", "met_pl", "sde_m"):
        o[col] = d[col]
    o["t2m_C"], o["d2m_C"] = land["t2m_C"], land["d2m_C"]
    OUT.mkdir(parents=True, exist_ok=True)
    o.round({c: 3 for c in o.columns if c not in ("time", "device")}).to_csv(out.with_suffix(".part"), index=False, compression="gzip")
    out.with_suffix(".part").replace(out)
    return row + [len(d)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = [f for f in sorted(NIGHTS.glob("*.csv.gz")) if not a.only or f.name.split(".")[0] in a.only]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, files))
    if a.only and UNITS.exists():                     # replace the rows of the photometers rerun
        done = {r[0] for r in rows}
        old = [r for r in csv.reader(UNITS.open())][1:]
        rows = rows + [r for r in old if r[0] not in done]
    UNITS.parent.mkdir(exist_ok=True)
    with UNITS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "candidates"] + [f"{k}_{form}" for form in FORMS for k in ("sd", "med", "bins_with_baseline")]
                   + ["bins"])
        w.writerows(sorted(rows))
    for r in rows[:5] if a.only is None else rows:
        print(r)


if __name__ == "__main__":
    main()
