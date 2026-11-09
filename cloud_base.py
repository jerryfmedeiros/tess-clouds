"""cloud_base.py — how much the photometer's infrared reading says about the height of the cloud base, against the
ceilometer of the paired airport. Design fixed on 2026-10-31, before this was run.

Reports: automated METAR (data/metar/labels/, metar_labels.py) at the photometers scored by classify_metar.py,
cloudy (BKN or OVC) with exactly one layer (lowest base = highest base), snow-free (snow depth <= SNOW_M), not at
the firmware floor. Photometer side: the bins of the 30 minutes before the report (WINDOW), medians, from
data/features/.
Target: base above the airport, log10 km (below MIN_KM set to MIN_KM), and classes low < 2000 ft, mid
2000-6500 ft, high >= 6500 ft.
Inputs: D = t2m - sky; R = (sky - clear baseline) / (t2m - clear baseline), clipped to [-0.5, 1.5] (base_brunt of
features.py, out of sample); dew-point depression t2m - d2m; t2m; OVC indicator.
  A  least squares on log10 height
  B  median log10 height in training cells of R deciles x dew-point-depression quintiles x OVC (cells with fewer
     than MIN_CELL reports fall back to the R decile)
  typical  the training-fold median height
Validation: 5 folds grouped by airport, assigned as in classify_metar.py. Scores: median absolute error (km),
share within a factor 2, Spearman rank correlation, 3-class balanced accuracy, confusion table; 95% night-block
bootstrap within photometers (B, SEED). Also the non-US airports (US ASOS ceilometers stop at 12,000 ft).

Not part of the fixed design (run afterwards): the least-squares fit with input subsets, to separate what the
photometer adds from what ERA5 humidity alone gives (the dew-point depression sets the lifting condensation level,
about 125 m per C): ERA5 only (dpd, t2m), photometer only (D, R), each with OVC, and the full set.

Writes results/cloud_base.csv, results/cloud_base.txt and data/metar/cloud_base.csv.gz (held-out predictions).

Usage: python cloud_base.py"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import local_night, longitudes

from classify_metar import FOLDS
from qc_ir import obstructed

ROOT = Path(__file__).parent
FEAT = ROOT / "data" / "features"
LABELS = ROOT / "data" / "metar" / "labels"
SCORED = ROOT / "data" / "metar" / "scored.csv.gz"
OUT_CSV = ROOT / "results" / "cloud_base.csv"
OUT_TXT = ROOT / "results" / "cloud_base.txt"
OUT_PRED = ROOT / "data" / "metar" / "cloud_base.csv.gz"
SEED, B = 20261030, 1000
WINDOW = pd.Timedelta(minutes=30)
SNOW_M = 0.01
MIN_KM = 0.03
MIN_CELL = 50
FT = 0.0003048
EDGES = [0.0, 2000 * FT, 6500 * FT, np.inf]
CLASSES = ["low", "mid", "high"]
COLS = ["sky_med", "base_brunt", "t2m_C", "d2m_C", "floor", "sde_m", "z_brunt"]


def reports(uid: str) -> pd.DataFrame:
    f = pd.read_csv(FEAT / f"{uid}.csv.gz", usecols=["time", *COLS], parse_dates=["time"])
    r = pd.read_csv(LABELS / f"{uid}.csv.gz")
    r["time"] = pd.to_datetime(r["time"], errors="coerce", format="mixed")
    r = r[(r["code"] == "LAYERS") & r["okta"].isin([6, 8]) & r["base_ft"].notna() & (r["base_ft"] == r["top_ft"])]
    r = r.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if r.empty:
        return pd.DataFrame()
    t = f["time"].to_numpy()
    lo = np.searchsorted(t, (r["time"] - WINDOW).to_numpy(), side="right")
    hi = np.searchsorted(t, r["time"].to_numpy(), side="right")
    v = f[COLS].to_numpy(float)
    out = np.full((len(r), len(COLS)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                out[i] = np.nanmedian(v[a:b], axis=0)
    o = pd.DataFrame(out, columns=COLS)
    o.insert(0, "id", uid)
    o["icao"], o["time"], o["okta"], o["base_ft"] = r["icao"], r["time"], r["okta"], r["base_ft"]
    o["night"] = local_night(r["time"], longitudes()[uid])
    return o


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), d["D"], d["R"], d["dpd"], d["t2m_C"], d["ovc"]])


def model_b(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    rq = np.quantile(tr["R"], np.linspace(0, 1, 11)[1:-1])
    dq = np.quantile(tr["dpd"], np.linspace(0, 1, 6)[1:-1])
    key = lambda g: (np.searchsorted(rq, g["R"]), np.searchsorted(dq, g["dpd"]), g["ovc"].to_numpy(int))
    a, b, c = key(tr)
    cells = pd.DataFrame({"r": a, "q": b, "o": c, "y": tr["y"].to_numpy()})
    full = cells.groupby(["r", "q", "o"])["y"].agg(["median", "size"])
    full = full[full["size"] >= MIN_CELL]["median"].to_dict()
    byr = cells.groupby("r")["y"].median().to_dict()
    a, b, c = key(te)
    return np.array([full.get((x, y, z), byr.get(x, np.median(tr["y"]))) for x, y, z in zip(a, b, c)])


def metrics(km: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    tc, pc = np.digitize(km, EDGES[1:-1]), np.digitize(p, EDGES[1:-1])
    ba = np.mean([(pc[tc == k] == k).mean() for k in range(3) if (tc == k).any()])
    rho = pd.Series(p).corr(pd.Series(km), method="spearman")
    return float(np.median(np.abs(p - km))), float((np.abs(np.log2(p / km)) < 1).mean()), float(rho), float(ba)


def boot(d: pd.DataFrame, col: str, rng) -> tuple[tuple, tuple]:
    key = (d["id"] + "|" + d["night"]).to_numpy()
    blk, inv = np.unique(key, return_inverse=True)
    rows = pd.Series(np.arange(len(d))).groupby(inv).apply(np.array).to_list()
    unit = pd.Series(blk).str.split("|").str[0].to_numpy()
    order = np.argsort(unit, kind="stable")
    _, start, size = np.unique(unit[order], return_index=True, return_counts=True)
    s0, n0 = np.repeat(start, size), np.repeat(size, size)
    km, p = d["km"].to_numpy(), d[col].to_numpy()
    mae, ba = np.empty(B), np.empty(B)
    for b in range(B):
        pick = order[s0 + (rng.random(len(blk)) * n0).astype(int)]
        idx = np.concatenate([rows[i] for i in pick])
        m = metrics(km[idx], p[idx])
        mae[b], ba[b] = m[0], m[3]
    return tuple(np.percentile(mae, [2.5, 97.5])), tuple(np.percentile(ba, [2.5, 97.5]))


def main():
    skip = obstructed()
    ids = sorted(set(pd.read_csv(SCORED, usecols=["id"])["id"]) - skip)
    d = pd.concat([reports(u) for u in ids], ignore_index=True)
    d = d[(d["sde_m"] <= SNOW_M) & (d["floor"] < 0.5) & d[["sky_med", "base_brunt", "t2m_C", "d2m_C"]].notna().all(axis=1)]
    d = d.reset_index(drop=True)
    d["km"] = (d["base_ft"] * 0.3048 / 1000).clip(lower=MIN_KM)
    d["y"] = np.log10(d["km"])
    d["D"] = d["t2m_C"] - d["sky_med"]
    d["R"] = ((d["sky_med"] - d["base_brunt"]) / (d["t2m_C"] - d["base_brunt"])).clip(-0.5, 1.5)
    d["dpd"] = d["t2m_C"] - d["d2m_C"]
    d["ovc"] = (d["okta"] == 8).astype(int)
    d = d[np.isfinite(d["R"])].reset_index(drop=True)
    airports = sorted(pd.read_csv(SCORED, usecols=["icao"])["icao"].unique())
    fold_of = dict(zip(airports, np.random.default_rng(SEED).permutation(np.arange(len(airports)) % FOLDS)))
    fold = d["icao"].map(fold_of).to_numpy()
    for c in ("A", "B", "typical"):
        d[c] = np.nan
    coef = []
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        if not te.any():
            continue
        X = design(d)
        b = np.linalg.lstsq(X[tr], d["y"].to_numpy()[tr], rcond=None)[0]
        coef.append(b)
        d.loc[te, "A"] = 10 ** (X[te] @ b)
        d.loc[te, "B"] = 10 ** model_b(d[tr], d[te])
        d.loc[te, "typical"] = 10 ** np.median(d["y"].to_numpy()[tr])
    pairs = pd.read_csv(ROOT / "data" / "metar" / "pairs.csv")[["icao", "country"]].drop_duplicates("icao")
    d = d.merge(pairs, on="icao", how="left")
    d[["id", "icao", "country", "time", "night", "okta", "km", "D", "R", "dpd", "t2m_C", "A", "B", "typical"]].to_csv(
        OUT_PRED, index=False, compression="gzip")
    rng = np.random.default_rng(SEED)
    rows = []
    lines = [f"single-layer BKN/OVC reports {len(d)} at {d['icao'].nunique()} airports, {d['id'].nunique()} photometers; "
             f"held out by airport (5 folds)",
             "least-squares coefficients on log10 km (mean over folds): "
             + ", ".join(f"{n} {v:+.4f}" for n, v in zip(["intercept", "D", "R", "dpd", "t2m", "ovc"], np.mean(coef, axis=0)))]
    for name, sel in (("all", np.ones(len(d), bool)), ("non-US airports", (d["country"] != "US").to_numpy())):
        g = d[sel].reset_index(drop=True)
        tc = pd.Categorical(np.array(CLASSES)[np.digitize(g["km"], EDGES[1:-1])], CLASSES)
        lines.append(f"\n{name}: {len(g)} reports; true classes "
                     + ", ".join(f"{k} {int((tc == k).sum())}" for k in CLASSES))
        for m in ("A", "B", "typical"):
            mae, f2, rho, ba = metrics(g["km"].to_numpy(), g[m].to_numpy())
            ci_mae, ci_ba = boot(g, m, rng)
            rows.append([name, m, len(g), round(mae, 6), round(ci_mae[0], 6), round(ci_mae[1], 6), round(f2, 6),
                         round(rho, 6) if np.isfinite(rho) else "", round(ba, 6), round(ci_ba[0], 6), round(ci_ba[1], 6)])
            lines.append(f"  {m:8s} median error {mae:.2f} km [{ci_mae[0]:.2f}-{ci_mae[1]:.2f}]  within x2 {f2:.2f}  "
                         f"rank {rho:.2f}  3-class balanced {ba:.2f} [{ci_ba[0]:.2f}-{ci_ba[1]:.2f}]")
            if m != "typical":
                pc = pd.Categorical(np.array(CLASSES)[np.digitize(g[m], EDGES[1:-1])], CLASSES)
                ct = pd.crosstab(tc, pc, normalize="index", dropna=False).reindex(index=CLASSES, columns=CLASSES).fillna(0)
                lines.append("    rows true, columns predicted: " + "; ".join(
                    f"{k}: " + " ".join(f"{ct.loc[k, j]:.2f}" for j in CLASSES) for k in CLASSES))
    lines.append("\nnot part of the fixed design: least squares with input subsets (all reports)")
    y = d["y"].to_numpy()
    for name, cols in (("ERA5 only (dpd, t2m)", ["dpd", "t2m_C"]), ("ERA5 + OVC", ["dpd", "t2m_C", "ovc"]),
                       ("photometer only (D, R)", ["D", "R"]), ("photometer + OVC", ["D", "R", "ovc"]),
                       ("full (model A)", ["D", "R", "dpd", "t2m_C", "ovc"])):
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
        p = np.empty(len(d))
        for k in range(FOLDS):
            tr, te = fold != k, fold == k
            if te.any():
                p[te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        m = metrics(d["km"].to_numpy(), 10 ** p)
        lines.append(f"  {name:24s} median error {m[0]:.2f} km  within x2 {m[1]:.2f}  rank {m[2]:.2f}  3-class balanced {m[3]:.2f}")
    q = pd.qcut(d["dpd"], 5)
    lines.append("  rank of D with height within dew-point-depression quintiles: "
                 + ", ".join(f"{g['D'].corr(g['km'], method='spearman'):.2f}" for _, g in d.groupby(q, observed=True)))
    pd.DataFrame(rows, columns=["subset", "model", "reports", "median_error_km", "ci_lo", "ci_hi", "within_x2", "spearman",
                                "balanced_3class", "ba_lo", "ba_hi"]).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
