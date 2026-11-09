"""classify_metar.py — the two-test classifier and the comparison methods, scored against automated METAR.

Each automated night report (data/metar/labels/) is compared with the photometer bins it covers (the 30 minutes
before it). Per report, the photometer side is summarised by the median over those bins of z_brunt, z_linear, sky_med and neg_dT
(candidates for the spectral test: the baseline residuals, the raw sky temperature, and the sky minus the
sensor die, whose temperature follows the local air when the heater is off), s5_excess, d30 and dfa_f20
(variability candidates), dT and the largest vis15; a report whose bins are mostly at the firmware floor gets
-inf for every spectral score (the sky was at or below -50 C: clear).

Reference: clear (no layer: CLR, SKC, NCD) against cloudy (BKN, OVC, VV). FEW/SCT reports and NSC/CAVOK reports
are scored separately, as the share the classifier calls cloudy.

Classifier: cloudy if z_brunt > kz or s5_excess > kv (TWO_TEST: the ERA5 baseline residual and the within-bin
scatter; the pair chosen in the exploratory runs, where the training folds picked it from every spectral and
variability candidate in nearly every fold). Only the two thresholds are learned, on training photometers only,
by maximising balanced accuracy over a grid of quantiles: grouped 5-fold cross-validation by airport (every
photometer paired with an airport falls in that airport's fold, so no test report shares its reference with a
training report), each fold scored with what was learned on the other four, so every reported score is for
photometers and airports no threshold has seen. The primary split is seeded with SEED; the same cross-validation is repeated for REPEATS splits (seeds SEED
to SEED + REPEATS - 1) and the spread of each score across them is reported beside the bootstrap interval, so the
headline does not depend on one grouping of photometers. The final thresholds (fitted on all photometers) are
written for use at sites with no airport.

Comparison methods, on the same reports: baseline test alone, each variability test alone, the network dashboard
rule (cloud = 100 - 3 dT; cloudy above 55, the midpoint of its clear < 40 and cloudy > 70 bands), the
Alarcon et al. 2021 cut (clear if dT > 54.5 C), the visible-band stability alone (threshold learned as above), and
"always clear". AUC for every continuous score (no threshold needed).

Scores: balanced accuracy, hit rate (cloudy called cloudy), false-alarm rate (clear called cloudy), on the
primary split, with 95% intervals from a bootstrap that resamples whole nights within photometers (B, seeded; a
night is the local solar date at its start, common.local_night), the mean, minimum and maximum over the REPEATS
splits, and a 95% interval from a bootstrap that resamples whole
airports (common.cluster_ci), which carries the between-airport spread that the night-block interval does not.

One airport (KJAC, Jackson Hole) reports every 5 min and holds 39% of the scored reports,
so every method is scored again on the reports thinned to one per 30-minute slot of the clock per airport (the
last in each slot; THIN), and the per-airport scores are written (results/classify_metar_airports.csv: airport,
photometers, reports, share of the scored reports, median interval between reports, scores per method).

Every method is also scored with FEW/SCT reports counted cloudy, the basis of Ahn et al. (2015), who counted any cloud base below 7.62 km as cloud (metrics *_fewsct_cloudy, airport-cluster
interval from its own seeded generator so the other intervals are unchanged).

Writes results/classify_metar.csv (method, metric, value, ci_lo, ci_hi; the metrics *_cluster carry the airport
interval, *_thinned the thinned scores), results/classify_metar_thresholds.csv (per fold of the primary split,
and final), results/classify_metar_repeats.csv (split seed, method, scores), results/classify_metar_airports.csv,
results/classify_metar_paired.csv (per method: airports where the two-test rule scores higher, mean and median
difference in balanced accuracy, 95% airport-paired bootstrap interval of the mean difference; airports with at least
MIN_EACH clear and cloudy reports), results/classify_metar.txt, and
data/metar/scored.csv.gz (one row per report, held-out calls).

Usage: python classify_metar.py"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import cluster_ci, local_night, longitudes

ROOT = Path(__file__).parent
FEAT = ROOT / "data" / "features"
LABELS = ROOT / "data" / "metar" / "labels"
OUT_CSV = ROOT / "results" / "classify_metar.csv"
OUT_THR = ROOT / "results" / "classify_metar_thresholds.csv"
OUT_TXT = ROOT / "results" / "classify_metar.txt"
SCORED = ROOT / "data" / "metar" / "scored.csv.gz"
SEED, FOLDS, B, REPEATS = 20261030, 5, 1000, 20
SPECTRAL = ("z_brunt", "z_linear", "sky_med", "neg_dT")
VARS = ("s5_excess", "d30", "dfa_f20")
SINGLE = (*SPECTRAL, *VARS, "vis15")
TWO_TEST = ("z_brunt", "s5_excess")
GRID = np.linspace(0.02, 0.98, 49)                 # quantiles of each score, as candidate thresholds
WINDOW = pd.Timedelta(minutes=30)
THIN = "30min"
ALARCON = 54.5
MIN_EACH = 100                                     # clear and cloudy reports for a per-airport score


def per_report(uid: str, lon: float) -> pd.DataFrame:
    f = pd.read_csv(FEAT / f"{uid}.csv.gz", parse_dates=["time"])
    r = pd.read_csv(LABELS / f"{uid}.csv.gz")
    r["time"] = pd.to_datetime(r["time"], errors="coerce", format="mixed")
    r = r.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    t = f["time"].to_numpy()
    lo = np.searchsorted(t, (r["time"] - WINDOW).to_numpy(), side="right")
    hi = np.searchsorted(t, r["time"].to_numpy(), side="right")
    f["neg_dT"] = -f["dT"]
    cols = [*SPECTRAL, *VARS, "dT", "floor", "moon_up", "sde_m", "t2m_C"]
    vals = f[cols + ["vis15"]].to_numpy(float)
    out = np.full((len(r), len(cols) + 1), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[i, :-1] = np.nanmedian(vals[a:b, :-1], axis=0)
                out[i, -1] = np.nanmax(vals[a:b, -1]) if np.isfinite(vals[a:b, -1]).any() else np.nan
    o = pd.DataFrame(out, columns=cols + ["vis15"])
    o.loc[o["floor"] >= 0.5, list(SPECTRAL)] = -np.inf
    o.insert(0, "id", uid)
    o.insert(1, "icao", r["icao"])
    o["time"], o["label"], o["code"], o["base_ft"], o["top_ft"] = r["time"], r["label"], r["code"], r["base_ft"], r["top_ft"]
    o["night"] = local_night(r["time"], lon)
    return o


def thinned(d: pd.DataFrame) -> np.ndarray:
    """One report per THIN slot of the clock per airport: the last report in each slot."""
    slot = d["time"].dt.floor(THIN)
    last = d.groupby([d["icao"], slot])["time"].transform("max")
    return (d["time"] == last).to_numpy()


def bal(cloudy_true: np.ndarray, call: np.ndarray) -> tuple[float, float, float]:
    hit = call[cloudy_true].mean() if cloudy_true.any() else np.nan
    fa = call[~cloudy_true].mean() if (~cloudy_true).any() else np.nan
    return (hit + 1 - fa) / 2, hit, fa


def auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s) | np.isneginf(s)
    y, s = y[ok], s[ok]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def thresholds(x: pd.Series) -> np.ndarray:
    v = x[np.isfinite(x)]
    return np.unique(np.quantile(v, GRID)) if len(v) else np.array([np.inf])


def learn(tr: pd.DataFrame) -> dict:
    """Best thresholds on training reports: each test alone, the visible test alone, and the two thresholds of
    the classifier (the OR of the TWO_TEST pair)."""
    y = (tr["label"] == "cloudy").to_numpy()
    best = {}
    for name in SINGLE:
        s = tr[name].to_numpy()
        sc = [(bal(y, s > k)[0], k) for k in thresholds(tr[name])]
        best[name] = max(sc)[1]
    x, v = TWO_TEST
    z, s = tr[x].to_numpy(), tr[v].to_numpy()
    two = []
    for kz in thresholds(tr[x]):
        cz = z > kz
        for kv in thresholds(tr[v]):
            two.append((bal(y, cz | (s > kv))[0], kz, kv))
    _, kz, kv = max(two)
    best["two_test"] = (x, v, kz, kv)
    return best


def calls(d: pd.DataFrame, th: dict) -> pd.DataFrame:
    c = pd.DataFrame(index=d.index)
    for name in SINGLE:
        c[name] = d[name].to_numpy() > th[name]
    x, v, kz, kv = th["two_test"]
    c["two_test"] = (d[x].to_numpy() > kz) | (d[v].to_numpy() > kv)
    c["dashboard"] = (100 - 3 * d["dT"]) > 55
    c["alarcon"] = ~(d["dT"] > ALARCON)
    c["always_clear"] = False
    return c


def boot(d: pd.DataFrame, c: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Night-block bootstrap within photometers: balanced accuracy, hit, false alarm per method."""
    key = d["id"] + "|" + d["night"]
    groups = {u: g for u, g in key.groupby(d["id"]).groups.items()}
    blocks = pd.Series(np.arange(len(d))).groupby(key.to_numpy()).apply(np.array).to_dict()
    unit_nights = {u: key[g].unique() for u, g in groups.items()}
    y = (d["label"] == "cloudy").to_numpy()
    out = {m: [] for m in c.columns}
    C = c.to_numpy()
    for _ in range(B):
        idx = np.concatenate([blocks[n] for u, ns in unit_nights.items() for n in rng.choice(ns, len(ns))])
        for j, m in enumerate(c.columns):
            out[m].append(bal(y[idx], C[idx, j]))
    return {m: np.array(v) for m, v in out.items()}


def cross_validate(d: pd.DataFrame, scored: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list]:
    """Held-out calls for every report, folds grouped by airport and assigned with this seed."""
    rng = np.random.default_rng(seed)
    airports = sorted(d["icao"].unique())
    fold_of = dict(zip(airports, rng.permutation(np.arange(len(airports)) % FOLDS)))
    fold = d["icao"].map(fold_of)
    held = pd.DataFrame(index=d.index, columns=[*SINGLE, "two_test", "dashboard", "alarcon", "always_clear"],
                        dtype=bool)
    thr = []
    for k in range(FOLDS):
        th = learn(scored[scored["icao"].map(fold_of) != k])
        te = fold == k
        held.loc[te] = calls(d[te], th).to_numpy()
        thr.append([k, *th["two_test"], *(th[n] for n in SINGLE)])
    return held.astype(bool), thr


def per_airport(d: pd.DataFrame, held: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Scores per airport on the held-out calls, with each airport's weight in the scored reports."""
    km = pd.read_csv(ROOT / "results" / "metar_labels.csv").set_index("id")["km"]
    rows = []
    for icao, g in d.groupby("icao"):
        yy, idx = (g["label"] == "cloudy").to_numpy(), g.index
        gaps = g.sort_values(["id", "time"]).groupby("id")["time"].diff().dt.total_seconds() / 60
        rows.append([icao, g["id"].nunique(), int((~yy).sum()), int(yy.sum()), round(len(g) / len(d), 6),
                     round(float(gaps.median()), 1) if gaps.notna().any() else "", round(float(g["id"].map(km).mean()), 2),
                     *[round(bal(yy, held.loc[idx, m].to_numpy())[0], 6) for m in ("two_test", "z_brunt", "neg_dT", "dashboard")]])
    out = pd.DataFrame(rows, columns=["icao", "photometers", "clear", "cloudy", "share", "interval_min", "km",
                                      "two_test", "z_brunt", "neg_dT", "dashboard"])
    return out.sort_values("share", ascending=False).reset_index(drop=True)


def paired_airports(airports: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The two-test rule against each other method, airport by airport (airports with >= MIN_EACH of each class):
    airports where it scores higher, mean and median difference, and a 95% bootstrap interval of the mean difference
    that resamples airports with replacement (paired, so the between-airport spread cancels)."""
    ok = airports[(airports["clear"] >= MIN_EACH) & (airports["cloudy"] >= MIN_EACH)]
    rows = []
    for m in ("z_brunt", "neg_dT", "dashboard"):
        diff = (ok["two_test"] - ok[m]).to_numpy(float)
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append([m, len(diff), int((diff > 0).sum()), round(diff.mean(), 6), round(float(np.median(diff)), 6),
                     round(float(lo), 6), round(float(hi), 6)])
    return pd.DataFrame(rows, columns=["method", "airports", "two_test_higher", "mean_diff", "median_diff", "ci_lo", "ci_hi"])


def main():
    uids = sorted(p.name.split(".")[0] for p in LABELS.glob("*.csv.gz") if (FEAT / p.name).exists())
    lons = longitudes()
    d = pd.concat([per_report(u, lons[u]) for u in uids], ignore_index=True)
    d = d[(np.isfinite(d["z_brunt"]) | np.isneginf(d["z_brunt"])) & d["sky_med"].notna()].reset_index(drop=True)
    scored = d[d["label"].isin(["clear", "cloudy"]) & d["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])]
    s = scored.index
    y = (d.loc[s, "label"] == "cloudy").to_numpy()
    rep_rows = []
    for seed in range(SEED, SEED + REPEATS):
        h, thr = cross_validate(d, scored, seed)
        if seed == SEED:
            held, thr_rows = h, thr
        for m in h.columns:
            rep_rows.append([seed, m, *(round(x, 6) for x in bal(y, h.loc[s, m].to_numpy()))])
    final = learn(scored)
    thr_rows.append(["all", *final["two_test"], *(final[n] for n in SINGLE)])
    reps = pd.DataFrame(rep_rows, columns=["seed", "method", "balanced_accuracy", "hit_rate", "false_alarm"])
    bs = boot(d.loc[s].reset_index(drop=True), held.loc[s].reset_index(drop=True), np.random.default_rng(SEED))
    rng_c = np.random.default_rng(SEED)
    icao = d.loc[s, "icao"].to_numpy()
    thin = thinned(d.loc[s])
    rows = []
    for m in held.columns:
        c = held.loc[s, m].to_numpy()
        est = bal(y, c)
        for j, metric in enumerate(("balanced_accuracy", "hit_rate", "false_alarm")):
            lo, hi = np.nanpercentile(bs[m][:, j], [2.5, 97.5])
            rows.append([m, metric, round(est[j], 6), round(lo, 6), round(hi, 6)])
            r = reps.loc[reps["method"] == m, metric]
            rows.append([m, f"{metric}_splits", round(r.mean(), 6), round(r.min(), 6), round(r.max(), 6)])
        lo, hi = cluster_ci(y, c, icao, rng_c)
        rows.append([m, "balanced_accuracy_cluster", round(est[0], 6), round(lo, 6), round(hi, 6)])
        et = bal(y[thin], c[thin])
        lo, hi = cluster_ci(y[thin], c[thin], icao[thin], rng_c)
        rows.append([m, "balanced_accuracy_thinned", round(et[0], 6), round(lo, 6), round(hi, 6)])
        rows.append([m, "hit_rate_thinned", round(et[1], 6), "", ""])
        rows.append([m, "false_alarm_thinned", round(et[2], 6), "", ""])
    airports = per_airport(d.loc[s], held.loc[s], y)
    paired = paired_airports(airports, rng_c)
    for name, score in ((n, d[n]) for n in SINGLE):
        rows.append([name, "auc", round(auc(y, score.loc[s].to_numpy()), 6), "", ""])
    for grp, mask in (("FEW_SCT", d["label"] == "partial"), ("NSC_CAVOK", d["code"].isin(["NSC", "CAVOK"]))):
        for m in held.columns:
            rows.append([m, f"called_cloudy_{grp}", round(held.loc[mask, m].mean(), 6) if mask.any() else "", "", ""])
    # Ahn et al. (2015) basis: FEW/SCT reports counted as cloudy, on the same report codes (own seeded generator, so
    # the intervals above are unchanged)
    ahn = (d["label"].isin(["clear", "cloudy", "partial"]) & d["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])).to_numpy()
    ya = d.loc[ahn, "label"].isin(["cloudy", "partial"]).to_numpy()
    rng_a = np.random.default_rng(SEED)
    for m in held.columns:
        c = held.loc[ahn, m].to_numpy()
        ea = bal(ya, c)
        lo, hi = cluster_ci(ya, c, d.loc[ahn, "icao"].to_numpy(), rng_a)
        rows.append([m, "balanced_accuracy_fewsct_cloudy", round(ea[0], 6), round(lo, 6), round(hi, 6)])
        rows.append([m, "hit_rate_fewsct_cloudy", round(ea[1], 6), "", ""])
        rows.append([m, "false_alarm_fewsct_cloudy", round(ea[2], 6), "", ""])

    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "metric", "value", "ci_lo", "ci_hi"])
        w.writerows(rows)
    with OUT_THR.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "spectral", "variability", "kx", "kv", *(f"k_{n}_alone" for n in SINGLE)])
        w.writerows([[r[0], r[1], r[2]] + [round(float(x), 4) for x in r[3:]] for r in thr_rows])
    reps.to_csv(ROOT / "results" / "classify_metar_repeats.csv", index=False)
    airports.to_csv(ROOT / "results" / "classify_metar_airports.csv", index=False)
    paired.to_csv(ROOT / "results" / "classify_metar_paired.csv", index=False)
    d.join(held.add_prefix("call_")).to_csv(SCORED, index=False, compression="gzip")
    t = pd.DataFrame(rows, columns=["method", "metric", "value", "lo", "hi"])
    lines = [f"photometers {d.loc[s, 'id'].nunique()} at {d.loc[s, 'icao'].nunique()} airports, reports scored {len(s)} "
             f"(cloudy {int(y.sum())}, clear {int((~y).sum())}), nights {d.loc[s, ['id', 'night']].drop_duplicates().shape[0]} "
             f"(local solar date); {len(uids)} photometers paired, {len(uids) - d.loc[s, 'id'].nunique()} with no scored report",
             f"held-out (grouped 5-fold by airport); 95% night-block bootstrap on the primary split, then the airport-cluster "
             f"interval in parentheses; mean and range over {REPEATS} splits"]
    for m in held.columns:
        g = t[t["method"] == m].set_index("metric")
        lines.append(f"{m:13s} BA {g.loc['balanced_accuracy','value']:.3f} [{g.loc['balanced_accuracy','lo']:.3f}-"
                     f"{g.loc['balanced_accuracy','hi']:.3f}] ({g.loc['balanced_accuracy_cluster','lo']:.3f}-"
                     f"{g.loc['balanced_accuracy_cluster','hi']:.3f})  splits {g.loc['balanced_accuracy_splits','value']:.3f} "
                     f"({g.loc['balanced_accuracy_splits','lo']:.3f}-{g.loc['balanced_accuracy_splits','hi']:.3f})  "
                     f"hit {g.loc['hit_rate','value']:.3f}  "
                     f"FA {g.loc['false_alarm','value']:.3f}  FEW/SCT cloudy {g.loc['called_cloudy_FEW_SCT','value']}")
    lines.append("AUC: " + ", ".join(f"{r.method} {r.value:.3f}" for r in t[t["metric"] == "auc"].itertuples()))
    lines.append(f"thinned to one report per {THIN} slot per airport: reports {int(thin.sum())} (cloudy {int(y[thin].sum())}); "
                 + "; ".join(f"{m} BA {t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_thinned'), 'value'].iloc[0]:.3f} "
                             f"({t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_thinned'), 'lo'].iloc[0]:.3f}-"
                             f"{t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_thinned'), 'hi'].iloc[0]:.3f}) "
                             f"hit {t.loc[(t['method'] == m) & (t['metric'] == 'hit_rate_thinned'), 'value'].iloc[0]:.3f} "
                             f"FA {t.loc[(t['method'] == m) & (t['metric'] == 'false_alarm_thinned'), 'value'].iloc[0]:.3f}"
                             for m in ("two_test", "z_brunt", "neg_dT", "dashboard")))
    lines.append(f"FEW/SCT counted cloudy (the basis of Ahn et al. 2015): reports {int(ahn.sum())} (cloudy {int(ya.sum())}); "
                 + "; ".join(f"{m} BA {t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_fewsct_cloudy'), 'value'].iloc[0]:.3f} "
                             f"({t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_fewsct_cloudy'), 'lo'].iloc[0]:.3f}-"
                             f"{t.loc[(t['method'] == m) & (t['metric'] == 'balanced_accuracy_fewsct_cloudy'), 'hi'].iloc[0]:.3f}) "
                             f"hit {t.loc[(t['method'] == m) & (t['metric'] == 'hit_rate_fewsct_cloudy'), 'value'].iloc[0]:.3f} "
                             f"FA {t.loc[(t['method'] == m) & (t['metric'] == 'false_alarm_fewsct_cloudy'), 'value'].iloc[0]:.3f}"
                             for m in ("two_test", "z_brunt", "dashboard")))
    ok = airports[(airports["clear"] >= MIN_EACH) & (airports["cloudy"] >= MIN_EACH)]
    top = airports.iloc[0]
    lines.append("paired by airport (two-test minus method, airports with >= 100 of each class): " + "; ".join(
        f"{r.method} higher at {r.two_test_higher} of {r.airports}, mean {r.mean_diff:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f}), median {r.median_diff:+.3f}"
        for r in paired.itertuples()))
    lines.append(f"per airport ({len(ok)} of {len(airports)} with >= {MIN_EACH} clear and cloudy reports): two_test mean "
                 f"{ok['two_test'].mean():.3f}, median {ok['two_test'].median():.3f}, min {ok['two_test'].min():.3f} "
                 f"({ok.loc[ok['two_test'].idxmin(), 'icao']}), max {ok['two_test'].max():.3f} ({ok.loc[ok['two_test'].idxmax(), 'icao']}); "
                 f"dashboard mean {ok['dashboard'].mean():.3f}, median {ok['dashboard'].median():.3f}; largest share of the scored "
                 f"reports {top['icao']} {top['share']:.3f} (median interval {top['interval_min']:.0f} min, {int(top['photometers'])} photometers)")
    x, v, kz, kv = final["two_test"]
    lines.append(f"final two-test: {x} > {kz:.3f} or {v} > {kv:.3f}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
