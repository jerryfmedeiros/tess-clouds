"""nsb_selection.py — does the cloud flag pick clear nights for night-sky brightness better than a selection on
brightness stability and a die minus sky threshold? An application of the frozen classifier (thresholds frozen on 2026-10-30, data/frozen_thresholds.csv), with
automated METAR as the check; nothing here is tuned.

Every minute reading of the METAR-paired photometers (metar_labels.py) with the Sun below SUN_MAX and the Moon
below MOON_MAX, in a 5-minute bin where the classifier is defined, goes through four selections:
  stable    the stability test (after Alarcon et al. 2021): standard deviation of the magnitudes over the preceding
            10 min below 0.010 mag, at least 5 readings
  threshold stable and sky minus enclosure at or below a threshold set for the photometer and season (Dec-Feb,
            Mar-May, Jun-Aug, Sep-Nov) from its own stable minutes (ir_threshold below): the sky minus enclosure
            temperature is binned in IR_BIN steps (bins of at least IR_MIN_BIN minutes) and the threshold is
            IR_CAP unless, between IR_SEARCH, a bin's median magnitude jumps by more than IR_JUMP from the previous
            bin and the next bin confirms it, in which case it is that bin's lower edge; where the infrared channel
            is missing or flat, or the coldest 5 % of stable minutes never reach IR_CAP, the stability test alone.
            The no-read (0.00 C) and error (-273.15 C) codes are not temperatures. Thresholds are written to
            results/nsb_thresholds.csv
  flag      stable and the bin called clear by the two-test classifier (z_brunt <= kz and s5_excess <= kv; a bin
            at the firmware floor is clear). The thresholds were learned on these METAR reports, so each photometer
            uses those of its held-out fold in the primary split of classify_metar.py (folds grouped by airport,
            seed SEED), learned without its airport; the frozen final thresholds (KZ, KV) are listed for reference
  both      threshold and flag
Each night (local solar date at its start) with at least MIN_READINGS selected readings after 3-sigma clipping
gives a night median magnitude.

Check against METAR: a reading is covered by the automated report whose 30-minute window holds it. Covered
readings count as clear (CLR, SKC, NCD), partial (FEW, SCT) or cloudy (BKN, OVC, VV); NSC and CAVOK are left out
(cloud above 5000 ft possible). A selected night is contaminated when at least half of its covered readings are
cloudy. Reference nights: stable readings with none covered by a partial or cloudy report and at least half
covered by clear reports (CLR, SKC, NCD), at least MIN_READINGS of them; the reference sky of a photometer is the median of its reference night medians (at least MIN_REF nights).
Per photometer and selection: nights, contaminated share, and the offset of the median night from the reference
sky (mag; negative = brighter). Nights kept by threshold and removed by the flag are reported on their own.
Photometers are grouped by the reference sky: bright (< 19), intermediate, dark (>= 20.5 mag/arcsec2).
The pooled share of covered selected readings under cloudy reports is
given too. Pooled contaminated shares carry a 95% night-block bootstrap interval (B, seeded).
The comparisons between selections (contaminated share of flag over
stable, flag minus threshold, and nights kept by flag over threshold) carry a 95% photometer-cluster bootstrap
interval (B_CLUSTER resamples of whole photometers, own generator seeded SEED, so the lines above are unchanged).

Writes results/nsb_selection.csv (per photometer and selection) and results/nsb_selection.txt.

Usage: python nsb_selection.py [--workers 12]"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from classify_metar import FOLDS
from qc_ir import obstructed

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "data" / "tess_archive"
NIGHTS = ROOT / "data" / "nights"
FEATURES = ROOT / "data" / "features"
LABELS = ROOT / "data" / "metar" / "labels"
OUT_THR = ROOT / "results" / "nsb_thresholds.csv"
OUT_CSV = ROOT / "results" / "nsb_selection.csv"
OUT_TXT = ROOT / "results" / "nsb_selection.txt"
SUN_MAX, MOON_MAX = -18.0, -5.0
SIGMA_MAX, SIGMA_WINDOW, SIGMA_MIN_N = 0.010, "10min", 5
MAG_RANGE = (10.0, 23.0)
IR_BIN, IR_JUMP, IR_MIN_BIN, IR_CAP = 2.5, 0.3, 200, -20.0
IR_SEARCH = (-35.0, -20.0)
KZ, KV = 2.449, 0.3605                        # frozen final thresholds (not used: see held_out)
MIN_READINGS, MIN_REF = 30, 20
WINDOW = pd.Timedelta(minutes=30)
CLEAR_CODES = ("CLR", "SKC", "NCD")
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
METHODS = ("stable", "threshold", "flag", "both")
CLASSES = [(-np.inf, 19.0, "bright"), (19.0, 20.5, "intermediate"), (20.5, np.inf, "dark")]
SEED, B = 20261030, 300
B_CLUSTER = 2000


def readings(uid: str) -> pd.DataFrame:
    """Minute readings of an identifier; parts filed while no position was set (key ending -N) are left out."""
    frames = []
    for f in sorted((ARCHIVE / uid).glob("*.dat")):
        if f.stem.split("_")[-1].startswith("-"):
            continue
        frames.append(pd.read_csv(f, sep=";", comment="#", header=None, on_bad_lines="skip", usecols=[0, 2, 3, 5],
                                  names=["utc", "t_box", "t_sky", "mag"], dtype=str, engine="c"))
    d = pd.concat(frames, ignore_index=True)
    d["utc"] = pd.to_datetime(d["utc"], errors="coerce", format="ISO8601")
    for c in ("t_box", "t_sky", "mag"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["utc", "mag"])
    d = d[(d["mag"] > MAG_RANGE[0]) & (d["mag"] < MAG_RANGE[1])]
    return d.drop_duplicates("utc").sort_values("utc").reset_index(drop=True)


def ir_threshold(dT: np.ndarray, mag: np.ndarray):
    """Threshold (C) for one photometer-season from its stable minutes, the rule it took, and the minutes used."""
    ok = np.isfinite(dT)
    if ok.mean() < 0.5 or np.std(dT[ok]) < 1.0:
        return None, "infrared missing or flat: stability only"
    dT, mag = dT[ok], mag[ok]
    if np.quantile(dT, 0.05) > IR_CAP:
        return None, "never below the cap: stability only"
    edges = np.arange(np.floor(dT.min() / IR_BIN) * IR_BIN, IR_CAP + IR_BIN, IR_BIN)
    bins = []
    for e in edges:
        sel = (dT > e) & (dT <= e + IR_BIN)
        if sel.sum() >= IR_MIN_BIN:
            bins.append((float(e), float(np.median(mag[sel]))))
    for k in range(1, len(bins) - 1):
        e, med = bins[k]
        if not (IR_SEARCH[0] <= e < IR_SEARCH[1]):
            continue
        d1, d2 = med - bins[k - 1][1], bins[k + 1][1] - bins[k - 1][1]
        if abs(d1) > IR_JUMP and abs(d2) > IR_JUMP and np.sign(d1) == np.sign(d2):
            return e, "jump"
    return IR_CAP, "cap"


def held_out() -> dict:
    """Two-test thresholds (kz, kv) per airport, from the fold that held it out in the primary split."""
    airports = sorted(pd.read_csv(ROOT / "data" / "metar" / "scored.csv.gz", usecols=["icao"])["icao"].unique())
    fold_of = dict(zip(airports, np.random.default_rng(SEED).permutation(np.arange(len(airports)) % FOLDS)))
    t = pd.read_csv(ROOT / "results" / "classify_metar_thresholds.csv")
    t = t.set_index(t["fold"].astype(str))
    return {a: (float(t.loc[str(k), "kx"]), float(t.loc[str(k), "kv"])) for a, k in fold_of.items()}


def night_stats(g: pd.DataFrame) -> float | None:
    if len(g) < MIN_READINGS:
        return None
    med, sd = g["mag"].median(), g["mag"].std()
    if sd > 0:
        g = g[np.abs(g["mag"] - med) <= 3 * sd]
    return g["mag"].median() if len(g) >= MIN_READINGS else None


def one(uid: str, th: dict) -> pd.DataFrame:
    d = readings(uid)
    d["time"] = d["utc"].dt.floor("5min") + pd.Timedelta(minutes=2.5)
    n = pd.read_csv(NIGHTS / f"{uid}.csv.gz", usecols=["time", "sun_alt", "moon_alt"], parse_dates=["time"])
    f = pd.read_csv(FEATURES / f"{uid}.csv.gz", usecols=["time", "z_brunt", "s5_excess", "floor"], parse_dates=["time"])
    d = d.merge(n, on="time").merge(f, on="time")
    d = d[(d["sun_alt"] < SUN_MAX) & (d["moon_alt"] < MOON_MAX)].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    roll = d.set_index("utc")["mag"].rolling(SIGMA_WINDOW)
    d["stable"] = (roll.std().to_numpy() < SIGMA_MAX) & (roll.count().to_numpy() >= SIGMA_MIN_N)
    t_sky = d["t_sky"].where(~d["t_sky"].isin([0.0, -273.15]))
    d["dT"] = t_sky - d["t_box"]
    d["season"] = d["utc"].dt.month.map(SEASONS)
    thr, thr_rows = {}, []
    for season, g in d[d["stable"]].groupby("season"):
        t, rule = ir_threshold(g["dT"].to_numpy(float), g["mag"].to_numpy(float))
        thr[season] = np.inf if t is None else t
        thr_rows.append([uid, season, "" if t is None else t, len(g), rule])
    z = d["z_brunt"].where(d["floor"] == 0, -np.inf)
    d = d[np.isfinite(z) | np.isneginf(z)].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    z, stable = d["z_brunt"].where(d["floor"] == 0, -np.inf), d["stable"].to_numpy()
    lim = d["season"].map(lambda s: thr.get(s, np.inf)).astype(float)
    ir_ok = np.isinf(lim) | (d["dT"] <= lim)
    lab = pd.read_csv(LABELS / f"{uid}.csv.gz", usecols=["icao", "time", "code", "label"], parse_dates=["time"])
    lab = lab.sort_values("time").reset_index(drop=True)
    if lab.empty or lab["icao"].iloc[0] not in th:
        return pd.DataFrame()
    kz, kv = th[lab["icao"].iloc[0]]
    clear = ~((z > kz) | (d["s5_excess"] > kv))
    sel = {"stable": stable, "threshold": stable & ir_ok.to_numpy(), "flag": stable & clear.to_numpy()}
    sel["both"] = sel["threshold"] & sel["flag"]
    rt = lab["time"].to_numpy()
    j = np.searchsorted(rt, d["utc"].to_numpy(), side="left")
    jj = np.minimum(j, len(rt) - 1)
    covered = (j < len(rt)) & ((rt[jj] - d["utc"].to_numpy()) < WINDOW.to_timedelta64())
    code, label = lab["code"].to_numpy()[jj], lab["label"].to_numpy()[jj]
    strict = covered & np.isin(code, (*CLEAR_CODES, "LAYERS"))
    d["ref"] = np.where(strict, label, "")
    lon = float(pd.read_csv(ARCHIVE / "selected.csv").set_index("id").loc[uid, "lon"])
    d["night"] = (d["utc"] + pd.to_timedelta(lon / 15.0, unit="h") - pd.Timedelta(hours=12)).dt.date
    rows = []
    for m in (*METHODS, "reference"):
        s = sel["stable"] if m == "reference" else sel[m]
        for night, g in d[s].groupby("night"):
            if m == "reference" and ((g["ref"].isin(["partial", "cloudy"])).any()
                                     or (g["ref"] == "clear").sum() < 0.5 * len(g)):
                continue
            med = night_stats(g)
            if med is None:
                continue
            r = g["ref"][g["ref"] != ""]
            rows.append([uid, m, night, med, len(g), len(r), int((r == "cloudy").sum()), int((r == "partial").sum()),
                         int((r == "clear").sum())])
    for night, g in d[sel["threshold"] & ~sel["flag"]].groupby("night"):
        r = g["ref"][g["ref"] != ""]
        rows.append([uid, "removed", night, g["mag"].median(), len(g), len(r), int((r == "cloudy").sum()),
                     int((r == "partial").sum()), int((r == "clear").sum())])
    print(uid, len(d), flush=True)
    out = pd.DataFrame(rows, columns=["id", "method", "night", "mag", "readings", "covered", "cloudy", "partial",
                                      "clear"])
    out.attrs["thresholds"] = thr_rows
    return out


def boot_share(g: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    x = g["contaminated"].to_numpy(float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def cluster_lines(pool: pd.DataFrame) -> list[str]:
    """Selections compared on the same photometers, with a 95% photometer-cluster bootstrap interval."""
    t = pool.groupby(["id", "method"])["contaminated"].agg(["sum", "size"]).unstack("method").fillna(0)

    def stats(tt):
        k, m = tt["sum"].sum(), tt["size"].sum()
        s = k / m
        return np.array([s["flag"] / s["stable"], s["flag"] - s["threshold"], m["flag"] / m["threshold"]])

    rng = np.random.default_rng(SEED)
    ids = t.index.to_numpy()
    bs = np.array([stats(t.loc[rng.choice(ids, len(ids))]) for _ in range(B_CLUSTER)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pt = stats(t)
    sh = t["sum"] / t["size"]
    names = ("contaminated share, flag / stable", "contaminated share, flag - threshold", "nights, flag / threshold")
    out = [f"\nselections compared over {len(ids)} photometers [95% photometer-cluster interval, {B_CLUSTER} resamples]"]
    out += [f"  {n:38s} {pt[i]:+.4f} [{lo[i]:+.4f} to {hi[i]:+.4f}]" for i, n in enumerate(names)]
    out.append(f"  photometers with a lower contaminated share under flag than stable {int((sh['flag'] < sh['stable']).sum())}, "
               f"than threshold {int((sh['flag'] < sh['threshold']).sum())}, higher than threshold "
               f"{int((sh['flag'] > sh['threshold']).sum())}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    skip = obstructed()
    ids = sorted(f.name.split(".")[0] for f in LABELS.glob("*.csv.gz") if f.name.split(".")[0] not in skip)
    th = held_out()
    with ProcessPoolExecutor(a.workers) as ex:
        parts = [x for x in ex.map(one, ids, [th] * len(ids)) if len(x)]
    nights = pd.concat(parts, ignore_index=True)
    thr = pd.DataFrame([r for x in parts for r in x.attrs.get("thresholds", [])],
                       columns=["id", "season", "ir_threshold_C", "stable_minutes", "rule"])
    OUT_THR.parent.mkdir(exist_ok=True)
    thr.to_csv(OUT_THR, index=False)
    nights.to_csv(ROOT / "data" / "metar" / "nsb_nights.csv.gz", index=False, compression="gzip")
    nights["contaminated"] = (nights["covered"] > 0) & (nights["cloudy"] >= 0.5 * nights["covered"])
    ref = nights[nights["method"] == "reference"].groupby("id")["mag"].agg(["median", "size"])
    ref = ref[ref["size"] >= MIN_REF]
    per = []
    for (uid, m), g in nights[nights["id"].isin(ref.index)].groupby(["id", "method"]):
        cov = g[g["covered"] > 0]
        per.append([uid, m, len(g), len(cov), round(cov["contaminated"].mean(), 6) if len(cov) else np.nan,
                    round(g["mag"].median() - ref.loc[uid, "median"], 6), round(ref.loc[uid, "median"], 6)])
    per = pd.DataFrame(per, columns=["id", "method", "nights", "covered_nights", "contaminated_share",
                                     "offset_mag", "reference_mag"])
    per["class"] = pd.cut(per["reference_mag"], [c[0] for c in CLASSES] + [np.inf], right=False,
                          labels=[c[2] for c in CLASSES])
    OUT_CSV.parent.mkdir(exist_ok=True)
    per.to_csv(OUT_CSV, index=False)
    rng = np.random.default_rng(SEED)
    pool = nights[nights["id"].isin(ref.index) & (nights["covered"] > 0)]
    lines = [f"{len(ref)} photometers with at least {MIN_REF} reference nights "
             f"({len(ids) - len(ref)} of {len(ids)} paired photometers have fewer)",
             "\npooled over covered nights: nights, contaminated share [95%], nights relative to threshold"]
    base = (pool["method"] == "threshold").sum()
    for m in (*METHODS, "removed"):
        g = pool[pool["method"] == m]
        lo, hi = boot_share(g, rng)
        lines.append(f"  {m:8s} {len(g):7d}  {g['contaminated'].mean():.4f} [{lo:.4f}-{hi:.4f}]  {len(g) / base:.3f}"
                     f"   readings under cloudy reports {g['cloudy'].sum() / g['covered'].sum():.4f}")
    lines.append("\nmedian over photometers of the offset of the median night from the reference sky (mag; "
                 "negative = brighter) [10th-90th pct], and of the contaminated share")
    for cls, h in per.groupby("class", observed=True):
        lines.append(f"  {cls} ({h['id'].nunique()} photometers)")
        for m in (*METHODS, "removed"):
            k = h[h["method"] == m]
            if len(k):
                lines.append(f"    {m:8s} offset {k['offset_mag'].median():+.3f} [{k['offset_mag'].quantile(.1):+.3f}"
                             f" to {k['offset_mag'].quantile(.9):+.3f}]  contaminated {k['contaminated_share'].median():.3f}"
                             f"  nights {int(k['nights'].sum())}")
    lines += cluster_lines(pool)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
