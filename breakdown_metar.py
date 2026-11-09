"""breakdown_metar.py — where the classifier works and where it fails, against automated METAR: the held-out calls
of classify_metar.py (data/metar/scored.csv.gz) split by condition.

Splits (clear vs cloudy reports, as in classify_metar.py):
  air       ERA5-Land 2 m air temperature at the photometer (C): the sensor heater holds the die near 11 C when
            it is colder, so the die stops following the air; the infrared contrast of cloud also changes with
            temperature and humidity.
  moon      Moon above or below the horizon.
  snow      ERA5-Land snow depth above SNOW_M or not.
  site      per photometer (balanced accuracy distribution over photometers with at least MIN_EACH clear and
            MIN_EACH cloudy reports; airports that report a clear night sky as CAVOK or NSC rather than NCD leave
            few scorable clear reports), and the same photometers by distance to the airport (DIST_KM bands,
            results/metar_labels.csv). vis15 in METHODS is the brightness-stability test, whose Moon-up and
            Moon-down scores Sect. 4.1 quotes.
  cavok     airports that code more than half of their no-layer night reports as CAVOK or NSC (results/metar_labels.csv),
            where the scored clear class is the NCD remnant made under reduced visibility or weather, against the
            other airports.
Cloud base (cloudy reports only, so the hit rate): the lowest reported layer, in bands up to and beyond the
12,000 ft range of the US ASOS ceilometer (Wagner & Kleiss 2016); the European sensors reach higher, but the
bands are the same everywhere. A clear report may still hide cloud above the ceilometer's reach; those false
alarms are checked against the satellite masks, not here.

Each cell gives reports, balanced accuracy (or hit rate), hit and false-alarm rates, with a 95% night-block
bootstrap interval within photometers (B, seeded; common.night_block_ci) for the balanced accuracy, and the airport
with the largest share of the cell's reports with that share (KJAC, Jackson Hole, reports every 5 min and holds
most of the cold-air and snow reports).

Also an airport-cluster interval for every cell (columns ci_lo_air, ci_hi_air; common.cluster_ci over airports, from
a second generator, SEED + 1, so the night-block draws do not depend on it); two more splits, scored after the others: season (local month at the photometer, Dec-Feb winter in the
north and summer in the south) and relative humidity (from the ERA5-Land air and dew-point temperatures of the
night bin nearest the report, data/nights, Magnus form over water; RH bands); and the Spearman rank correlation of
the per-photometer score with the distance to the airport, across photometers and within airports (score and
distance less their airport means, airports with two or more photometers).

Writes results/breakdown_metar.csv and results/breakdown_metar.txt.

Usage: python breakdown_metar.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from common import cluster_ci, night_block_ci

ROOT = Path(__file__).parent
SCORED = ROOT / "data" / "metar" / "scored.csv.gz"
OUT_CSV = ROOT / "results" / "breakdown_metar.csv"
OUT_TXT = ROOT / "results" / "breakdown_metar.txt"
METHODS = ("two_test", "z_brunt", "neg_dT", "dashboard", "vis15")
SEED, B = 20261030, 1000
SNOW_M = 0.01
MIN_EACH = 100
AIR = [-np.inf, -10, 0, 10, 20, np.inf]
DIST_KM = [0, 7, 14, 21]
BASE = [0, 2000, 6500, 12000, 20000, np.inf]
RH = [0, 70, 85, 95, np.inf]
NIGHTS = ROOT / "data" / "nights"
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
NORTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
         6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
FLIP = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}


def rates(y: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    hit = c[y].mean() if y.any() else np.nan
    fa = c[~y].mean() if (~y).any() else np.nan
    return (hit + 1 - fa) / 2, hit, fa


def cell(g: pd.DataFrame, m: str, rng: np.random.Generator, rng_air: np.random.Generator, hit_only: bool = False) -> list:
    y = (g["label"] == "cloudy").to_numpy()
    c = g[f"call_{m}"].to_numpy(bool)
    ba, hit, fa = rates(y, c)
    est = hit if hit_only else ba
    stat = "hit" if hit_only else "ba"
    lo, hi = night_block_ci(y, c, g["id"].to_numpy(), g["night"].to_numpy(), rng, B, stat)
    alo, ahi = cluster_ci(y, c, g["icao"].to_numpy(), rng_air, B, stat)
    share = g["icao"].value_counts(normalize=True)
    return [len(g), int(y.sum()), round(est, 6), round(lo, 6), round(hi, 6), round(hit, 6),
            round(fa, 6) if np.isfinite(fa) else "", share.index[0], round(float(share.iloc[0]), 6),
            round(alo, 6), round(ahi, 6)]


def relative_humidity(d: pd.DataFrame) -> np.ndarray:
    """RH (%) of each report from the night bin of data/nights nearest in time (within 10 min)."""
    out = pd.Series(np.nan, index=d.index)
    for uid, h in d.groupby("id"):
        n = pd.read_csv(NIGHTS / f"{uid}.csv.gz", usecols=["time", "t2m_C", "d2m_C"], parse_dates=["time"]).dropna()
        n = n.sort_values("time")
        h2 = h[["time"]].reset_index().sort_values("time")
        j = pd.merge_asof(h2, n, on="time", direction="nearest", tolerance=pd.Timedelta(minutes=10))
        es = lambda t: np.exp(17.625 * t / (243.04 + t))
        out.loc[j["index"].to_numpy()] = (100 * es(j["d2m_C"]) / es(j["t2m_C"])).to_numpy()
    return out.to_numpy()


def main():
    d = pd.read_csv(SCORED)
    d = d[d["label"].isin(["clear", "cloudy"]) & d["code"].isin(["LAYERS", "CLR", "SKC", "NCD"])].copy()
    d["air"] = pd.cut(d["t2m_C"], AIR, labels=["< -10", "-10 to 0", "0 to 10", "10 to 20", "> 20"])
    d["moon"] = np.where(d["moon_up"] >= 0.5, "up", "down")
    d["snow"] = np.where(d["sde_m"] > SNOW_M, "snow", "no snow")
    lab = pd.read_csv(ROOT / "results" / "metar_labels.csv").groupby("icao")[["CLR", "SKC", "NCD", "NSC", "CAVOK"]].sum()
    cavok_airports = set(lab.index[(lab["NSC"] + lab["CAVOK"]) > 0.5 * lab.sum(axis=1)])
    d["cavok"] = np.where(d["icao"].isin(cavok_airports), "CAVOK/NSC airports", "other airports")
    d["base"] = pd.cut(d["base_ft"], BASE, right=False,
                       labels=["< 2000", "2000-6500", "6500-12000", "12000-20000", ">= 20000"])
    rng = np.random.default_rng(SEED)
    rng_air = np.random.default_rng(SEED + 1)
    rows = []
    for split in ("air", "moon", "snow", "cavok"):
        for level, g in d.groupby(split, observed=True):
            for m in METHODS:
                rows.append([split, str(level), m, "balanced_accuracy", *cell(g, m, rng, rng_air)])
    cl = d[d["label"] == "cloudy"]
    for level, g in cl.groupby("base", observed=True):
        for m in METHODS:
            rows.append(["cloud_base_ft", str(level), m, "hit_rate", *cell(g, m, rng, rng_air, hit_only=True)])
    pos = pd.read_csv(SELECTED).set_index("id")
    d["time"] = pd.to_datetime(d["time"])
    month = (d["time"] + pd.to_timedelta(d["id"].map(pos["lon"]) / 15.0, unit="h")).dt.month
    north = d["id"].map(pos["lat"]) >= 0
    season = month.map(NORTH)
    d["season"] = pd.Categorical(np.where(north, season, season.map(FLIP)), ["winter", "spring", "summer", "autumn"])
    d["rh"] = pd.cut(relative_humidity(d), RH, right=False, labels=["< 70", "70-85", "85-95", ">= 95"])
    for split in ("season", "rh"):
        for level, g in d.groupby(split, observed=True):
            for m in METHODS:
                rows.append([split, str(level), m, "balanced_accuracy", *cell(g, m, rng, rng_air)])
    per = []
    for uid, g in d.groupby("id"):
        y = (g["label"] == "cloudy").to_numpy()
        per.append([uid, int((~y).sum()), int(y.sum())] + [rates(y, g[f"call_{m}"].to_numpy(bool))[0] for m in METHODS])
    per = pd.DataFrame(per, columns=["id", "clear", "cloudy", *METHODS])
    few = per[(per["clear"] < MIN_EACH) | (per["cloudy"] < MIN_EACH)]
    per = per.drop(few.index)
    lab_id = pd.read_csv(ROOT / "results" / "metar_labels.csv").set_index("id")
    per["km"] = per["id"].map(lab_id["km"])
    per["icao"] = per["id"].map(lab_id["icao"])
    rho, p = spearmanr(per["km"], per["two_test"])
    multi = per[per.groupby("icao")["id"].transform("size") >= 2]
    wr, wp = spearmanr(multi["km"] - multi.groupby("icao")["km"].transform("mean"),
                       multi["two_test"] - multi.groupby("icao")["two_test"].transform("mean"))
    per["band"] = pd.cut(per["km"], DIST_KM, right=False, labels=[f"{a}-{b} km" for a, b in zip(DIST_KM[:-1], DIST_KM[1:])])
    dist = per.groupby("band", observed=True).agg(photometers=("id", "size"), **{m: (m, "median") for m in METHODS}).reset_index()
    dist.round(6).to_csv(ROOT / "results" / "breakdown_metar_distance.csv", index=False)
    cols = ["split", "level", "method", "metric", "reports", "cloudy", "value", "ci_lo", "ci_hi", "hit", "false_alarm",
            "top_airport", "top_share", "ci_lo_air", "ci_hi_air"]
    t = pd.DataFrame(rows, columns=cols)
    OUT_CSV.parent.mkdir(exist_ok=True)
    t.to_csv(OUT_CSV, index=False)
    per.drop(columns="icao").round(6).to_csv(ROOT / "results" / "breakdown_metar_sites.csv", index=False)
    lines = []
    for split, g in t.groupby("split", sort=False):
        lines.append(f"\n{split} ({g['metric'].iloc[0]} [night-block] (airport-cluster); hit / false alarm)")
        for level, h in g.groupby("level", sort=False):
            n = h["reports"].iloc[0]
            lines.append(f"  {level:12s} n={n:7d}  " + "  ".join(
                f"{r.method} {r.value:.3f} [{r.ci_lo:.3f}-{r.ci_hi:.3f}] ({r.ci_lo_air:.3f}-{r.ci_hi_air:.3f}) "
                f"({r.hit:.2f}/{r.false_alarm})" for r in h.itertuples())
                         + f"  largest airport {h['top_airport'].iloc[0]} {h['top_share'].iloc[0]:.2f}")
    lines.append(f"\nper photometer balanced accuracy ({len(per)} photometers; {len(few)} with fewer than {MIN_EACH} "
                 f"clear or cloudy reports left out: {', '.join(few['id'])}): " + "; ".join(
        f"{m} median {per[m].median():.3f}, 10th pct {per[m].quantile(.1):.3f}, min {per[m].min():.3f} ({per.loc[per[m].idxmin(), 'id']})"
        for m in METHODS))
    lines.append("per photometer by distance to the airport (median balanced accuracy): " + "; ".join(
        f"{r.band} n={r.photometers} " + " ".join(f"{m} {getattr(r, m):.3f}" for m in METHODS) for r in dist.itertuples()))
    lines.append(f"two-test score against distance, Spearman: across {len(per)} photometers rho {rho:+.3f} (p {p:.3f}); "
                 f"within airports ({len(multi)} photometers at {multi['icao'].nunique()} airports with two or more) "
                 f"rho {wr:+.3f} (p {wp:.3f})")
    OUT_TXT.write_text("\n".join(lines).lstrip() + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
