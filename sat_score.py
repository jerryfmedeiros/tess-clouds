"""sat_score.py — the frozen classifier against the satellite cloud masks (frozen before the satellite
scoring on 2026-10-30; Sect. 3.5 of the paper). Nothing here is tuned: the thresholds are read from
data/frozen_thresholds.csv (the frozen copy of row "all" of results/classify_metar_thresholds.csv, which
classify_metar.py refits on every run) and checked against the frozen values.

References (goes_extract.py, msg_extract.py, viirs_extract.py):
  goes    GOES-East ABI ACMF, binary mask over the 3x3 box (good DQF only), time = scan start
  msg     Meteosat 0 deg cloud mask over the 3x3 box (classes 0/1 clear, 2 cloudy, 3 excluded), time = sensing end
  viirs   NOAA-20 CLDMSK_L2 over the 7x7 box (codes 0-1 cloudy, 2-3 clear), time = granule start; where a granule
          was reprocessed, the latest version is kept
A box counts only with every pixel valid (9, or 49). Satellite clear = no cloudy pixel; cloudy = at least
CLOUDY_9 of 9 (CLOUDY_49 of 49); otherwise partial. View zenith angle below MAX_VZA (GOES, Meteosat; the VIIRS
extraction keeps pixels within 2 km of the photometer, and the swath edge is below 70 deg).

Photometer side: the 5-minute bins whose centres are within MATCH of the satellite time (at least one), reduced as
in classify_metar.py (medians; the largest vis15; a reduction mostly at the firmware floor sets every spectral
score to -inf, clear). Only reductions where the classifier is defined (z_brunt finite or floor) are scored, and
every method is scored on the same ones. Photometers: those of the record (qc_baseline.with_baseline: a clear-sky
baseline fitted, not excluded by data/excluded_ir.csv, the placeholder positions of data/excluded_positions.csv or
the obstructed units of qc_ir.py).

Subsets: primary = snow-free (ERA5-Land snow depth <= SNOW_M) and not mountain; snow = snow depth > SNOW_M, not
mountain; mountain = MOUNTAIN ids and any header elevation >= MOUNTAIN_M, snow-free; metar = the primary rows of
the photometers scored against METAR. Scores: balanced accuracy, hit and false-alarm rates of each method on
clear vs cloudy references, with a 95% night-block bootstrap within photometers (B, SEED; a night is the local
solar date at its start, common.local_night) and a 95% interval from a bootstrap that resamples whole photometers (common.cluster_ci: the between-site spread, columns ci_lo_phot and
ci_hi_phot); the share called cloudy of partial references; and, per photometer with at least MIN_EACH clear and
cloudy references, the balanced accuracy distribution.

METAR cross-checks (held-out METAR calls from data/metar/scored.csv.gz; a report is matched to the geostationary
rows (goes, msg) in its 30-minute window, and its satellite state is clear if every matched row is clear, cloudy
if any is cloudy):
  high cloud  for CLR/SKC/NCD reports, the share the satellite calls cloudy, and the classifier's cloudy share on
              those reports by satellite state
  CAVOK/NSC   these reports counted clear where the satellite is clear, scored with the cloudy (BKN/OVC/VV)
              reports of the same photometers; the photometers left out of the METAR score for lack of clear
              reports reported as their own row (also written to the csv as reference "metar", subsets cavok_all
              and cavok_left, for the tables)

Writes data/sat/<ref>.csv.gz (matched rows), results/sat_score.csv, results/sat_score_sites.csv,
results/sat_score.txt.

Correction after the frozen run (--dem): 340 of the 406 archive headers give 0 m, so the header rule found 9 of the
38 photometers at or above MOUNTAIN_M. With --dem the mountain set is MOUNTAIN plus every photometer whose
elevation in data/elevation.csv (fetch_elevation.py, ASTER GDEM) is at or above MOUNTAIN_M; the matched rows of
the frozen run are rescored unchanged, and the outputs carry the suffix _dem.

Also added after the frozen run: the subset primary_qc = primary without the photometers whose clear-sky baseline
SD lies beyond the network's Tukey fence (qc_baseline.py), reported beside primary. --rescore rescores the matched
rows without matching again.

Also added after the frozen run, with --dem only and from a second generator (SEED + 1), so the rows
above keep their draws: (1) the photometer-paired interval of the two-test minus the dashboard rule and minus the
baseline test alone on the primary rows (common.paired_cluster_ci; subset rows "paired" in the csv); (2) the months
that fail the obstruction rule month by month (qc_ir.py, results/qc_ir_months.csv): the satellite's cloudy share and
the classifier's cloudy share in those months against the rest, and the primary and primary_qc rows scored without
them (subsets primary_months, primary_qc_months); (3) the high-cloud cross-check on snow-free reports at photometers
that are not mountain sites, the sample of sat_vs_metar.py, with the largest airport's share.

Usage: python sat_score.py [--workers 12] [--refs goes msg viirs] [--dem] [--rescore]"""
from __future__ import annotations

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import cluster_ci, local_night, longitudes, night_block_ci, paired_cluster_ci
from qc_baseline import weak, with_baseline
from qc_ir import obstructed_months

ROOT = Path(__file__).parent
FEAT = ROOT / "data" / "features"
SAT = ROOT / "data" / "sat"
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT_CSV = ROOT / "results" / "sat_score.csv"
OUT_SITES = ROOT / "results" / "sat_score_sites.csv"
OUT_TXT = ROOT / "results" / "sat_score.txt"
REFS = ("goes", "msg", "viirs")
CLOUDY_9, CLOUDY_49 = 7, 40
MAX_VZA = 70.0
MATCH = pd.Timedelta(minutes=5)
SNOW_M = 0.01
MOUNTAIN = {"stars211", "stars426", "stars427", "stars507", "stars90"}
MOUNTAIN_M = 1500.0
SEED, B = 20261030, 1000
MIN_EACH = 100
WINDOW = pd.Timedelta(minutes=30)
FROZEN = {"kx": 2.449, "kv": 0.3605, "z_brunt": 2.147, "neg_dT": -18.560, "vis15": 0.110}
ALARCON = 54.5
SPECTRAL = ("z_brunt", "z_linear", "sky_med", "neg_dT")
COLS = ("z_brunt", "z_linear", "sky_med", "neg_dT", "s5_excess", "dT", "floor", "sde_m")
METHODS = ("two_test", "z_brunt", "neg_dT", "vis15", "dashboard", "alarcon", "always_clear")


def thresholds() -> dict:
    r = next(r for r in csv.DictReader(l for l in (ROOT / "data" / "frozen_thresholds.csv").open() if not l.startswith("#")))
    th = {"kx": float(r["kx"]), "kv": float(r["kv"]), "z_brunt": float(r["k_z_brunt_alone"]),
          "neg_dT": float(r["k_neg_dT_alone"]), "vis15": float(r["k_vis15_alone"])}
    for k, v in FROZEN.items():
        assert abs(th[k] - v) < 6e-4, f"threshold {k} {th[k]} differs from the frozen {v}"
    return th


def load_ref(ref: str) -> pd.DataFrame:
    """Satellite rows with a fully valid box: id, time, frac (cloudy fraction), label."""
    frames = []
    for f in sorted(p for p in (ROOT / "data" / ref).iterdir() if p.name.endswith((".csv.gz", ".csv"))):
        if ref == "goes":
            d = pd.read_csv(f, usecols=["scan_start", "id", "vza", "bcm_3", "n_3"])
            d = d[(d["n_3"] == 9) & (d["vza"] < MAX_VZA)].rename(columns={"scan_start": "time", "bcm_3": "frac"})
        elif ref == "msg":
            d = pd.read_csv(f, usecols=["sensing_end", "id", "vza", "cloudy_3", "n_3"])
            d = d[(d["n_3"] == 9) & (d["vza"] < MAX_VZA)].rename(columns={"sensing_end": "time", "cloudy_3": "frac"})
        else:
            d = pd.read_csv(f, usecols=["granule_start", "id", "cloudy_7", "n_7", "granule"])
            d = d[d["n_7"] == 49].sort_values("granule").drop_duplicates(["granule_start", "id"], keep="last")
            d = d.rename(columns={"granule_start": "time", "cloudy_7": "frac"})
        frames.append(d[["id", "time", "frac"]])
    d = pd.concat(frames, ignore_index=True)
    d["time"] = pd.to_datetime(d["time"], format="ISO8601")
    n = 49 if ref == "viirs" else 9
    k = np.rint(pd.to_numeric(d["frac"], errors="coerce").to_numpy(float) * n)                   # cloudy pixels (fractions are rounded to 3 decimals)
    d = d[np.isfinite(k)].copy()
    k = k[np.isfinite(k)]
    d["label"] = np.where(k == 0, "clear", np.where(k >= (CLOUDY_49 if n == 49 else CLOUDY_9), "cloudy", "partial"))
    return d.drop_duplicates(["id", "time"]).sort_values(["id", "time"]).reset_index(drop=True)


def reduce(f: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """The bins within MATCH of each time t, reduced as in classify_metar.per_report."""
    bins = f["time"].to_numpy()
    lo = np.searchsorted(bins, t - MATCH.to_timedelta64(), side="left")
    hi = np.searchsorted(bins, t + MATCH.to_timedelta64(), side="right")
    vals = f[list(COLS)].to_numpy(float)
    vis = f["vis15"].to_numpy(float)
    k = int((hi - lo).max()) if len(t) else 0
    stack = np.full((len(t), max(k, 1), len(COLS)), np.nan)
    vstack = np.full((len(t), max(k, 1)), np.nan)
    for j in range(k):
        idx = lo + j
        ok = idx < hi
        stack[ok, j] = vals[idx[ok]]
        vstack[ok, j] = vis[idx[ok]]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=1)
        vmax = np.nanmax(vstack, axis=1)
    o = pd.DataFrame(med, columns=list(COLS))
    o["vis15"] = vmax
    o["n_bins"] = hi - lo
    o.loc[o["floor"] >= 0.5, list(SPECTRAL)] = -np.inf
    return o


def calls(d: pd.DataFrame, th: dict) -> pd.DataFrame:
    c = pd.DataFrame(index=d.index)
    c["two_test"] = (d["z_brunt"].to_numpy() > th["kx"]) | (d["s5_excess"].to_numpy() > th["kv"])
    for name in ("z_brunt", "neg_dT", "vis15"):
        c[name] = d[name].to_numpy() > th[name]
    c["dashboard"] = (100 - 3 * d["dT"]) > 55
    c["alarcon"] = ~(d["dT"] > ALARCON)
    c["always_clear"] = False
    return c


def match(args) -> pd.DataFrame:
    uid, ref, th = args
    p = SAT / "by_id" / ref / f"{uid}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_csv(p, parse_dates=["time"])
    f = pd.read_csv(FEAT / f"{uid}.csv.gz", parse_dates=["time"])
    f["neg_dT"] = -f["dT"]
    o = reduce(f, s["time"].to_numpy())
    ok = (o["n_bins"] >= 1) & (np.isfinite(o["z_brunt"]) | np.isneginf(o["z_brunt"])) & o["sky_med"].notna()
    o, s = o[ok].reset_index(drop=True), s[ok.to_numpy()].reset_index(drop=True)
    if o.empty:
        return pd.DataFrame()
    c = calls(o, th).add_prefix("call_")
    out = pd.concat([s[["id", "time", "frac", "label"]], o[["sde_m", "floor", "n_bins"]], c], axis=1)
    out["night"] = local_night(out["time"], longitudes()[uid])
    return out


def elevations() -> dict:
    """Header elevation of the latest monthly file of each photometer (0 where the header gives none)."""
    out = {}
    for r in csv.DictReader(SELECTED.open()):
        files = sorted((ROOT / "data" / "tess_archive" / r["id"]).glob("*.dat"))
        z = 0.0
        for f in files[-1:]:
            with f.open(errors="replace") as fh:
                for line in fh:
                    if line.startswith("# Position:"):
                        p = [x.strip() for x in line.split(":", 1)[1].split(",")]
                        try:
                            z = float(p[2]) if len(p) > 2 else 0.0
                        except ValueError:
                            z = 0.0
                        break
        out[r["id"]] = z
    return out


def rates(y: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    hit = c[y].mean() if y.any() else np.nan
    fa = c[~y].mean() if (~y).any() else np.nan
    return (hit + 1 - fa) / 2, hit, fa


def score(g: pd.DataFrame, rng: np.random.Generator) -> list[list]:
    """Rows per method: references, cloudy, balanced accuracy [95%], hit, false alarm, partial share called cloudy."""
    part = g[g["label"] == "partial"]
    g = g[g["label"] != "partial"].reset_index(drop=True)
    if g.empty:
        return []
    y = (g["label"] == "cloudy").to_numpy()
    key = (g["id"] + "|" + g["night"]).to_numpy()
    blk, inv = np.unique(key, return_inverse=True)
    unit = pd.Series(blk).str.split("|").str[0].to_numpy()
    order = np.argsort(unit, kind="stable")
    _, start, size = np.unique(unit[order], return_index=True, return_counts=True)
    owner_start = np.repeat(start, size)
    owner_size = np.repeat(size, size)
    rows = []
    for m in METHODS:
        c = g[f"call_{m}"].to_numpy(bool)
        k = np.zeros((len(blk), 4))                           # per night block: hits, cloudy, false alarms, clear
        np.add.at(k, (inv, 0), c & y)
        np.add.at(k, (inv, 1), y)
        np.add.at(k, (inv, 2), c & ~y)
        np.add.at(k, (inv, 3), ~y)
        k = k[order]
        bs = np.empty(B)
        for b in range(B):
            idx = owner_start + (rng.random(len(blk)) * owner_size).astype(int)
            s = k[idx].sum(axis=0)
            bs[b] = (s[0] / s[1] + 1 - s[2] / s[3]) / 2
        ba, hit, fa = rates(y, c)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        plo, phi = cluster_ci(y, c, g["id"].to_numpy(), rng, B)
        ps = part[f"call_{m}"].mean() if len(part) else np.nan
        rows.append([m, len(g), int(y.sum()), round(ba, 6), round(lo, 6), round(hi, 6), round(plo, 6), round(phi, 6),
                     round(hit, 6), round(fa, 6), len(part), round(ps, 6) if np.isfinite(ps) else ""])
    return rows


def sites(g: pd.DataFrame) -> list[list]:
    g = g[g["label"] != "partial"]
    out = []
    for uid, h in g.groupby("id"):
        y = (h["label"] == "cloudy").to_numpy()
        if y.sum() < MIN_EACH or (~y).sum() < MIN_EACH:
            continue
        out.append([uid, int((~y).sum()), int(y.sum())] + [round(rates(y, h[f"call_{m}"].to_numpy(bool))[0], 4)
                                                          for m in METHODS])
    return out


def metar_geo(m: dict) -> pd.DataFrame:
    """Held-out METAR calls with the geostationary state in each report's 30-minute window (column sat)."""
    sc = pd.read_csv(ROOT / "data" / "metar" / "scored.csv.gz", parse_dates=["time"])
    geo = pd.concat([m[r][["id", "time", "label"]] for r in ("goes", "msg") if r in m], ignore_index=True)
    geo = geo.sort_values(["id", "time"])
    state = np.full(len(sc), "", dtype=object)
    for uid, h in sc.groupby("id"):
        s = geo[geo["id"] == uid]
        if s.empty:
            continue
        t = s["time"].to_numpy()
        lab = s["label"].to_numpy()
        lo = np.searchsorted(t, (h["time"] - WINDOW).to_numpy(), side="right")
        hi = np.searchsorted(t, h["time"].to_numpy(), side="right")
        for i, a, b in zip(h.index, lo, hi):
            if b > a:
                L = lab[a:b]
                state[i] = "cloudy" if (L == "cloudy").any() else "clear" if (L == "clear").all() else "partial"
    sc["sat"] = state
    return sc


def later_checks(matched: dict, mountain: set, weak_ids: set) -> tuple[list[str], list[list]]:
    """The later checks of the module docstring; own generator."""
    rng = np.random.default_rng(SEED + 1)
    bad = obstructed_months()
    lines, rows = ["\nAdded after the frozen run (second generator, SEED + 1)"], []
    lines.append("  paired by photometer, primary rows: pooled two-test balanced accuracy minus the method's [95% interval]")
    prim = {}
    for ref, m in matched.items():
        g = m[m["sde_m"].notna() & (m["sde_m"] <= SNOW_M) & ~m["id"].isin(mountain)]
        prim[ref] = g
        g = g[g["label"] != "partial"]
        y = (g["label"] == "cloudy").to_numpy()
        parts = []
        for other in ("dashboard", "z_brunt"):
            d, lo, hi = paired_cluster_ci(y, g["call_two_test"].to_numpy(bool), g[f"call_{other}"].to_numpy(bool),
                                          g["id"].to_numpy(), rng, B)
            rows.append([ref, "paired", g["id"].nunique(), f"two_test_minus_{other}", len(g), int(y.sum()),
                         round(d, 6), round(lo, 6), round(hi, 6), "", "", "", "", 0, ""])
            parts.append(f"{other} {d:+.3f} [{lo:+.3f} to {hi:+.3f}]")
        lines.append(f"    {ref:6s} photometers {g['id'].nunique():4d}  " + "  ".join(parts))
    lines.append("  obstructed months (qc_ir.py month by month): primary rows in flagged months against the rest")
    for ref, g in prim.items():
        ym = g["time"].dt.strftime("%Y-%m")
        flag = pd.Series([(u, x) in bad for u, x in zip(g["id"], ym)], index=g.index)
        lab = g[g["label"] != "partial"]
        f2 = flag.loc[lab.index]
        def shares(h):
            return (h["label"] == "cloudy").mean(), h["call_two_test"].mean()
        a_sat, a_cls = shares(lab[f2])
        b_sat, b_cls = shares(lab[~f2])
        lines.append(f"    {ref:6s} flagged rows {int(f2.sum())} ({f2.mean():.4f}) at {lab.loc[f2, 'id'].nunique()} photometers "
                     f"(weak-flagged {len(set(lab.loc[f2, 'id']) & weak_ids)}): satellite cloudy {a_sat:.3f} vs {b_sat:.3f}, "
                     f"classifier cloudy {a_cls:.3f} vs {b_cls:.3f}")
        for name, sel in (("primary_months", ~flag), ("primary_qc_months", ~flag & ~g["id"].isin(weak_ids))):
            h = g[sel]
            res = score(h, rng)
            for r in res:
                rows.append([ref, name, h["id"].nunique(), *r])
            t = res[0]
            lines.append(f"      {name:17s} photometers {h['id'].nunique():4d}  two-test {t[3]:.3f} [{t[4]:.3f}-{t[5]:.3f}] "
                         f"({t[6]:.3f}-{t[7]:.3f})  hit {t[8]:.3f}  false alarm {t[9]:.3f}")
    if {"goes", "msg"} & set(matched):
        sc = metar_geo(matched)
        clr = sc[sc["code"].isin(["CLR", "SKC", "NCD"]) & (sc["sat"] != "")]
        low = clr[(clr["sde_m"] <= SNOW_M) & ~clr["id"].isin(mountain)]
        for name, h0 in (("all CLR/SKC/NCD reports", clr), ("snow-free, not mountain (as sat_vs_metar.py)", low)):
            lines.append(f"  high cloud, {name}: {len(h0)} reports at {h0['id'].nunique()} photometers")
            for st in ("clear", "partial", "cloudy"):
                h = h0[h0["sat"] == st]
                if len(h):
                    top = h["icao"].value_counts(normalize=True)
                    lines.append(f"    satellite {st:7s} n={len(h):6d}: classifier cloudy share {h['call_two_test'].mean():.3f}; "
                                 f"largest airport {top.index[0]} {top.iloc[0]:.3f}")
    return lines, rows


def metar_checks(m: dict, rng: np.random.Generator) -> tuple[list[str], list[list]]:
    sc = metar_geo(m)
    lines = ["\nMETAR cross-checks (geostationary rows in the report's 30-minute window; held-out METAR calls)"]
    clr = sc[sc["code"].isin(["CLR", "SKC", "NCD"]) & (sc["sat"] != "")]
    lines.append(f"  high cloud: {len(clr)} CLR/SKC/NCD reports with a satellite match; satellite clear "
                 f"{(clr['sat'] == 'clear').mean():.3f}, partial {(clr['sat'] == 'partial').mean():.3f}, "
                 f"cloudy {(clr['sat'] == 'cloudy').mean():.3f}")
    for st in ("clear", "partial", "cloudy"):
        h = clr[clr["sat"] == st]
        if len(h):
            lines.append(f"    satellite {st:7s} n={len(h):6d}: classifier cloudy share {h['call_two_test'].mean():.3f}")
    cav = sc[sc["code"].isin(["NSC", "CAVOK"]) & (sc["sat"] == "clear")].assign(label="clear")
    cld = sc[sc["label"] == "cloudy"]
    few = set(pd.read_csv(ROOT / "results" / "breakdown_metar_sites.csv")["id"])
    scored_ids = set(sc["id"])
    left = sorted(scored_ids - few)
    rows = []
    for subset, name, ids in (("cavok_all", "all METAR photometers", scored_ids),
                              ("cavok_left", f"photometers without enough clear reports ({', '.join(left)})", set(left))):
        g = pd.concat([cav[cav["id"].isin(ids)], cld[cld["id"].isin(ids)]], ignore_index=True)
        if g.empty or not (g["label"] == "clear").any():
            continue
        y = (g["label"] == "cloudy").to_numpy()
        for meth in METHODS:
            c = g[f"call_{meth}"].to_numpy(bool)
            ba, hit, fa = rates(y, c)
            lo, hi = night_block_ci(y, c, g["id"].to_numpy(), g["night"].to_numpy(), rng, B)
            plo, phi = cluster_ci(y, c, g["id"].to_numpy(), rng, B)
            rows.append(["metar", subset, g["id"].nunique(), meth, len(g), int(y.sum()), round(ba, 6), round(lo, 6), round(hi, 6),
                         round(plo, 6), round(phi, 6), round(hit, 6), round(fa, 6), 0, ""])
            if meth == "two_test":
                lines.append(f"  CAVOK/NSC with a clear satellite as clear vs BKN/OVC/VV, {name}: clear {int((~y).sum())}, "
                             f"cloudy {int(y.sum())}; two-test balanced accuracy {ba:.3f} [{lo:.3f}-{hi:.3f}] ({plo:.3f}-{phi:.3f}) "
                             f"(hit {hit:.3f}, false alarm {fa:.3f})")
    return lines, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--refs", nargs="*", default=list(REFS))
    ap.add_argument("--dem", action="store_true", help="mountain sites from the DEM; rescore the matched rows")
    ap.add_argument("--rescore", action="store_true", help="rescore the matched rows of the last run")
    a = ap.parse_args()
    global OUT_CSV, OUT_SITES, OUT_TXT
    if a.dem:
        OUT_CSV, OUT_SITES, OUT_TXT = (p.with_name(p.stem + "_dem" + p.suffix) for p in (OUT_CSV, OUT_SITES, OUT_TXT))
    th = thresholds()
    ids = sorted(with_baseline() & {f.name.split(".")[0] for f in FEAT.glob("*.csv.gz")})
    elev = elevations()
    if a.dem:
        dem = pd.read_csv(ROOT / "data" / "elevation.csv")
        elev = dict(zip(dem["id"], dem["dem_m"].fillna(0.0)))
    mountain = MOUNTAIN | {u for u, z in elev.items() if z >= MOUNTAIN_M}
    weak_ids = weak()
    metar_ids = set(pd.read_csv(ROOT / "data" / "metar" / "scored.csv.gz", usecols=["id"])["id"])
    matched = {}
    for ref in a.refs:
        out = SAT / f"{ref}.csv.gz"
        if a.dem or a.rescore:
            mm = pd.read_csv(out, parse_dates=["time"])
            lons = longitudes()
            mm["night"] = pd.concat([local_night(h["time"], lons[u]) for u, h in mm.groupby("id", sort=False)]).loc[mm.index]
            matched[ref] = mm
            continue
        r = load_ref(ref)
        r = r[r["id"].isin(ids)]
        d = SAT / "by_id" / ref
        d.mkdir(parents=True, exist_ok=True)
        for uid, g in r.groupby("id"):
            g[["time", "frac", "label"]].assign(id=uid).to_csv(d / f"{uid}.csv.gz", index=False, compression="gzip")
        with ProcessPoolExecutor(a.workers) as ex:
            parts = [x for x in ex.map(match, [(u, ref, th) for u in ids]) if len(x)]
        m = pd.concat(parts, ignore_index=True)
        m.to_csv(out, index=False, compression="gzip")
        matched[ref] = m
        print(ref, len(r), "satellite rows,", len(m), "matched", flush=True)
    rng = np.random.default_rng(SEED)
    rows, site_rows = [], []
    lines = [f"photometers {len(ids)}; mountain {len(mountain & set(ids))} ({', '.join(sorted(mountain & set(ids)))}); "
             f"thresholds: two-test z > {th['kx']} or s5_excess > {th['kv']} (frozen); intervals: 95% night-block within "
             f"photometers [..] then photometer-cluster (..); weak baseline (qc_baseline.py) "
             f"{len(weak_ids & set(ids))}; photometers with at least one matched scan "
             f"{len(set().union(*(set(m['id']) for m in matched.values())))}"]
    for ref, m in matched.items():
        snow = m["sde_m"] > SNOW_M
        known = m["sde_m"].notna()
        mnt = m["id"].isin(mountain)
        subsets = {"primary": known & ~snow & ~mnt, "primary_qc": known & ~snow & ~mnt & ~m["id"].isin(weak_ids),
                   "snow": snow & ~mnt, "mountain": known & ~snow & mnt,
                   "metar": known & ~snow & ~mnt & m["id"].isin(metar_ids)}
        lines.append(f"\n{ref}: {len(m)} matched rows, {m['id'].nunique()} photometers ({int((~known).sum())} without "
                     f"snow depth, left out)")
        for name, sel in subsets.items():
            g = m[sel]
            res = score(g, rng)
            for r in res:
                rows.append([ref, name, g["id"].nunique(), *r])
            if res:
                lines.append(f"  {name:9s} photometers {g['id'].nunique():4d}  references {res[0][1]:8d} (cloudy {res[0][2]})")
                for r in res:
                    lines.append(f"    {r[0]:12s} {r[3]:.3f} [{r[4]:.3f}-{r[5]:.3f}] ({r[6]:.3f}-{r[7]:.3f})  hit {r[8]:.3f}  "
                                 f"false alarm {r[9]:.3f}  partial called cloudy {r[11]}")
            if name == "primary":
                s = sites(g)
                site_rows += [[ref, *x] for x in s]
                if s:
                    p = pd.DataFrame(s, columns=["id", "clear", "cloudy", *METHODS])
                    lines.append(f"    per photometer ({len(p)} with >= {MIN_EACH} clear and cloudy): two-test median "
                                 f"{p['two_test'].median():.3f}, 10th pct {p['two_test'].quantile(.1):.3f}, min "
                                 f"{p['two_test'].min():.3f} ({p.loc[p['two_test'].idxmin(), 'id']})")
    if {"goes", "msg"} & set(matched):
        more, extra = metar_checks(matched, rng)
        lines += more
        rows += extra
    if a.dem:
        more, extra = later_checks(matched, mountain, weak_ids)
        lines += more
        rows += extra
    pd.DataFrame(rows, columns=["reference", "subset", "photometers", "method", "references", "cloudy", "balanced_accuracy",
                                "ci_lo", "ci_hi", "ci_lo_phot", "ci_hi_phot", "hit", "false_alarm", "partial",
                                "partial_called_cloudy"]).to_csv(OUT_CSV, index=False)
    pd.DataFrame(site_rows, columns=["reference", "id", "clear", "cloudy", *METHODS]).to_csv(OUT_SITES, index=False)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
