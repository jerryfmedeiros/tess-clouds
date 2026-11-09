"""viirs_why.py — is the VIIRS cloud the classifier misses at mountain photometers real thin cloud or a mask error?
(exploratory, after cloud_top.py, 2026-11-03)

At the Canary photometers about half of the VIIRS cloudy scans are called clear by the classifier, with cloud tops
reported 0.5-2 km and more above the site (results/cloud_top.txt), so the cloud sea below the summit does not explain
them. Two readings, and what each predicts, fixed before looking:
  thin cloud   real cloud too thin for the sensor: the mask is confident, the 11 um effective emissivity is lower
               than in the scans the classifier also calls cloudy (agreed), the air at the reported top is moist, and
               the photometer's baseline residual sits above that of satellite-clear scans (a signal below threshold).
  mask error   no cloud: the mask is uncertain rather than confident, the air at the reported top is dry, and the
               photometer reads as it does under a satellite-clear sky.
For the VIIRS rows of data/cloudtop/viirs.csv (missed and agreed, Canary and other mountain photometers), from the
kept CLDPROP_L2_VIIRS_NOAA20 granule, over the 7 x 7 box of cloud_top.py:
  conf        share of box pixels the mask calls confident cloudy (Cloud_Mask byte 0, bits 2-1 = 00); uncertain (01)
              is the other cloudy-side value
  emis        median Cloud_Effective_Emissivity (CLAVR-x AWG, 11 um) of the pixels with a cloud-top height
  ctp_hPa     median Cloud_Top_Pressure of those pixels
  phase       most common Cloud_Phase_Cloud_Top_Properties of those pixels (1 water, 2 ice, 3 mixed, 6 undetermined)
From the full ERA5 column of ARCO-ERA5 (37 levels, the hour nearest the scan, the nearest 0.25 deg point; the
files of fetch_mountain_met.py stop at 650 hPa over the Canaries, so a first run had no value for higher tops):
rh_top, the relative humidity (IFS mixed phase, arco_check.rh_ifs) at ctp_hPa, log-pressure interpolation; and, added
after that first run, ERA5's own cloud: cc_top, the highest fraction_of_cloud_cover on the levels within CC_HPA of
ctp_hPa, and cc_above, the highest on any level above the site's standard-atmosphere pressure. From data/features (the 5-minute bin nearest the scan, within 5 min): z_brunt and
s5_excess, the classifier's two inputs. The same photometer inputs for a seeded sample of N_CLEAR satellite-clear
VIIRS rows per region from results/mountain_below.csv.gz.
Per-row reading, fixed before looking: thin when conf >= 0.5 and rh_top >= RH_MOIST; mask error when conf < 0.5 and
rh_top < RH_DRY; otherwise unresolved.

Needs the raw granules and the ARCO store (not a verify target). The box statistics are cached in
data/viirs_why_box.csv and the columns in data/viirs_why_era5.csv.gz (resumable). Writes results/viirs_why.csv and results/viirs_why.txt.

Usage: python viirs_why.py [--keep RAW] [--workers 8]"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import arco_check as arco
import goes_extract as ge
from cloud_top import ROWS as BELOW, viirs_pixel
from fetch_mountain_met import p_std

RAW = Path(os.environ.get("TESS_CLOUDS_RAW", Path(__file__).parent / "data" / "raw"))   # raw files, outside git

ROOT = Path(__file__).parent
ROWS = ROOT / "data" / "cloudtop" / "viirs.csv"
CLEAR = BELOW
OUT_CSV = ROOT / "results" / "viirs_why.csv"
OUT_TXT = ROOT / "results" / "viirs_why.txt"
CACHE = ROOT / "data" / "viirs_why_era5.csv.gz"
BOX = ROOT / "data" / "viirs_why_box.csv"
VARS = {"t": "temperature", "q": "specific_humidity", "cc": "fraction_of_cloud_cover"}
CC_HPA = 50.0
N_CLEAR = 400
SEED = 20261103
RH_MOIST, RH_DRY = 70.0, 50.0


def granule(job: tuple) -> list:
    """Box statistics for every row read from one granule."""
    path, rows = job
    out = []
    with netCDF4.Dataset(path) as nc:
        geo, phys = nc.groups["geolocation_data"], nc.groups["geophysical_data"]
        glat, glon = np.asarray(geo["latitude"][:], float), np.asarray(geo["longitude"][:], float)
        for r in rows:
            px = viirs_pixel(glat, glon, r["lat"], r["lon"])
            if px is None:
                out.append({**r, "conf": np.nan, "uncertain": np.nan, "emis": np.nan, "ctp_hPa": np.nan, "phase": np.nan})
                continue
            iy, ix = px
            sl = (slice(max(0, iy - 3), iy + 4), slice(max(0, ix - 3), ix + 4))
            mask = np.asarray(phys["Cloud_Mask"][sl[0], sl[1], 0]).astype(np.uint8)
            fov = (mask >> 1) & 3
            det = (mask & 1) == 1
            cth = np.ma.asarray(phys["Cloud_Top_Height"][sl]).astype(float).filled(np.nan)
            ok = np.isfinite(cth) & (cth > -9999)
            emis = np.ma.asarray(phys["Cloud_Effective_Emissivity"][sl]).astype(float).filled(np.nan)[ok]
            ctp = np.ma.asarray(phys["Cloud_Top_Pressure"][sl]).astype(float).filled(np.nan)[ok]
            ph = np.ma.asarray(phys["Cloud_Phase_Cloud_Top_Properties"][sl]).astype(float).filled(np.nan)[ok]
            ph = ph[np.isfinite(ph) & (ph > 0)]
            out.append({**r, "conf": float(np.mean(fov[det] == 0)) if det.any() else np.nan,
                        "uncertain": float(np.mean(fov[det] == 1)) if det.any() else np.nan,
                        "emis": float(np.nanmedian(emis)) if np.isfinite(emis).any() else np.nan,
                        "ctp_hPa": float(np.nanmedian(ctp)) if np.isfinite(ctp).any() else np.nan,
                        "phase": float(pd.Series(ph).mode().iloc[0]) if len(ph) else np.nan})
    return out


def columns(points: pd.DataFrame, workers: int) -> pd.DataFrame:
    """ERA5 t, q and cloud fraction on the 37 levels at each (id, hour) of points, from the ARCO store; cached."""
    m, ax = arco.meta()
    shape = m["temperature/.zarray"]["shape"]
    have = read_cache() if CACHE.exists() else pd.DataFrame(columns=["id", "hour"])
    done = set(zip(have["id"], have["hour"]))
    todo = points[[(u, h) not in done for u, h in zip(points["id"], points["hour"])]]

    def one(hour, g):
        i = int((hour - pd.Timestamp("1900-01-01")) / pd.Timedelta("1h")) - arco.EPOCH_HOURS
        yi = [int(np.abs(ax["latitude"] - y).argmin()) for y in g["lat"]]
        xi = [int(np.abs(ax["longitude"] - x % 360).argmin()) for x in g["lon"]]
        cols = {k: arco.field(v, i, shape)[:, yi, xi] for k, v in VARS.items()}
        return pd.DataFrame([{"id": u, "hour": hour, "level": float(lv), **{k: float(cols[k][li, n]) for k in VARS}}
                             for n, u in enumerate(g["id"]) for li, lv in enumerate(ax["level"])])

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, h, g) for h, g in todo.groupby("hour")]
        for k, f in enumerate(as_completed(futs), 1):
            part = f.result()
            part.to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False, compression="gzip")
            if k % 50 == 0:
                print(f"era5 columns: {k}/{len(futs)}", flush=True)
    return read_cache()


def read_cache() -> pd.DataFrame:
    c = pd.read_csv(CACHE)
    c["hour"] = pd.to_datetime(c["hour"], format="mixed")       # midnight hours are written without the time
    return c


def era5_at(col: pd.DataFrame, ctp: float, dem_m: float) -> tuple[float, float, float]:
    """rh_top, cc_top and cc_above for one row from its column (levels in hPa)."""
    col = col.sort_values("level")
    lev = col["level"].to_numpy(float)
    rh = arco.rh_ifs(col["t"].to_numpy(float), col["q"].to_numpy(float), lev * 100)
    cc = col["cc"].to_numpy(float)
    above = cc[lev < p_std(dem_m)]
    cc_above = float(above.max()) if len(above) else np.nan
    if not np.isfinite(ctp):
        return np.nan, np.nan, cc_above
    near = cc[np.abs(lev - ctp) <= CC_HPA]
    return (float(np.interp(np.log(ctp), np.log(lev), rh)), float(near.max()) if len(near) else np.nan, cc_above)


def photometer(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for uid, g in rows.groupby("id"):
        f = pd.read_csv(ROOT / "data" / "features" / f"{uid}.csv.gz", usecols=["time", "z_brunt", "s5_excess"],
                        parse_dates=["time"]).sort_values("time")
        parts.append(pd.merge_asof(g.sort_values("time"), f, on="time", direction="nearest",
                                   tolerance=pd.Timedelta("5min")))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, default=RAW)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    pos = {r["id"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(ge.SELECTED.open())}
    d = pd.read_csv(ROWS, parse_dates=["time"]).dropna(subset=["file"])
    d["lat"] = [pos[u][0] for u in d["id"]]
    d["lon"] = [pos[u][1] for u in d["id"]]
    jobs = [(a.keep / "cloudtop" / "viirs" / f, g[["id", "time", "group", "canary", "dem_m", "top_m", "lat", "lon"]]
             .to_dict("records")) for f, g in d.groupby("file")]
    if BOX.exists():
        sat = pd.read_csv(BOX, parse_dates=["time"])
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            sat = pd.DataFrame([r for part in ex.map(granule, jobs) for r in part])
        sat.to_csv(BOX, index=False)
    sat["hour"] = sat["time"].dt.round("h")
    col = columns(sat[["id", "hour", "lat", "lon"]].drop_duplicates(["id", "hour"]), a.workers)
    by = {k: g for k, g in col.groupby(["id", "hour"])}
    vals = [era5_at(by[(r.id, r.hour)], r.ctp_hPa, r.dem_m) for r in sat.itertuples()]
    sat["rh_top"], sat["cc_top"], sat["cc_above"] = zip(*vals)

    c = pd.read_csv(CLEAR, parse_dates=["time"])
    c = c[(c["ref"] == "viirs") & (c["group"] == "clear") & c["id"].isin(set(d["id"]))]
    rng = np.random.default_rng(SEED)
    c = pd.concat(g.iloc[np.sort(rng.choice(len(g), size=min(N_CLEAR, len(g)), replace=False))]
                  for _, g in c.groupby("canary"))[["id", "time", "group", "canary"]]
    rows = photometer(pd.concat([sat, c], ignore_index=True))
    rows["reading"] = np.select([(rows["conf"] >= 0.5) & (rows["rh_top"] >= RH_MOIST),
                                 (rows["conf"] < 0.5) & (rows["rh_top"] < RH_DRY)], ["thin", "mask_error"], "unresolved")
    rows.loc[rows["group"] == "clear", "reading"] = ""
    rows.round(4).to_csv(OUT_CSV, index=False)

    lines = [f"VIIRS rows of data/cloudtop/viirs.csv and satellite-clear rows; box = 7 x 7 pixels; readings fixed before "
             f"looking: thin = conf >= 0.5 and rh_top >= {RH_MOIST:.0f}%, mask_error = conf < 0.5 and rh_top < "
             f"{RH_DRY:.0f}%"]
    for region, sel in (("Canary", True), ("other", False)):
        x = rows[rows["canary"] == sel]
        lines.append(f"{region}: photometers {x['id'].nunique()}")
        for grp in ("missed", "agreed", "clear"):
            y = x[x["group"] == grp]
            if y.empty:
                continue
            line = f"  {grp:7s} n {len(y):4d}  z_brunt median {y['z_brunt'].median():+.2f} share > 1 " \
                   f"{(y['z_brunt'] > 1).mean():.3f}  s5_excess median {y['s5_excess'].median():+.3f}"
            if grp != "clear":
                ph = y["phase"].value_counts(normalize=True)
                line += (f"\n           conf median {y['conf'].median():.2f} share >= 0.5 {(y['conf'] >= 0.5).mean():.3f}  "
                         f"uncertain median {y['uncertain'].median():.2f}  emis median {y['emis'].median():.2f} "
                         f"share < 0.5 {(y['emis'] < 0.5).mean():.3f}  ctp median {y['ctp_hPa'].median():.0f} hPa"
                         f"\n           rh_top median {y['rh_top'].median():.1f} (with value {y['rh_top'].notna().mean():.3f})  "
                         f"phase water {ph.get(1.0, 0):.2f} ice {ph.get(2.0, 0):.2f} mixed {ph.get(3.0, 0):.2f} "
                         f"undetermined {ph.get(6.0, 0):.2f}"
                         f"\n           ERA5 cc_top median {y['cc_top'].median():.2f} share >= 0.1 {(y['cc_top'] >= 0.1).mean():.3f}  "
                         f"cc_above share >= 0.1 {(y['cc_above'] >= 0.1).mean():.3f}"
                         f"\n           reading thin {(y['reading'] == 'thin').mean():.3f}  mask_error "
                         f"{(y['reading'] == 'mask_error').mean():.3f}  unresolved {(y['reading'] == 'unresolved').mean():.3f}")
            lines.append(line)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
