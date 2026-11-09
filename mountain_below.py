"""mountain_below.py — is the satellite's cloud at mountain photometers often below the photometer? (exploratory)

At mountain photometers (DEM >= MOUNTAIN_M) the classifier misses much of the cloud the satellite masks report. If
the missed cloud lies below the site (the trade-wind cloud sea under the Canary summits, valley fog), the photometer,
looking up from above it, reads a clear sky correctly. Three groups of matched satellite rows (data/sat/<ref>.csv.gz,
snow-free, two-test call of the frozen classifier):
  missed   satellite cloudy, classifier clear
  agreed   satellite cloudy, classifier cloudy
  clear    satellite clear
For each row, from the ERA5 pressure levels of fetch_mountain_met.py --arco at the nearest 0.25 deg point (00/06/12/
18 UTC, the nearest time within 3 h): relative humidity at the site's standard-atmosphere pressure (log-pressure
interpolation), the highest relative humidity on the levels at least BELOW_HPA below the site (higher pressure), and
the temperature difference site minus that level's nearest one (positive: warmer above, an inversion between).
Fixed before looking: a "cloud-below" profile is RH below >= RH_WET and RH at the site < RH_DRY. The prediction is
that missed rows show it far more often than agreed rows, and that at the Canary sites the missed share of
satellite-cloudy rows peaks in summer, when the cloud sea is most frequent.

The group files hold only the levels that bracket the group's sites (lowest
850 hPa), which leaves a cloud sea below about 1.5 km outside the profile; where the files of fetch_mountain_met.py
--arco-canary exist, the Canary photometers use them instead (every level 1000-600 hPa).

Needs the raw ERA5 files (not a verify target). Writes results/mountain_below.csv.gz (per row) and
results/mountain_below.txt.

Usage: python mountain_below.py [--keep RAW]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from fetch_mountain_met import GROUPS, MOUNTAIN_M, YEARS, p_std

RAW = Path(os.environ.get("TESS_CLOUDS_RAW", Path(__file__).parent / "data" / "raw"))   # raw files, outside git

ROOT = Path(__file__).parent
OUT_CSV = ROOT / "results" / "mountain_below.csv.gz"
OUT_TXT = ROOT / "results" / "mountain_below.txt"
BELOW_HPA = 10.0
RH_WET, RH_DRY = 90.0, 70.0
SNOW_M = 0.01
CANARY = (27.0, 29.5, -18.5, -13.0)                 # lat min, lat max, lon min, lon max


def profiles(f: Path, units: pd.DataFrame) -> dict:
    """Per photometer inside the file's box: time, RH at the site, highest RH below, site minus below temperature."""
    out = {}
    with netCDF4.Dataset(f) as nc:
        la, lo = np.asarray(nc["latitude"][:]), np.asarray(nc["longitude"][:])
        inside = [r for r in units.itertuples() if la.min() - 0.125 <= r.lat <= la.max() + 0.125
                  and lo.min() - 0.125 <= r.lon <= lo.max() + 0.125]
        if not inside:
            return out
        lev = np.asarray(nc["pressure_level"][:], float)
        times = pd.to_datetime(np.asarray(nc["valid_time"][:], "int64"), unit="s")
        t_all, r_all = np.asarray(nc["t"][:]), np.asarray(nc["r"][:])
    lp, o = np.log(lev), np.argsort(np.log(lev))
    for r in inside:
        i, j = int(np.abs(la - r.lat).argmin()), int(np.abs(lo - r.lon).argmin())
        p = p_std(r.dem_m)
        below = lev >= p + BELOW_HPA
        if not below.any():
            continue
        t, rh = t_all[:, :, i, j].astype(float), r_all[:, :, i, j].astype(float)
        rh_site = np.array([np.interp(np.log(p), lp[o], row[o]) for row in rh])
        t_site = np.array([np.interp(np.log(p), lp[o], row[o]) for row in t])
        k = int(np.where(below)[0][np.argmin(lev[below])])          # the level just below the site
        out[r.id] = pd.DataFrame({"pl_time": times, "rh_site": rh_site, "rh_below": rh[:, below].max(axis=1),
                                  "dt_inv": t_site - t[:, k]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, default=RAW)
    a = ap.parse_args()
    e = pd.read_csv(ROOT / "data" / "elevation.csv")
    e = e[e["dem_m"] >= MOUNTAIN_M]
    got = {uid: [] for uid in e["id"]}
    in_canary = (e["lat"].between(CANARY[0], CANARY[1]) & e["lon"].between(CANARY[2], CANARY[3])).to_numpy()
    full = [a.keep / "era5pl_arco_canary" / f"canary_{y}.nc" for y in YEARS]
    full = full if all(f.exists() for f in full) else []      # all years or none, never a partial download
    for g, (w, east) in GROUPS.items():
        units = e[(e["lon"] >= w) & (e["lon"] < east) & ~(in_canary & bool(full))]
        for f in sorted((a.keep / "era5pl_arco").glob(f"{g}_*.nc")):
            for uid, s in profiles(f, units).items():
                got[uid].append(s)
    for f in full:                                   # every level 1000-600 hPa at the Canary photometers
        for uid, s in profiles(f, e[in_canary]).items():
            got[uid].append(s)
    prof = {u: pd.concat(v).drop_duplicates("pl_time").sort_values("pl_time") for u, v in got.items() if v}
    rows = []
    for ref in ("goes", "msg", "viirs"):
        s = pd.read_csv(ROOT / "data" / "sat" / f"{ref}.csv.gz", usecols=["id", "time", "label", "sde_m", "call_two_test"],
                        parse_dates=["time"])
        s = s[s["id"].isin(prof) & (s["sde_m"] <= SNOW_M) & s["label"].isin(["clear", "cloudy"])]
        for uid, g in s.groupby("id"):
            m = pd.merge_asof(g.sort_values("time"), prof[uid], left_on="time", right_on="pl_time",
                              direction="nearest", tolerance=pd.Timedelta("3h"))
            m["ref"] = ref
            rows.append(m)
    d = pd.concat(rows, ignore_index=True).dropna(subset=["rh_site"])
    d["group"] = np.where(d["label"] == "clear", "clear", np.where(d["call_two_test"], "agreed", "missed"))
    d["below_profile"] = (d["rh_below"] >= RH_WET) & (d["rh_site"] < RH_DRY)
    pos = e.set_index("id")
    d["canary"] = [CANARY[0] <= pos.loc[u, "lat"] <= CANARY[1] and CANARY[2] <= pos.loc[u, "lon"] <= CANARY[3]
                   for u in d["id"]]
    d["month"] = d["time"].dt.month
    d[["ref", "id", "time", "label", "call_two_test", "group", "rh_site", "rh_below", "dt_inv", "below_profile",
       "canary"]].round(3).to_csv(OUT_CSV, index=False, compression="gzip")

    lines = [f"cloud-below profile: RH >= {RH_WET:.0f}% on a level >= {BELOW_HPA:.0f} hPa below the site and RH < "
             f"{RH_DRY:.0f}% at the site; rows snow-free, satellite clear or cloudy, two-test call of the frozen classifier",
             f"photometers with profiles: {len(prof)}; rows: {len(d)}; Canary profiles "
             + ("every level 1000-600 hPa (era5pl_arco_canary)" if full else "the bracketing levels only (lowest 850 hPa)")]
    for ref in ("goes", "msg", "viirs"):
        for region, sel in (("Canary", True), ("other", False)):
            x = d[(d["ref"] == ref) & (d["canary"] == sel)]
            if x.empty:
                continue
            lines.append(f"{ref} {region}: photometers {x['id'].nunique()}")
            for grp in ("missed", "agreed", "clear"):
                y = x[x["group"] == grp]
                if y.empty:
                    continue
                lines.append(f"  {grp:7s} n {len(y):7d}  cloud-below profile {y['below_profile'].mean():.6f}  "
                             f"median RH site {y['rh_site'].median():.1f}  RH below {y['rh_below'].median():.1f}  "
                             f"inversion (dt>0) {(y['dt_inv'] > 0).mean():.3f}")
            cl = x[x["label"] == "cloudy"]
            if len(cl):
                by = cl.groupby("month")["group"].apply(lambda v: (v == "missed").mean())
                lines.append("  missed share of satellite-cloudy rows by month: "
                             + " ".join(f"{k}:{v:.2f}" for k, v in by.items()))
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
