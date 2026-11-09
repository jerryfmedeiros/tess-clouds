"""mountain_baseline.py — does the clear-sky baseline at mountain photometers improve when the air temperature and
humidity come from the site's height instead of the ERA5-Land surface cell?

Sources compared, for each photometer in SITES (same clear candidates, forms, device offsets and month-fold
scoring as baseline_explore.py, restricted to hours where every source has data so the scores share their nights):
  era5land   ERA5-Land 2 m temperature and dew point at the nearest 0.1 deg cell (data/era5_ts)
  station    Izana (WMO 60010, 2368 m) hourly reports from NOAA ISD, linear in time within 3 h (Tenerife sites)
  era5pl     ERA5 pressure-level temperature and relative humidity at the nearest 0.25 deg point, interpolated
             in log pressure to the site's standard-atmosphere pressure; dew point from relative humidity (Magnus)
  era5pl6    the same at 00/06/12/18 UTC from ARCO-ERA5 at the DEM elevation (data/mountain_pl, mountain_pl.py), the
             series features.py uses at every mountain photometer
Writes results/mountain_baseline.csv and results/mountain_baseline.txt.

--network: every mountain photometer of data/mountain_pl, ERA5-Land against era5pl6 on the same clear candidates
(held-out robust SD and median of the Brunt and linear forms). Writes results/mountain_network.csv and .txt.

Usage: python mountain_baseline.py [--no-pl] [--network]   (--no-pl: ERA5-Land and station only)"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import baseline_explore as be

ROOT = Path(__file__).parent
MOUNT = ROOT / "data" / "mountain"
MOUNT_PL = ROOT / "data" / "mountain_pl"
NET_CSV = ROOT / "results" / "mountain_network.csv"
NET_TXT = ROOT / "results" / "mountain_network.txt"
OUT_CSV = ROOT / "results" / "mountain_baseline.csv"
OUT_TXT = ROOT / "results" / "mountain_baseline.txt"
# id: (name, elevation m, Tenerife?) — elevations of the sites, not of the archive header (which gives 0)
SITES = {"stars211": ("Teide Observatory", 2390, True), "stars427": ("Teide Observatory", 2390, True),
         "stars507": ("Pico del Teide, cable-car top station", 3555, True),
         "stars90": ("Roque de los Muchachos Observatory", 2396, False)}


def p_std(z_m: float) -> float:
    return 1013.25 * (1 - 2.25577e-5 * z_m) ** 5.25588


def dew_from_rh(t_c, rh):
    a, b = 17.62, 243.12
    g = np.log(np.clip(rh, 1, 100) / 100) + a * t_c / (b + t_c)
    return b * g / (a - g)


def era5pl_series(lat: float, lon: float, z_m: float) -> pd.DataFrame:
    frames = []
    for f in sorted(MOUNT.glob("era5pl_*.nc")):
        with netCDF4.Dataset(f) as nc:
            la, lo = nc.variables["latitude"][:], nc.variables["longitude"][:]
            i, j = int(np.abs(la - lat).argmin()), int(np.abs(lo - lon).argmin())
            lev = np.asarray(nc.variables["pressure_level"][:], float)
            tv = nc.variables["valid_time"]
            times = pd.to_datetime(netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False).astype("datetime64[s]"))
            t = np.asarray(nc.variables["t"][:, :, i, j], float) - 273.15
            r = np.asarray(nc.variables["r"][:, :, i, j], float)
        p, lp = p_std(z_m), np.log(lev)
        o = np.argsort(lp)
        ti = np.array([np.interp(np.log(p), lp[o], row[o]) for row in t])
        ri = np.array([np.interp(np.log(p), lp[o], row[o]) for row in r])
        frames.append(pd.DataFrame({"t": ti, "td": dew_from_rh(ti, ri)}, index=times))
    return pd.concat(frames).sort_index()


def station_series() -> pd.DataFrame:
    s = pd.read_csv(MOUNT / "isd_600100.csv.gz", parse_dates=["time"]).set_index("time")
    return s.rename(columns={"t_C": "t", "td_C": "td"})


def at(series: pd.DataFrame, times: pd.Series, max_gap_h: float) -> pd.DataFrame:
    st = series.index.as_unit("ns").asi8 / 1e9
    bt = pd.DatetimeIndex(times).as_unit("ns").asi8 / 1e9
    k = np.clip(np.searchsorted(st, bt), 1, len(st) - 1)
    gap = np.minimum(np.abs(st[k] - bt), np.abs(st[k - 1] - bt))
    out = {}
    for c in ("t", "td"):
        v = np.interp(bt, st, series[c].to_numpy(float))
        v[(gap > max_gap_h * 3600) | (bt < st[0]) | (bt > st[-1])] = np.nan
        out[c] = v
    return pd.DataFrame(out, index=times.index)


def score(c: pd.DataFrame, form: str) -> tuple[float, float, int]:
    sd, med = be.robust(np.concatenate(be.cv_residuals(form, c)))
    return round(sd, 2), round(med, 2), c["device"].nunique()


def pl6_series(uid: str) -> pd.DataFrame:
    s = pd.read_csv(MOUNT_PL / f"{uid}.csv.gz", parse_dates=["time"]).set_index("time")
    return s.rename(columns={"t_C": "t", "td_C": "td"})


def network():
    elev = pd.read_csv(ROOT / "data" / "elevation.csv").set_index("id")
    rows = []
    for f in sorted(MOUNT_PL.glob("*.csv.gz")):
        uid = f.name.split(".")[0]
        nf = ROOT / "data" / "nights" / f"{uid}.csv.gz"
        if not nf.exists():
            continue
        d = be.attach_device(pd.read_csv(nf, parse_dates=["time"]), uid)
        c = be.candidates(d, uid).copy()
        pl = at(pl6_series(uid), c["time"], 3.5)
        c["pl_t"], c["pl_td"] = pl["t"], pl["td"]
        c = c.dropna(subset=["pl_t", "pl_td"])
        if len(c) < be.MIN_CANDIDATES:
            rows.append([uid, elev.loc[uid, "dem_m"], len(c)] + [""] * 8)
            continue
        row = [uid, elev.loc[uid, "dem_m"], len(c)]
        for form in ("brunt", "linear"):
            for tc, dc in (("t2m_C", "d2m_C"), ("pl_t", "pl_td")):
                cc = c.copy()
                cc["t2m_C"], cc["d2m_C"] = c[tc], c[dc]
                sd, med = be.robust(np.concatenate(be.cv_residuals(form, cc)))
                row += [f"{sd:.6f}", f"{med:.6f}"]
        rows.append(row)
    cols = ["id", "dem_m", "candidates"] + [f"{k}_{form}_{src}" for form in ("brunt", "linear")
                                             for src in ("land", "pl") for k in ("sd", "med")]
    t = pd.DataFrame(rows, columns=cols)
    t.to_csv(NET_CSV, index=False)
    v = t[t["sd_brunt_land"] != ""].copy()
    for k in cols[3:]:
        v[k] = v[k].astype(float)
    v["ratio"] = v["sd_brunt_pl"] / v["sd_brunt_land"]
    lines = [f"mountain photometers (DEM >= 1500 m) with series: {len(t)}, with >= {be.MIN_CANDIDATES} candidates: {len(v)}",
             f"Brunt held-out SD, median over photometers: ERA5-Land {v['sd_brunt_land'].median():.6f} "
             f"pressure levels {v['sd_brunt_pl'].median():.6f}; lower with levels at {(v['ratio'] < 1).sum()} of {len(v)}",
             f"median ratio levels/Land {v['ratio'].median():.6f}",
             v[["id", "dem_m", "candidates", "sd_brunt_land", "sd_brunt_pl", "ratio"]].sort_values("dem_m")
             .round(3).to_string(index=False)]
    NET_TXT.write_text("\n".join(lines) + "\n")
    print(NET_TXT.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pl", action="store_true")
    ap.add_argument("--network", action="store_true")
    a = ap.parse_args()
    if a.network:
        return network()
    sel = {r["id"]: r for r in csv.DictReader((ROOT / "data" / "tess_archive" / "selected.csv").open())}
    station = station_series()
    rows = []
    for uid, (name, z, tenerife) in SITES.items():
        d = be.attach_device(pd.read_csv(ROOT / "data" / "nights" / f"{uid}.csv.gz", parse_dates=["time"]), uid)
        c = be.candidates(d, uid).copy()
        sources = {"era5land": ("t2m_C", "d2m_C")}
        if not a.no_pl:
            pl = at(era5pl_series(float(sel[uid]["lat"]), float(sel[uid]["lon"]), z), c["time"], 1.5)
            c["pl_t"], c["pl_td"] = pl["t"], pl["td"]
            sources["era5pl"] = ("pl_t", "pl_td")
            if (MOUNT_PL / f"{uid}.csv.gz").exists():
                p6 = at(pl6_series(uid), c["time"], 3.5)
                c["p6_t"], c["p6_td"] = p6["t"], p6["td"]
                sources["era5pl6"] = ("p6_t", "p6_td")
        if tenerife:
            stn = at(station, c["time"], 3.0)
            c["st_t"], c["st_td"] = stn["t"], stn["td"]
            sources["station"] = ("st_t", "st_td")
        c = c.dropna(subset=[col for pair in sources.values() for col in pair])
        for src, (tc, dc) in sources.items():
            cc = c.copy()
            cc["t2m_C"], cc["d2m_C"] = c[tc], c[dc]
            for form in ("temp", "linear", "brunt"):
                sd, med, ny = score(cc, form)
                rows.append([uid, name, src, form, ny, len(cc), sd, med])
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "site", "source", "form", "devices", "candidates", "heldout_robust_sd_C", "heldout_median_C"])
        w.writerows(rows)
    t = pd.DataFrame(rows, columns=["id", "site", "source", "form", "devices", "n", "sd", "med"])
    txt = t.pivot_table(index=["id", "source"], columns="form", values="sd").round(2).to_string()
    OUT_TXT.write_text("held-out robust SD of clear-sky residuals (C), same candidate bins for every source\n" + txt + "\n")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
