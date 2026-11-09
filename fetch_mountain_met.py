"""fetch_mountain_met.py — air temperature and humidity at the height of mountain photometers, two ways, for the
clear-sky baseline test at high sites (Teide, La Palma).

1. Station: hourly temperature and dew point from NOAA's Integrated Surface Database (global-hourly, public),
   for the stations in STATIONS (Izana, WMO 60010, 2368 m, about 5 km from the Teide Observatory). Writes
   data/mountain/isd_<usaf>.csv.gz (time UTC, t_C, td_C).
2. ERA5 pressure levels: hourly temperature and relative humidity at LEVELS hPa over a small box around the
   Canary Islands summits (reanalysis-era5-pressure-levels, 0.25 deg), night hours only (HOURS UTC), one CDS request
   per quarter (a year is over the CDS cost limit; CDS holds each request in its queue for hours, so few large requests finish sooner than many small
   ones). Writes data/mountain/era5pl_<YYYY>-Q<n>.nc; mountain_baseline.py interpolates to each site's pressure.
3. ERA5 pressure levels at every mountain photometer (DEM elevation >= MOUNTAIN_M, data/elevation.csv), for the
   network-wide version of the test above. Photometers are grouped by region (GROUPS, by longitude); each group asks
   for the ERA5 levels just above and below each of its sites' standard-atmosphere pressure, at 00/06/12/18 UTC
   (on the Canaries hourly and 6-hourly levels gave the same baseline scatter, and 6-hourly lets one request hold
   several months). A request spans as many months of one year as fit under the CDS cost limit (COST_LIMIT, cost 6 per field
   and day) and its box covers only the group's photometers with nights in those months (from data/nights).
   The Canaries are inside the Europe group again, to compare with the hourly files of 2. Writes
   <keep>/era5pl/<group>_<first month>_<last month>.nc (raw files, kept outside the repository).
4. The files of 3 from ARCO-ERA5 (Google's public Zarr copy of ERA5; no queue), with relative humidity from specific
   humidity (arco_check.py compares the two sources). Writes <keep>/era5pl_arco/<same names>.
5. Every level from 1000 to 600 hPa over the Canary box from ARCO-ERA5, for the cloud-below test of mountain_below.py
   (the files of 4 stop at 850 hPa there). Writes <keep>/era5pl_arco_canary/.

Usage: python fetch_mountain_met.py --isd | --era5 | --era5-sites --keep RAW
       | --arco | --arco-canary --keep RAW [--workers 6]"""
from __future__ import annotations

import argparse
import gzip
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "mountain"
STATIONS = {"600100": "IZANA"}
YEARS = range(2019, 2027)
LEVELS = ["650", "750", "775"]                  # Teide summit about 654 hPa, observatories about 762 hPa
HOURS = [f"{h:02d}:00" for h in list(range(19, 24)) + list(range(0, 8))]   # covers every Sun < -12 deg bin
AREA = [29.0, -18.25, 28.0, -16.25]            # N, W, S, E: Tenerife and La Palma
MOUNTAIN_M = 1500
ERA5_LEVELS = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
GROUPS = {"europe": (-30, 30), "indian": (30, 180), "americas": (-180, -30)}   # longitude ranges
SYNOPTIC = ["00:00", "06:00", "12:00", "18:00"]
COST_LIMIT = 60000


def isd():
    OUT.mkdir(parents=True, exist_ok=True)
    for usaf, name in STATIONS.items():
        frames = []
        for y in YEARS:
            url = f"https://noaa-global-hourly-pds.s3.amazonaws.com/{y}/{usaf}99999.csv"   # NCEI moved global-hourly to AWS
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "4", url], capture_output=True, text=True)
            if r.returncode:
                print(f"{name} {y}: none ({r.stderr.strip()[:80]})")
                continue
            d = pd.read_csv(io.StringIO(r.stdout), dtype=str)
            if not {"DATE", "TMP", "DEW"} <= set(d.columns):
                print(f"{name} {y}: no TMP/DEW columns")
                continue
            frames.append(d[["DATE", "TMP", "DEW"]])
            print(f"{name} {y}: {len(d)} reports")
        d = pd.concat(frames)

        def val(s):                                     # "+0123,1" -> 12.3; 9999 missing; keep good QC codes
            v, q = s.str.split(",", expand=True)[0], s.str.split(",", expand=True)[1]
            x = pd.to_numeric(v, errors="coerce") / 10
            return x.where((x.abs() < 99) & q.isin(list("0145")))

        out = pd.DataFrame({"time": pd.to_datetime(d["DATE"]), "t_C": val(d["TMP"]), "td_C": val(d["DEW"])})
        out = out.dropna().drop_duplicates("time").sort_values("time")
        out.to_csv(OUT / f"isd_{usaf}.csv.gz", index=False, compression="gzip")
        print(f"{name}: {len(out)} hours with temperature and dew point")


def era5():
    import cdsapi
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)

    def one(yq):
        y, q = yq
        f = OUT / f"era5pl_{y}-Q{q}.nc"
        if f.exists():
            return f"{y}-Q{q}: exists"
        months = [f"{m:02d}" for m in range(3 * q - 2, 3 * q + 1)]
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-pressure-levels", {
                "product_type": ["reanalysis"], "variable": ["temperature", "relative_humidity"],
                "pressure_level": LEVELS, "year": [str(y)], "month": months,
                "day": [f"{d:02d}" for d in range(1, 32)], "time": HOURS,
                "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}, str(f.with_suffix(".part")))
        except Exception as e:
            return f"{y}-Q{q}: ERROR {str(e)[:150]}"
        f.with_suffix(".part").replace(f)
        return f"{y}-Q{q}: done"

    with ThreadPoolExecutor(3) as ex:
        for msg in ex.map(one, [(y, q) for y in YEARS for q in range(1, 5) if not (y == 2026 and q > 3)]):
            print(msg, flush=True)


def p_std(z_m: float) -> float:
    return 1013.25 * (1 - 2.25577e-5 * z_m) ** 5.25588


def bracket(p: float) -> list[int]:
    """The ERA5 levels just above and below pressure p (hPa)."""
    lower = max(lv for lv in ERA5_LEVELS if lv <= p)
    upper = min(lv for lv in ERA5_LEVELS if lv >= p)
    return sorted({lower, upper})


def site_jobs(keep: Path) -> list[tuple]:
    e = pd.read_csv(ROOT / "data" / "elevation.csv")
    e = e[e["dem_m"] >= MOUNTAIN_M]
    spans = {}
    for uid in e["id"]:
        f = ROOT / "data" / "nights" / f"{uid}.csv.gz"
        if f.exists():
            t = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"]
            if len(t):
                spans[uid] = (t.min().to_period("M"), t.max().to_period("M"))
    e = e[e["id"].isin(spans)]
    months = pd.period_range(f"{YEARS[0]}-01", "2026-09", freq="M")
    jobs = []
    for g, (w, east) in GROUPS.items():
        u = e[(e["lon"] >= w) & (e["lon"] < east)]
        if u.empty:
            continue
        levels = sorted({lv for z in u["dem_m"] for lv in bracket(p_std(z))})
        per = COST_LIMIT // (6 * 31 * len(SYNOPTIC) * len(levels) * 2)
        blocks = []
        for y in sorted({m.year for m in months}):        # a request names years x months, so a block stays in one year
            ym = months[months.year == y]
            n = -(-len(ym) // per)
            blocks += [ym[i * len(ym) // n:(i + 1) * len(ym) // n] for i in range(n)]
        for block in blocks:
            act = u[[spans[x][0] <= block[-1] and spans[x][1] >= block[0] for x in u["id"]]]
            if act.empty:
                continue
            area = [min(90, float(np.ceil((act["lat"].max() + 0.5) * 4) / 4)), float(np.floor((act["lon"].min() - 0.5) * 4) / 4),
                    max(-90, float(np.floor((act["lat"].min() - 0.5) * 4) / 4)), float(np.ceil((act["lon"].max() + 0.5) * 4) / 4)]
            out = keep / "era5pl" / f"{g}_{block[0]}_{block[-1]}.nc"
            jobs.append((out, [str(m) for m in block], [str(lv) for lv in levels], area, len(act)))
    return jobs


def era5_sites(keep: Path, dry: bool):
    import cdsapi
    from concurrent.futures import ThreadPoolExecutor
    jobs = site_jobs(keep)
    for out, months, levels, area, n in jobs:
        print(f"{out.name}: {len(months)} months, levels {','.join(levels)}, area {area}, {n} photometers", flush=True)
    if dry:
        return
    (keep / "era5pl").mkdir(parents=True, exist_ok=True)

    def one(job):
        out, months, levels, area, _ = job
        if out.exists():
            return f"{out.name}: exists"
        mm = pd.PeriodIndex(months, freq="M")
        req = {"product_type": ["reanalysis"], "variable": ["temperature", "relative_humidity"], "pressure_level": levels,
               "year": sorted({str(m.year) for m in mm}), "month": sorted({f"{m.month:02d}" for m in mm}),
               "day": [f"{d:02d}" for d in range(1, 32)], "time": SYNOPTIC, "area": area,
               "data_format": "netcdf", "download_format": "unarchived"}
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-pressure-levels", req, str(out.with_suffix(".part")))
        except Exception as e:
            return f"{out.name}: ERROR {str(e)[:150]}"
        out.with_suffix(".part").replace(out)
        return f"{out.name}: done"

    with ThreadPoolExecutor(3) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


def era5_arco(keep: Path, workers: int):
    """The files of era5_sites from ARCO-ERA5 instead of the CDS (no queue): same groups, months, levels, boxes and
    hours, with relative humidity computed from specific humidity as the IFS does (arco_check.py). Each hour of the
    globe is read once and cut for every group; finished months are kept as parts, so a restart resumes."""
    from concurrent.futures import ThreadPoolExecutor
    import netCDF4
    import arco_check as arco
    jobs = site_jobs(keep)
    m, ax = arco.meta()
    shape = m["temperature/.zarray"]["shape"]
    cuts = {}
    for out, months, levels, area, _ in jobs:
        lv = sorted((int(x) for x in levels), reverse=True)            # 850 ... 550, as the CDS files
        lat = np.arange(area[0], area[2] - 0.125, -0.25)
        lon = np.arange(area[1], area[3] + 0.125, 0.25)
        cuts[out] = (lv, lat, lon, [int(np.where(ax["level"] == x)[0][0]) for x in lv],
                     [int(np.where(np.isclose(ax["latitude"], y))[0][0]) for y in lat],
                     [int(np.where(np.isclose(ax["longitude"], x % 360))[0][0]) for x in lon])
    dest = keep / "era5pl_arco"
    parts = dest / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for month in pd.period_range(f"{YEARS[0]}-01", "2026-09", freq="M"):
        todo = [out for out, months, *_ in jobs if str(month) in months
                and not (dest / out.name).exists() and not (parts / f"{out.stem}_{month}.npz").exists()]
        if not todo:
            continue
        hours = pd.date_range(month.start_time, month.end_time.floor("D") + pd.Timedelta("18h"), freq="6h")

        def one(h):
            i = int((h - pd.Timestamp("1900-01-01")) / pd.Timedelta("1h")) - arco.EPOCH_HOURS
            t, q = arco.field("temperature", i, shape), arco.field("specific_humidity", i, shape)
            res = {}
            for out in todo:
                lv, _, _, li, yi, xi = cuts[out]
                tt, qq = t[np.ix_(li, yi, xi)], q[np.ix_(li, yi, xi)]
                res[out] = (tt, qq, arco.rh_ifs(tt, qq, (np.asarray(lv, float) * 100)[:, None, None]))
            return res

        with ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(one, hours))
        secs = ((hours - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()
        for out in todo:
            np.savez(parts / f"{out.stem}_{month}.npz", time=secs,
                     **{k: np.stack([g[out][n] for g in got]) for n, k in enumerate(("t", "q", "r"))})
        print(f"{month}: {len(hours)} hours, {len(todo)} files", flush=True)
        for out, months, *_ in jobs:                                   # assemble every file whose months are all in
            fp = [parts / f"{out.stem}_{mm}.npz" for mm in months]
            if (dest / out.name).exists() or not all(p.exists() for p in fp):
                continue
            z = [np.load(p) for p in fp]
            lv, lat, lon, *_ = cuts[out]
            tmp = (dest / out.name).with_suffix(".part")
            with netCDF4.Dataset(tmp, "w") as nc:
                nc.source = "ARCO-ERA5 (gs://gcp-public-data-arco-era5, full_37-1h-0p25deg-chunk-1.zarr-v3); r from q (IFS)"
                for name, vals in (("valid_time", None), ("pressure_level", lv), ("latitude", lat), ("longitude", lon)):
                    nc.createDimension(name, None if vals is None else len(vals))
                v = nc.createVariable("valid_time", "i8", ("valid_time",))
                v.units = "seconds since 1970-01-01"
                v[:] = np.concatenate([x["time"] for x in z])
                for name, vals, u in (("pressure_level", lv, "hPa"), ("latitude", lat, "degrees_north"),
                                      ("longitude", lon, "degrees_east")):
                    c = nc.createVariable(name, "f8", (name,))
                    c.units = u
                    c[:] = vals
                for k, u in (("t", "K"), ("q", "kg kg**-1"), ("r", "%")):
                    c = nc.createVariable(k, "f4", ("valid_time", "pressure_level", "latitude", "longitude"), zlib=True)
                    c.units = u
                    c[:] = np.concatenate([x[k] for x in z])
            tmp.replace(dest / out.name)
            for p in fp:
                p.unlink()
            print(f"{out.name}: done", flush=True)


CANARY_BOX = [29.5, -18.5, 27.0, -13.0]                # N, W, S, E: the Canary photometers (mountain_below.CANARY)
CANARY_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600]


def era5_arco_canary(keep: Path, workers: int):
    """Every ERA5 level from 1000 to 600 hPa over the Canary box, 00/06/12/18 UTC, from ARCO-ERA5. The files of era5_arco hold only the levels that bracket the group's sites, so their
    lowest level is 850 hPa and a cloud sea below it is outside the profile; mountain_below.py reads these files for
    the Canary photometers. Writes <keep>/era5pl_arco_canary/canary_<year>.nc (monthly parts kept until a year is
    complete, so a restart resumes)."""
    from concurrent.futures import ThreadPoolExecutor
    import netCDF4
    import arco_check as arco
    m, ax = arco.meta()
    shape = m["temperature/.zarray"]["shape"]
    lv = CANARY_LEVELS
    lat = np.arange(CANARY_BOX[0], CANARY_BOX[2] - 0.125, -0.25)
    lon = np.arange(CANARY_BOX[1], CANARY_BOX[3] + 0.125, 0.25)
    li = [int(np.where(ax["level"] == x)[0][0]) for x in lv]
    yi = [int(np.where(np.isclose(ax["latitude"], y))[0][0]) for y in lat]
    xi = [int(np.where(np.isclose(ax["longitude"], x % 360))[0][0]) for x in lon]
    dest = keep / "era5pl_arco_canary"
    parts = dest / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(f"{YEARS[0]}-01", "2026-09", freq="M")
    for month in months:
        part = parts / f"canary_{month}.npz"
        if part.exists() or (dest / f"canary_{month.year}.nc").exists():
            continue
        hours = pd.date_range(month.start_time, month.end_time.floor("D") + pd.Timedelta("18h"), freq="6h")

        def one(h):
            i = int((h - pd.Timestamp("1900-01-01")) / pd.Timedelta("1h")) - arco.EPOCH_HOURS
            t = arco.field("temperature", i, shape)[np.ix_(li, yi, xi)]
            q = arco.field("specific_humidity", i, shape)[np.ix_(li, yi, xi)]
            return t, q, arco.rh_ifs(t, q, (np.asarray(lv, float) * 100)[:, None, None])

        with ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(one, hours))
        secs = ((hours - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()
        np.savez(part, time=secs, **{k: np.stack([g[n] for g in got]) for n, k in enumerate(("t", "q", "r"))})
        print(f"{month}: {len(hours)} hours", flush=True)
    for year in sorted({mm.year for mm in months}):
        out = dest / f"canary_{year}.nc"
        fp = [parts / f"canary_{mm}.npz" for mm in months if mm.year == year]
        if out.exists() or not all(p.exists() for p in fp):
            continue
        z = [np.load(p) for p in fp]
        tmp = out.with_suffix(".part")
        with netCDF4.Dataset(tmp, "w") as nc:
            nc.source = "ARCO-ERA5 (gs://gcp-public-data-arco-era5, full_37-1h-0p25deg-chunk-1.zarr-v3); r from q (IFS)"
            for name, vals in (("valid_time", None), ("pressure_level", lv), ("latitude", lat), ("longitude", lon)):
                nc.createDimension(name, None if vals is None else len(vals))
            v = nc.createVariable("valid_time", "i8", ("valid_time",))
            v.units = "seconds since 1970-01-01"
            v[:] = np.concatenate([x["time"] for x in z])
            for name, vals, u in (("pressure_level", lv, "hPa"), ("latitude", lat, "degrees_north"),
                                  ("longitude", lon, "degrees_east")):
                c = nc.createVariable(name, "f8", (name,))
                c.units = u
                c[:] = vals
            for k, u in (("t", "K"), ("q", "kg kg**-1"), ("r", "%")):
                c = nc.createVariable(k, "f4", ("valid_time", "pressure_level", "latitude", "longitude"), zlib=True)
                c.units = u
                c[:] = np.concatenate([x[k] for x in z])
        tmp.replace(out)
        for p in fp:
            p.unlink()
        print(f"{out.name}: done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isd", action="store_true")
    ap.add_argument("--era5", action="store_true")
    ap.add_argument("--era5-sites", action="store_true")
    ap.add_argument("--arco", action="store_true", help="the --era5-sites files from ARCO-ERA5")
    ap.add_argument("--arco-canary", action="store_true", help="every level 1000-600 hPa over the Canaries, ARCO-ERA5")
    ap.add_argument("--workers", type=int, default=6, help="hours read at once with --arco")
    ap.add_argument("--keep", type=Path)
    ap.add_argument("--dry", action="store_true", help="list the requests only")
    a = ap.parse_args()
    if a.isd:
        isd()
    if a.era5:
        era5()
    if a.era5_sites:
        era5_sites(a.keep, a.dry)
    if a.arco:
        era5_arco(a.keep, a.workers)
    if a.arco_canary:
        era5_arco_canary(a.keep, a.workers)
