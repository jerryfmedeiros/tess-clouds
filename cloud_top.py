"""cloud_top.py — satellite cloud-top height at the mountain photometers, for the cloud the masks report there
(exploratory, after mountain_below.py, 2026-11-03).

Question: when a satellite mask calls a mountain photometer's sky cloudy and the classifier calls it clear (missed),
is the cloud top below the photometer, so the photometer is above the cloud? The same is measured for the cloudy
rows the classifier also calls cloudy (agreed). Rows from results/mountain_below.csv.gz (snow-free, satellite
cloudy, frozen two-test call); a seeded random sample of up to N[ref] rows per reference, region (Canary / other)
and group, and every scan or granule they need is read once.

Products, all heights in metres above sea level:
  goes   ABI-L2-ACHAF (GOES-East cloud-top height, full disk, 10 km), the scan with the same start as the mask's
         (ABI-L2-ACMF); the pixel containing the photometer, DQF 0 or 1 (in this product 1 means not geolocated;
         such pixels carry no height, so only DQF 0 contributes).
  msg    EO:EUM:DAT:MSG:CTH ("Cloud Top Height - MSG - 0 degree", height of the highest cloud, 1237 x 1237 grid of
         3 x 3 SEVIRI pixels, 320 m steps), the product with the mask's sensing end; the cell containing the
         photometer (the area of the mask's 3 x 3 box).
  viirs  CLDPROP_L2_VIIRS_NOAA20 (NASA, Cloud_Top_Height of the NOAA CLAVR-x AWG algorithm, 750 m), the granule with
         the mask granule's start; the valid pixels of the 7 x 7 box around the nearest pixel (the mask's box): the
         highest and the median.
Classes fixed before looking, from the top (VIIRS: the highest in the box) and the DEM elevation of the photometer:
  below  top at or below the photometer;  low  up to 1 km above it;  mid  1 km above it to 6 km above sea level;
  high   6 km above sea level and higher;  none  no valid height.
Raw files are kept in <keep>/cloudtop/<ref>/. Writes data/cloudtop/<ref>.csv (per row, resumable) and, with
--summary, results/cloud_top.txt.

Usage: python cloud_top.py --ref goes|msg|viirs [--workers 8] [--keep RAW]
       python cloud_top.py --summary"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import goes_extract as ge
import msg_extract as me
import viirs_extract as ve

RAW = Path(os.environ.get("TESS_CLOUDS_RAW", Path(__file__).parent / "data" / "raw"))   # raw files, outside git

ROOT = Path(__file__).parent
ROWS = ROOT / "results" / "mountain_below.csv.gz"
OUT = ROOT / "data" / "cloudtop"
TXT = ROOT / "results" / "cloud_top.txt"
N = {"goes": 1500, "msg": 1500, "viirs": 400}
SEED = 20261103
HIGH_M, LOW_KM = 6000.0, 1.0
MSG_NONE = 9999.0          # Meteosat CTH value for no height; valid heights run 320-15040 m in 320 m steps, above it too
READ_LOCK = threading.Lock()                       # netCDF4/HDF5 is not safe across threads: downloads run in
                                                    # parallel, files are read one at a time
FIELDS = ["ref", "id", "time", "group", "canary", "dem_m", "top_m", "top_med_m", "n_valid", "file"]


def rows_for(ref: str) -> pd.DataFrame:
    d = pd.read_csv(ROWS, parse_dates=["time"])
    d = d[(d["ref"] == ref) & d["group"].isin(["missed", "agreed"])]
    rng = np.random.default_rng(SEED)
    parts = []
    for _, g in d.groupby(["canary", "group"]):
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=min(N[ref], len(g)), replace=False))])
    s = pd.concat(parts)
    pos = {r["id"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(ge.SELECTED.open())}
    dem = pd.read_csv(ROOT / "data" / "elevation.csv").set_index("id")["dem_m"]
    s["lat"] = [pos[u][0] for u in s["id"]]
    s["lon"] = [pos[u][1] for u in s["id"]]
    s["dem_m"] = [float(dem[u]) for u in s["id"]]
    return s.sort_values("time").reset_index(drop=True)


def get(url: str, dest: Path, headers: dict | None = None) -> Path:
    if dest.exists():
        return dest
    for k in range(6):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=600) as r:
                dest.with_suffix(".part").write_bytes(r.read())
            dest.with_suffix(".part").replace(dest)
            return dest
        except OSError:
            if k == 5:
                raise
            time.sleep(10 * 2 ** k)


def goes_file(t: pd.Timestamp, keep: Path) -> Path | None:
    stamp = f"s{t.year}{t.dayofyear:03d}{t.hour:02d}{t.minute:02d}{t.second:02d}"
    for bucket in (("noaa-goes16", "noaa-goes19") if t < pd.Timestamp(ge.HANDOVER) else ("noaa-goes19", "noaa-goes16")):
        keys = ge.s3_keys(bucket, f"ABI-L2-ACHAF/{t.year}/{t.dayofyear:03d}/{t.hour:02d}/")
        k = next((k for k in keys if stamp in k), None)
        if k:
            return get(f"https://{bucket}.s3.amazonaws.com/{k}", keep / k.rsplit("/", 1)[-1])
    return None


def goes_read(path: Path, units: pd.DataFrame) -> list:
    out = []
    with netCDF4.Dataset(path) as nc:
        p = nc["goes_imager_projection"]
        x, y = nc["x"][:], nc["y"][:]
        for u in units.itertuples():
            g = ge.geos_xy(u.lat, u.lon, p)
            if g is None:
                out.append((np.nan, np.nan, 0))
                continue
            i, j = int(np.abs(y - g[1]).argmin()), int(np.abs(x - g[0]).argmin())
            h, q = nc["HT"][i, j], nc["DQF"][i, j]
            ok = not np.ma.is_masked(h) and int(q) in (0, 1)
            out.append((float(h), float(h), 1) if ok else (np.nan, np.nan, 0))
    return out


def msg_file(t: pd.Timestamp, keep: Path, ds) -> Path | None:
    c = ds.get_collection("EO:EUM:DAT:MSG:CTH")
    t0 = t.to_pydatetime()
    for p in c.search(dtstart=t0 - dt.timedelta(minutes=16), dtend=t0 + dt.timedelta(minutes=1)):
        if abs((pd.Timestamp(p.sensing_end) - t).total_seconds()) < 60:
            dest = keep / f"{p}.zip"
            if not dest.exists():
                with p.open() as f:
                    dest.with_suffix(".part").write_bytes(f.read())
                dest.with_suffix(".part").replace(dest)
            return dest
    return None


def msg_read(path: Path, units: pd.DataFrame) -> list:
    v, g = me.read_grib(path.read_bytes())
    out = []
    for u in units.itertuples():
        ij = me.pixel(u.lat, u.lon, g)
        h = np.nan if ij is None else float(v[ij])
        h = np.nan if not np.isfinite(h) or h == MSG_NONE or h <= 0 else h
        out.append((h, h, int(np.isfinite(h))))
    return out


def viirs_file(t: pd.Timestamp, keep: Path) -> Path | None:
    q = {"short_name": "CLDPROP_L2_VIIRS_NOAA20", "page_size": 5,
         "temporal": f"{(t - pd.Timedelta('1min')):%Y-%m-%dT%H:%M:%SZ},{(t + pd.Timedelta('1min')):%Y-%m-%dT%H:%M:%SZ}"}
    with urllib.request.urlopen(ve.CMR + "?" + urllib.parse.urlencode(q), timeout=120) as r:
        import json
        entries = json.load(r)["feed"]["entry"]
    stamp = f".A{t.year}{t.dayofyear:03d}.{t.hour:02d}{t.minute:02d}."
    for e in entries:
        url = next((l["href"] for l in e.get("links", []) if l.get("href", "").endswith(".nc")), "")
        if stamp in url:
            return get(url, keep / url.rsplit("/", 1)[-1], {"Authorization": f"Bearer {ve.token()}"})
    return None


def viirs_pixel(glat: np.ndarray, glon: np.ndarray, lat: float, lon: float) -> tuple[int, int] | None:
    """The granule pixel nearest (lat, lon): a coarse search on every 16th pixel, then the full grid around it."""
    dc = ve.km(lat, lon, glat[::16, ::16], glon[::16, ::16])
    cy, cx = np.unravel_index(int(np.nanargmin(dc)), dc.shape)
    y0, x0 = max(0, cy * 16 - 32), max(0, cx * 16 - 32)
    d = ve.km(lat, lon, glat[y0:y0 + 64, x0:x0 + 64], glon[y0:y0 + 64, x0:x0 + 64])
    iy, ix = np.unravel_index(int(np.nanargmin(d)), d.shape)
    return None if d[iy, ix] > ve.COVERAGE_MAX_KM else (int(iy + y0), int(ix + x0))


def viirs_read(path: Path, units: pd.DataFrame) -> list:
    out = []
    with netCDF4.Dataset(path) as nc:
        geo, phys = nc.groups["geolocation_data"], nc.groups["geophysical_data"]
        glat, glon = np.asarray(geo["latitude"][:], float), np.asarray(geo["longitude"][:], float)
        cth = phys["Cloud_Top_Height"]
        for u in units.itertuples():
            px = viirs_pixel(glat, glon, u.lat, u.lon)
            if px is None:
                out.append((np.nan, np.nan, 0))
                continue
            iy, ix = px
            box = np.ma.asarray(cth[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]).astype(float).filled(np.nan)
            box = box[np.isfinite(box) & (box > -9999)]
            out.append((float(box.max()), float(np.median(box)), len(box)) if len(box) else (np.nan, np.nan, 0))
    return out


def run(ref: str, workers: int, keep: Path):
    s = rows_for(ref)
    keep = keep / "cloudtop" / ref
    keep.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{ref}.csv"
    done = set(pd.read_csv(out, parse_dates=["time"])["time"]) if out.exists() else set()
    todo = [(t, g) for t, g in s.groupby("time") if t not in done]
    print(f"{ref}: {len(s)} rows, {s['time'].nunique()} scans, {len(todo)} to do", flush=True)
    ds = me.datastore() if ref == "msg" else None
    fetch = {"goes": lambda t: goes_file(t, keep), "msg": lambda t: msg_file(t, keep, ds),
             "viirs": lambda t: viirs_file(t, keep)}[ref]
    read = {"goes": goes_read, "msg": msg_read, "viirs": viirs_read}[ref]

    def one(job):
        t, g = job
        try:
            f = fetch(t)
            if f:
                with READ_LOCK:
                    vals = read(f, g)
            else:
                vals = [(np.nan, np.nan, 0)] * len(g)
        except Exception as e:                                       # report and move on; a rerun retries
            return t, None, f"ERROR {t} {str(e)[:120]}"
        return t, [[ref, u.id, t.isoformat(), u.group, u.canary, u.dem_m, *v, f.name if f else ""]
                   for u, v in zip(g.itertuples(), vals)], ""

    new = not out.exists()
    with out.open("a", newline="") as fh, ThreadPoolExecutor(workers) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        for k, (t, rows, err) in enumerate(ex.map(one, todo), 1):
            if err:
                print(err, flush=True)
                continue
            w.writerows(rows)
            fh.flush()
            if k % 200 == 0:
                print(f"{ref}: {k}/{len(todo)}", flush=True)
    print(f"{ref}: done", flush=True)


def klass(d: pd.DataFrame) -> pd.Series:
    top, z = d["top_m"], d["dem_m"]
    return pd.Series(np.select([top.isna(), top <= z, top <= z + LOW_KM * 1000, top < HIGH_M],
                               ["none", "below", "low", "mid"], "high"), index=d.index)


def summary():
    lines = [f"cloud-top classes (fixed before looking): below = top <= photometer (DEM); low = up to {LOW_KM:.0f} km "
             f"above it; mid = to {HIGH_M / 1000:.0f} km above sea level; high = {HIGH_M / 1000:.0f} km and higher; "
             "none = no valid height. VIIRS: the highest valid pixel of the 7 x 7 box; the median pixel is given too."]
    for ref in ("goes", "msg", "viirs"):
        f = OUT / f"{ref}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["class"] = klass(d)
        for region, sel in (("Canary", True), ("other", False)):
            x = d[d["canary"] == sel]
            if x.empty:
                continue
            lines.append(f"{ref} {region}: photometers {x['id'].nunique()}")
            for grp in ("missed", "agreed"):
                y = x[x["group"] == grp]
                if y.empty:
                    continue
                sh = y["class"].value_counts(normalize=True)
                v = y[y["class"] != "none"]
                line = (f"  {grp:7s} n {len(y):5d}  " + "  ".join(f"{c} {sh.get(c, 0):.3f}" for c in
                        ("below", "low", "mid", "high", "none"))
                        + f"  median top minus site {np.median(v['top_m'] - v['dem_m']) / 1000:+.2f} km")
                if ref == "viirs" and "top_med_m" in y and y["top_med_m"].notna().any():
                    w = y[y["top_med_m"].notna()]
                    line += (f"; median pixel: at or below the site {(w['top_med_m'] <= w['dem_m']).mean():.3f}, "
                             f"median top minus site {np.median(w['top_med_m'] - w['dem_m']) / 1000:+.2f} km")
                lines.append(line)
    TXT.write_text("\n".join(lines) + "\n")
    print(TXT.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", choices=("goes", "msg", "viirs"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep", type=Path, default=RAW)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        run(a.ref, a.workers, a.keep)
