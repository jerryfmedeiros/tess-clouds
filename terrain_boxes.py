"""terrain_boxes.py — how much of the satellite box around each Canary mountain photometer lies below the trade-wind
cloud sea (exploratory, with cloud_top.py, 2026-11-03).

A layer of cloud below the summit can fill a satellite box only where the box reaches terrain below the layer's top.
For each Canary photometer of results/mountain_below.csv.gz, the ASTER GDEM elevation (OpenTopoData, as
fetch_elevation.py; positions from data/elevation.csv) on a square grid of STEP_KM around the site, and the share of
grid points below SEA_TOP_M within half-widths of 2.6 km (the VIIRS 7 x 7 box of 750 m pixels), 4.5 km (the Meteosat
3 x 3 box and cloud-top cell) and 6 km. Sea points give 0 m. SEA_TOP_M is a round value for the top of the cloud sea,
fixed before looking; results/cloud_top.txt gives the measured tops.

Writes results/terrain_boxes.txt.

Usage: python terrain_boxes.py"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_elevation import API, DATASET

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "terrain_boxes.txt"
STEP_KM, HALF_KM = 0.5, 6.0
WIDTHS = (2.6, 4.5, 6.0)
SEA_TOP_M = 1500.0


def elevations(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = []
    for k in range(0, len(lat), 50):
        locs = "|".join(f"{a:.5f},{b:.5f}" for a, b in zip(lat[k:k + 50], lon[k:k + 50]))
        req = urllib.request.Request(f"{API}{DATASET}?locations={locs}", headers={"User-Agent": "tess-clouds research script"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out += [x["elevation"] if x["elevation"] is not None else 0.0 for x in json.load(r)["results"]]
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(5 * 2 ** attempt)
        time.sleep(1.1)                                  # public API: one request a second
    return np.asarray(out, float)


def main():
    e = pd.read_csv(ROOT / "data" / "elevation.csv").set_index("id")
    ids = sorted(pd.read_csv(ROOT / "results" / "mountain_below.csv.gz", usecols=["id", "canary"]).query("canary")["id"].unique())
    g = np.arange(-HALF_KM, HALF_KM + 1e-9, STEP_KM)
    dx, dy = np.meshgrid(g, g)
    lines = [f"share of terrain below {SEA_TOP_M:.0f} m within half-widths {', '.join(f'{w} km' for w in WIDTHS)} "
             f"(grid {STEP_KM} km, ASTER GDEM via OpenTopoData; sea = 0 m)"]
    for uid in ids:
        lat, lon = float(e.loc[uid, "lat"]), float(e.loc[uid, "lon"])
        z = elevations(lat + dy.ravel() / 111.2, lon + dx.ravel() / (111.2 * np.cos(np.radians(lat)))).reshape(dx.shape)
        cells = []
        for w in WIDTHS:
            m = (np.abs(dx) <= w + 1e-9) & (np.abs(dy) <= w + 1e-9)
            cells.append(f"{w} km: {np.mean(z[m] < SEA_TOP_M):.3f} (min {z[m].min():.0f} m)")
        lines.append(f"{uid} {e.loc[uid, 'dem_m']:.0f} m  " + "  ".join(cells))
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
