"""die_air.py — how the die (enclosure) temperature of the TESS-W infrared sensor follows the air temperature.

The archive's "ambient" column is the sensor die, warmed by a heater on cold nights. Over every astronomical-night bin
(Sun below -18 deg) of the photometers of the record (qc_baseline.with_baseline) with an ERA5-Land 2 m air
temperature, the bins are split by air temperature (AIR bands, as breakdown_metar.py) and each band gives the
median die temperature and the median die minus air. The same for the summit photometers of sat_score.MOUNTAIN
(Teide and Roque de los Muchachos), whose heated die Sect. 4.1 of the paper compares with the Alarcon et al. cut.

Writes results/die_air.txt.

Usage: python die_air.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sat_score as ss
from qc_baseline import with_baseline

ROOT = Path(__file__).parent
NIGHTS = ROOT / "data" / "nights"
OUT = ROOT / "results" / "die_air.txt"
AIR = [-np.inf, -10, 0, 10, 20, np.inf]
LABELS = ["< -10", "-10 to 0", "0 to 10", "10 to 20", "> 20"]


def main():
    ids = sorted(with_baseline() & {f.name.split(".")[0] for f in NIGHTS.glob("*.csv.gz")})
    parts = []
    for uid in ids:
        d = pd.read_csv(NIGHTS / f"{uid}.csv.gz", usecols=["die_med", "t2m_C", "sun_alt"]).dropna()
        d = d[d["sun_alt"] < -18]
        parts.append(pd.DataFrame({"id": uid, "die": d["die_med"].to_numpy(), "air": d["t2m_C"].to_numpy()}))
    d = pd.concat(parts, ignore_index=True)
    d["band"] = pd.cut(d["air"], AIR, labels=LABELS)
    lines = [f"die temperature against ERA5-Land air temperature, astronomical-night bins of {d['id'].nunique()} "
             f"photometers ({len(d)} bins); median die (C), median die minus air (K), share of bins"]
    for name, h0 in (("network", d), (f"summits ({', '.join(sorted(ss.MOUNTAIN & set(ids)))})", d[d["id"].isin(ss.MOUNTAIN)])):
        lines.append(f"  {name}: median die {h0['die'].median():.2f}")
        for band, h in h0.groupby("band", observed=True):
            lines.append(f"    air {band:9s} bins {len(h):9d} ({len(h) / len(h0):.3f})  die {h['die'].median():6.2f}  "
                         f"die minus air {(h['die'] - h['air']).median():6.2f}")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
