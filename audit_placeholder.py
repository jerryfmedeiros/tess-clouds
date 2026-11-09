"""audit_placeholder.py — the evidence that the photometers filed at one placeholder position do not share a sky,
against photometers that really share a site.

For each group of identifiers at one position, from the raw archive (build_nights.read_unit, so the same parsing;
the placeholder photometers have no night files, build_nights.py skips them): the readings are binned to 5 minutes,
each photometer's median dark-sky brightness is taken over bins with the Sun below -18 deg and the Moon below -5 deg
at the registered position, and the Pearson correlation of the median sky temperature between each pair is taken on
their common bins (at least MIN_COMMON). Photometers that share a site track the same sky closely; the seven TESS4C
units filed at "The Hebrew University of Jerusalem" (audit_positions.PLACEHOLDERS) differ by up to several
magnitudes and correlate loosely, so their position is unknown and they are excluded.

Reads the raw archive, so it is not a verify target (run_all.py --raw, stage 1). Writes
results/audit_placeholder.csv (group, id_a, id_b, common_bins, corr_sky, mag_a, mag_b) and
results/audit_placeholder.txt.

Usage: python audit_placeholder.py"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from build_nights import read_unit, sun_moon

ROOT = Path(__file__).parent
SELECTED = ROOT / "data" / "tess_archive" / "selected.csv"
OUT_CSV = ROOT / "results" / "audit_placeholder.csv"
OUT_TXT = ROOT / "results" / "audit_placeholder.txt"
MIN_COMMON = 2000
BIN = "5min"
GROUPS = {
    "placeholder (Hebrew University of Jerusalem)": ["stars705", "stars706", "stars707", "stars854", "stars855", "stars856", "stars857"],
    "shared site stars75/76": ["stars75", "stars76"],
    "shared site stars40/41": ["stars40", "stars41"],
    "shared site stars312/313": ["stars312", "stars313"],
    "shared site Villaverde del Ducado": ["stars2", "stars489", "stars717"],
}


def bins(uid: str, lat: float, lon: float) -> pd.DataFrame | None:
    d = read_unit(uid)
    d = d[(d["sky"] != 0.0) & (d["sky"] > -273)]
    if d.empty:
        return None
    b = d.set_index("t").resample(BIN).agg(sky=("sky", "median"), mag=("mag", "median")).dropna(subset=["sky"])
    hours = b.index.floor("h")
    sm = sun_moon(pd.DatetimeIndex(hours.unique()), lat, lon)
    b["sun_alt"] = sm["sun_alt"].reindex(hours).to_numpy()
    b["moon_alt"] = sm["moon_alt"].reindex(hours).to_numpy()
    return b


def main():
    pos = pd.read_csv(SELECTED).set_index("id")
    rows, lines = [], []
    for group, ids in GROUPS.items():
        data = {u: b for u in ids if u in pos.index and (b := bins(u, pos.at[u, "lat"], pos.at[u, "lon"])) is not None}
        mags = {u: float(b.loc[(b["sun_alt"] < -18) & (b["moon_alt"] < -5), "mag"].median()) for u, b in data.items()}
        corrs = []
        for a, c in combinations(sorted(data), 2):
            j = data[a][["sky"]].join(data[c][["sky"]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
            if len(j) < MIN_COMMON:
                continue
            r = float(j["sky_a"].corr(j["sky_b"]))
            corrs.append(r)
            rows.append([group, a, c, len(j), round(r, 3), round(mags[a], 2), round(mags[c], 2)])
        if corrs:
            lines.append(f"{group}: {len(data)} photometers; median dark-sky brightness {min(mags.values()):.2f}-{max(mags.values()):.2f} "
                         f"mag (largest difference {max(mags.values()) - min(mags.values()):.2f}); sky-temperature correlation "
                         f"{min(corrs):.2f}-{max(corrs):.2f} over {len(corrs)} pairs")
        else:
            lines.append(f"{group}: {len(data)} photometers; no pair with {MIN_COMMON} common bins")
    pd.DataFrame(rows, columns=["group", "id_a", "id_b", "common_bins", "corr_sky", "mag_a", "mag_b"]).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
