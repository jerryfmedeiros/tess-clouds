"""metar_labels.py — the METAR reference: sky condition of every automated night report at the airport paired
with each photometer, and the photometer bins that report covers.

Pairing: each photometer takes the nearest airport within MAX_KM (data/metar/pairs.csv) whose night reports are
at least MIN_AUTO automated (results/metar_auto.csv). Only reports carrying AUTO are used: human night reports
overstate clearing (Fernandes Medeiros 2026). Reports with the Sun at or above SUN_MAX deg at the airport are dropped.

Sky condition of a report, from its cloud groups (FEW 2, SCT 4, BKN 6, OVC 8 oktas; VV is a sky obscured, 8):
  okta      the largest cover reported (0 when no layer is reported)
  code      CLR, SKC, NCD (no layer: clear), NSC or CAVOK (no cloud below 5000 ft or the highest minimum sector
            altitude, and no CB or TCU: higher cloud is possible), LAYERS, or NONE (no sky group, or only //////
            groups: unusable, dropped)
  label     clear (okta 0), partial (FEW or SCT), cloudy (BKN, OVC or VV)
  base_ft   base of the lowest layer, top_ft base of the highest layer reported (ft above the airport)
Coverage: an automated report summarises the ceilometer's last 30 minutes, so the photometer bins whose centres
fall in (t - WINDOW, t] are the ones it is compared with (n_bins of them, at least MIN_BINS).

Photometers flagged obstructed by qc_ir.py (results/qc_ir.csv) are left out.

Writes data/metar/labels/<id>.csv.gz (one row per report) and results/metar_labels.csv (per photometer: airport,
distance, reports by label and code).

Usage: python metar_labels.py"""
from __future__ import annotations

import csv
import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ir import sun_altitude
from qc_ir import obstructed

ROOT = Path(__file__).parent
METAR = ROOT / "data" / "metar"
OUT = METAR / "labels"
SUMMARY = ROOT / "results" / "metar_labels.csv"
MAX_KM = 20.0
MIN_AUTO = 0.90
SUN_MAX = -12.0
WINDOW = pd.Timedelta(minutes=30)
MIN_BINS = 3
OKTA = {"FEW": 2, "SCT": 4, "BKN": 6, "OVC": 8, "VV": 8}
LAYER = re.compile(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3}|///)")
NIL = re.compile(r"\b(CLR|SKC|NCD|NSC|CAVOK)\b")


def sky(metar: str) -> tuple[str, int, float, float]:
    body = metar.split(" RMK ")[0].split(" TEMPO ")[0].split(" BECMG ")[0]
    layers = LAYER.findall(body)
    if layers:
        okta = max(OKTA[k] for k, _ in layers)
        h = [int(b) * 100 for _, b in layers if b != "///"]
        return "LAYERS", okta, (min(h) if h else np.nan), (max(h) if h else np.nan)
    m = NIL.search(body)
    return (m.group(1), 0, np.nan, np.nan) if m else ("NONE", -1, np.nan, np.nan)


def label(okta: int) -> str:
    return "clear" if okta == 0 else "partial" if okta <= 4 else "cloudy"


def pairing() -> dict[str, tuple[str, float]]:
    auto = {r["icao"]: r for r in csv.DictReader((ROOT / "results" / "metar_auto.csv").open())}
    best = {}
    for r in csv.DictReader((METAR / "pairs.csv").open()):
        a, km = auto.get(r["icao"]), float(r["km"])
        if km > MAX_KM or not a or not a["auto_share"] or float(a["auto_share"]) < MIN_AUTO:
            continue
        if r["id"] not in best or km < best[r["id"]][1]:
            best[r["id"]] = (r["icao"], km)
    return best


def reports(icao: str) -> pd.DataFrame:
    rows = []
    for f in sorted(METAR.glob(f"{icao}_*.csv.gz")):
        for r in csv.DictReader(io.StringIO(gzip.open(f, "rt").read())):
            m = r.get("metar", "")
            if " AUTO " not in f" {m} ":
                continue
            try:
                t = pd.Timestamp(r["valid"])
                if sun_altitude(t.to_pydatetime(), float(r["lat"]), float(r["lon"])) >= SUN_MAX:
                    continue
            except (ValueError, KeyError):
                continue
            code, okta, base, top = sky(m)
            if code != "NONE":
                rows.append((t, code, okta, label(okta), base, top))
    d = pd.DataFrame(rows, columns=["time", "code", "okta", "label", "base_ft", "top_ft"])
    return d.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


def main():
    skip = obstructed()
    pairs = {u: v for u, v in pairing().items() if u not in skip}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.csv.gz"):                        # labels of photometers no longer scored
        if f.name.split(".")[0] not in pairs:
            f.unlink()
    summary = []
    for uid, (icao, km) in sorted(pairs.items()):
        f = ROOT / "data" / "features" / f"{uid}.csv.gz"
        if not f.exists():
            continue
        bins = pd.read_csv(f, usecols=["time"], parse_dates=["time"])["time"].to_numpy()
        d = reports(icao)
        if d.empty:
            continue
        t = d["time"].to_numpy()
        lo = np.searchsorted(bins, t - WINDOW.to_timedelta64(), side="right")
        hi = np.searchsorted(bins, t, side="right")
        d["n_bins"] = hi - lo
        d = d[d["n_bins"] >= MIN_BINS]
        d.insert(0, "icao", icao)
        d.to_csv(OUT / f"{uid}.csv.gz", index=False, compression="gzip")
        lab, code = d["label"].value_counts(), d["code"].value_counts()
        summary.append([uid, icao, round(km, 2), len(d)] + [int(lab.get(k, 0)) for k in ("clear", "partial", "cloudy")]
                       + [int(code.get(k, 0)) for k in ("CLR", "SKC", "NCD", "NSC", "CAVOK")])
        print(uid, icao, len(d), flush=True)
    with SUMMARY.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "icao", "km", "reports", "clear", "partial", "cloudy", "CLR", "SKC", "NCD", "NSC", "CAVOK"])
        w.writerows(summary)


if __name__ == "__main__":
    main()
