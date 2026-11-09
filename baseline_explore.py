"""baseline_explore.py — first look at the clear-sky baseline: how well ERA5-Land air temperature and dew point
predict the sky temperature the TESS-W infrared sensor reads under a clear sky, and in which form.

Clear candidates are chosen without the infrared and without any satellite, from the visible channel only: Sun
below -18 deg, Moon below -5 deg, and a brightness that changes by less than STABLE_MAG over three consecutive
5-minute bins (a stable sky; after the scatter criterion of Alarcon et al. 2021, which keeps about 57% of clear
time and admits about 9% of cloudy time). Stable overcast passes this test, so every fit trims the warm side:
after each fit, candidates more than TRIM_K robust standard deviations warmer than the model are dropped and the
fit is repeated (clouds only ever read warmer than the clear sky). Bins at the firmware floor are left out
(censored, see build_nights.py), and so are the months in which the sensor sees a cover (qc_ir.py, monthly test).

Three forms, fitted per photometer identifier, with one offset per device (the infrared sensor belongs to the
device: results/audit_devices.csv, from audit_devices.py, gives the device behind each monthly file):
  temp     sky = a_dev + b T                                  (air temperature only)
  linear   sky = a_dev + b T + c Td                           (air temperature and dew point)
  brunt    sky = T_K (A + B sqrt(e))^(1/4) - 273.15 + k_dev   (effective emissivity in the Brunt form, e the
                                                               vapour pressure in hPa from the dew point)
Each form is scored by FOLDS-fold cross-validation over interleaved calendar months (fold = month index mod
FOLDS, so every fold spans all seasons and years, and a device that ran for less than a year is still in every
training set): fitted on the other folds, it predicts the clear candidates of the left-out fold; the score is the
robust standard deviation (1.4826 x MAD) and the median of those residuals.

Writes results/baseline_explore.csv (photometer, form, years, candidates, held-out robust SD and median) and
results/baseline_explore.txt (summary over photometers).

Usage: python baseline_explore.py [--only stars645 ...]"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from qc_ir import obstructed_months

ROOT = Path(__file__).parent
NIGHTS = ROOT / "data" / "nights"
OUT_CSV = ROOT / "results" / "baseline_explore.csv"
OUT_TXT = ROOT / "results" / "baseline_explore.txt"
STABLE_MAG = 0.03
TRIM_K = 2.0
MIN_CANDIDATES = 300
FOLDS = 5
DEVICES = ROOT / "results" / "audit_devices.csv"


def vapour_hpa(td_c):
    return 6.112 * np.exp(17.62 * td_c / (243.12 + td_c))          # Magnus over water (WMO)


def attach_device(d: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Device (MAC tail) and cross-validation fold of each bin, from the month of its archive file."""
    seg = pd.read_csv(DEVICES)
    seg = seg[seg["id"] == uid]
    ym = d["time"].dt.strftime("%Y-%m")
    dev = pd.Series(index=d.index, dtype=object)
    for r in seg.itertuples():
        dev[(ym >= r.first) & (ym <= r.last)] = r.mac if isinstance(r.mac, str) else f"seg{r.segment}"
    d = d.copy()
    d["device"] = dev.fillna("unknown")
    d["fold"] = (d["time"].dt.year * 12 + d["time"].dt.month) % FOLDS
    return d


def covered_months(uid: str) -> set[str]:
    """YYYY-MM (UTC) months of the photometer flagged by the monthly obstruction test (qc_ir.py)."""
    return {m for i, m in obstructed_months() if i == uid}


def candidates(d: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Clear candidates of one photometer, outside the months in which its sensor sees a cover (a cover that lasts
    most of a record otherwise passes for clear sky in the fit)."""
    d = d.sort_values("time").copy()
    dm = d["mag_med"].diff().abs()
    gap = d["time"].diff() > pd.Timedelta(minutes=6)
    stable = (dm < STABLE_MAG) & (dm.shift(-1) < STABLE_MAG) & ~gap & ~gap.shift(-1, fill_value=True)
    covered = d["time"].dt.strftime("%Y-%m").isin(covered_months(uid))
    c = d[stable & ~covered & (d["sun_alt"] < -18) & (d["moon_alt"] < -5) & (d["floor_share"] < 0.5)
          & d[["sky_med", "t2m_C", "d2m_C"]].notna().all(axis=1)]
    return c


def onehot(c: pd.DataFrame, devices: list) -> np.ndarray:
    dev = c["device"].to_numpy() if "device" in c else np.full(len(c), "unknown", object)
    return np.column_stack([dev == k for k in devices]).astype(float)


def design(form: str, c: pd.DataFrame, devices: list):
    T, Td = c["t2m_C"].to_numpy(), c["d2m_C"].to_numpy()
    D = onehot(c, devices)
    if form == "temp":
        return np.column_stack([D, T])
    if form == "linear":
        return np.column_stack([D, T, Td])
    return np.column_stack([T + 273.15, np.sqrt(vapour_hpa(Td)), D])


def fit(form: str, c: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Parameters and the devices they cover (one offset per device seen in c)."""
    devices = sorted(c["device"].unique()) if "device" in c else ["unknown"]
    y = c["sky_med"].to_numpy()
    X = design(form, c, devices)
    keep = np.ones(len(y), bool)
    p = None
    for _ in range(6):
        if form == "brunt":
            def res(q, X=X[keep], y=y[keep]):
                eps = np.clip(q[0] + q[1] * X[:, 1], 0.05, 1.5)
                return X[:, 0] * eps ** 0.25 - 273.15 + X[:, 2:] @ q[2:] - y
            p = least_squares(res, p if p is not None else [0.6, 0.05] + [0.0] * len(devices), loss="soft_l1").x
        else:
            p = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - predict(form, (p, devices), c)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = r < np.median(r[keep]) + TRIM_K * s
        if new.sum() == keep.sum():
            break
        keep = new
    return p, devices


def predict(form: str, model: tuple[np.ndarray, list], c: pd.DataFrame) -> np.ndarray:
    """Expected clear-sky temperature; NaN for a device the fit did not see."""
    p, devices = model
    X = design(form, c, devices)
    nd = len(devices)
    if form == "brunt":
        eps = np.clip(p[0] + p[1] * X[:, 1], 0.05, 1.5)
        out = X[:, 0] * eps ** 0.25 - 273.15 + X[:, 2:] @ p[2:]
        seen = X[:, 2:].sum(axis=1) > 0
    else:
        out = X @ p
        seen = X[:, :nd].sum(axis=1) > 0
    return np.where(seen, out, np.nan)


def robust(r: np.ndarray) -> tuple[float, float]:
    med = float(np.median(r))
    return 1.4826 * float(np.median(np.abs(r - med))), med


def cv_residuals(form: str, c: pd.DataFrame) -> list:
    """Held-out residuals of the clear core of each fold (folds with fewer than 50 candidates are skipped)."""
    res = []
    for k in range(FOLDS):
        tr, te = c[c["fold"] != k], c[c["fold"] == k]
        if len(te) < 50 or len(tr) < MIN_CANDIDATES:
            continue
        try:
            model = fit(form, tr)
        except Exception:
            continue
        r = te["sky_med"].to_numpy() - predict(form, model, te)
        r = r[np.isfinite(r)]
        if len(r):
            res.append(r[r < np.median(r) + TRIM_K * robust(r)[0]])     # the clear core of the left-out fold
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rows = []
    for f in sorted(NIGHTS.glob("*.csv.gz")):
        uid = f.name.split(".")[0]
        if a.only and uid not in a.only:
            continue
        c = candidates(attach_device(pd.read_csv(f, parse_dates=["time"]), uid), uid)
        if len(c) < MIN_CANDIDATES:
            continue
        for form in ("temp", "linear", "brunt"):
            res = cv_residuals(form, c)
            if res:
                sd, med = robust(np.concatenate(res))
                rows.append([uid, form, c["device"].nunique(), len(c), round(sd, 2), round(med, 2)])
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "form", "devices", "candidates", "heldout_robust_sd_C", "heldout_median_C"])
        w.writerows(rows)
    t = pd.DataFrame(rows, columns=["id", "form", "devices", "candidates", "sd", "med"])
    lines = [f"photometers scored: {t['id'].nunique()}"]
    for form, g in t.groupby("form"):
        lines.append(f"{form:7s} held-out robust SD: median {g['sd'].median():.2f} C, IQR {g['sd'].quantile(.25):.2f}-"
                     f"{g['sd'].quantile(.75):.2f}; |median residual| median {g['med'].abs().median():.2f} C")
    w = t.pivot(index="id", columns="form", values="sd").dropna()
    lines.append(f"best form per photometer: {w.idxmin(axis=1).value_counts().to_dict()}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
