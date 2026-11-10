"""tables.py — the rows of the manuscript's tables, written from the results files so that no table value is typed.

Each table is written as LaTeX rows only (tables/<name>.tex); the column heads, rules and captions are in the
manuscript. Values are rounded once, here, from the unrounded (6-decimal) results.
  tab_data     the references: coverage, period, photometers and comparisons (classify_metar.txt, sat_score.txt for
               the matched rows and photometers, data/sat/<ref>.csv.gz and data/metar/scored.csv.gz for the periods);
  tab_qc       the quality checks, what each removes or flags, and how many (data/excluded_ir.csv, results/qc_ir.csv,
               results/audit_devices_ids.csv, data/excluded_months.csv, data/excluded_positions.csv,
               data/station_fixes.csv, results/qc_baseline.csv);
  tab_metar    the classifier and comparison methods against automated METAR (results/classify_metar.csv), with the
               night-block and the airport-cluster interval, then the two-test classifier at the CAVOK/NSC airports
               (results/sat_score_dem.csv, reference "metar");
  tab_agree    the satellite masks, METAR and the classifier against each other (results/sat_vs_metar.csv), one row
               of scores with night-block intervals and one of photometer-cluster intervals per satellite;
  tab_sat      the classifier against the satellite masks by subset, with mountain sites from the terrain model and
               the METAR photometers as a subset (results/sat_score_dem.csv), both intervals;
  counts       values the text quotes outside the tables, plain only: the countries of the METAR airports, with
               photometers and airports in each (results/metar_labels.csv, data/metar/pairs.csv), and the
               photometers with a clear-sky baseline, the cloud record (results/qc_baseline.csv).
Writes tables/*.tex and results/tables.txt (the same values, plain).

Usage: python tables.py"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from qc_ir import excluded_ids

ROOT = Path(__file__).parent
OUT = ROOT / "tables"
TXT = ROOT / "results" / "tables.txt"
METHODS = [("two_test", "Two-test classifier"), ("z_brunt", "Clear-sky baseline, Brunt form"),
           ("z_linear", "Clear-sky baseline, linear form"), ("neg_dT", r"Die minus sky ($\Delta T$)"),
           ("sky_med", "Sky temperature"), ("s5_excess", "Scatter within the 5~min bin"),
           ("dfa_f20", "Detrended fluctuation at 20~min (DFA)"), ("d30", "Scatter of bin-to-bin changes over 30~min"),
           ("vis15", "Brightness range over 15~min"), ("dashboard", r"Network dashboard (cloudy if $\Delta T < 15$~K)"),
           ("alarcon", r"\citet{Alarcon2021NaturalMinimum} (clear if $\Delta T > 54.5$~K)"), ("always_clear", "Always clear")]
REFS = [("goes", "GOES ABI", "Americas"), ("msg", "Meteosat", "Europe, Africa"), ("viirs", "VIIRS NOAA-20", "global")]
SHORT = {"goes": "GOES ABI", "msg": "Meteosat", "viirs": "VIIRS"}      # Tables 4 and 5 (NOAA-20 is stated in Table 2)
# Table 3 groups, a rule and a named header row before each (BA order within each)
GROUP_START = {"z_brunt": "Sky temperature level", "s5_excess": "Sky temperature variability",
               "vis15": "Sky brightness", "dashboard": "Fixed rules"}
# Flagged photometer, unflagged photometer at the same coordinates (Quebec; weak-baseline flag)
COLOCATED_KM = 0.2
COLOCATED = [("stars472", "stars979"), ("stars474", "stars981"), ("stars475", "stars982"), ("stars657", "stars237")]


def n(x) -> str:
    """Integer with a thin space between thousands (Copernicus style)."""
    s = f"{int(x):d}"
    return s if len(s) <= 4 else re.sub(r"(\d)(?=(\d{3})+$)", r"\1\\,", s)


def f3(x) -> str:
    return f"{float(x):.3f}"


def ci(lo, hi) -> str:
    return f"[{f3(lo)}--{f3(hi)}]"


def cl(lo, hi) -> str:
    """The cluster interval, in parentheses."""
    return f"({f3(lo)}--{f3(hi)})"


def write(name: str, rows: list[str], plain: list[str], log: list[str]):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(rows) + "\n")
    log += [name] + ["  " + p for p in plain]


def month(t) -> str:
    return pd.Timestamp(t).strftime("%b %Y")


def tab_data(log):
    head = (ROOT / "results" / "classify_metar.txt").read_text().splitlines()[0]
    m = re.match(r"photometers (\d+) at (\d+) airports, reports scored (\d+).*nights (\d+)", head)
    phot, airports, reports, nights = map(int, m.groups())
    t = pd.read_csv(ROOT / "data" / "metar" / "scored.csv.gz", usecols=["time"], parse_dates=["time"])["time"]
    rows = [f"Automated METAR & {airports} airports & {month(t.min())}--{month(t.max())} & {phot} & {n(reports)} reports"
            r" \\"]
    plain = [f"METAR {airports} airports {t.min():%Y-%m}..{t.max():%Y-%m} photometers {phot} reports {reports} nights {nights}"]
    sat = (ROOT / "results" / "sat_score.txt").read_text()
    for ref, name, area in REFS:
        mm = re.search(rf"^{ref}: (\d+) matched rows, (\d+) photometers", sat, re.M)
        matched, p = int(mm.group(1)), int(mm.group(2))
        st = pd.read_csv(ROOT / "data" / "sat" / f"{ref}.csv.gz", usecols=["time"], parse_dates=["time"])["time"]
        rows.append(f"{name} cloud mask & {area} & {month(st.min())}--{month(st.max())} & {p} & {n(matched)} scans \\\\")
        plain.append(f"{ref} {area} {st.min():%Y-%m}..{st.max():%Y-%m} photometers {p} matched {matched}")
    write("tab_data", rows, plain, log)


def ids(path: Path, col: str = "id", where=None) -> list[str]:
    rows = list(csv.DictReader(l for l in path.open() if not l.startswith("#")))
    return [r[col] for r in rows if where is None or where(r)]


def tab_qc(log):
    d = ROOT / "data"
    r = ROOT / "results"
    broken = ids(d / "excluded_ir.csv")
    obstructed = ids(r / "qc_ir.csv", where=lambda x: x["obstructed"] == "1")
    multi = ids(r / "audit_devices_ids.csv", where=lambda x: int(x["devices"]) > 1)
    seq = list(csv.DictReader(l for l in (d / "excluded_months.csv").open() if not l.startswith("#")))
    pos = list(csv.DictReader(l for l in (d / "excluded_positions.csv").open() if not l.startswith("#")))
    moved = [x for x in pos if x["month"] != "all"]
    placeholder = [x["id"] for x in pos if x["month"] == "all"]
    fixes = ids(d / "station_fixes.csv")
    qb = pd.read_csv(r / "qc_baseline.csv")
    weak, fence = int(qb["weak"].sum()), float(qb["fence"].iloc[0])
    units = pd.read_csv(r / "features_units.csv")
    no_baseline = sorted(set(units.loc[units["sd_brunt"].isna(), "id"]) - excluded_ids())
    months = pd.read_csv(r / "qc_ir_months.csv")
    months = months[(months["obstructed_month"] == 1) & ~months["id"].isin(excluded_ids())]
    items = [
        ("Infrared never usable", "error code or no reading in nearly every reading", "photometer excluded",
         f"{len(broken)} photometers"),
        ("Obstructed field of view", r"95th percentile of ERA5-Land air minus sky temperature below 8~K", "photometer excluded",
         f"{len(obstructed)} photometers"),
        ("Obstructed in some months", r"the same test in each calendar month with at least 500 dark bins",
         "months flagged and left out of the baseline fit; scores given with and without", f"{len(months)} months, {months['id'].nunique()} photometers"),
        ("Device change", "MAC address in the monthly header, checked against the network register",
         "one baseline offset per device", f"{len(multi)} photometers with two or more devices"),
        ("Two devices under one name", "message counters running at the same time", "months removed",
         f"{len(seq)} months, {len({x['id'] for x in seq})} photometer"),
        ("Recorded away from the position used", "header position more than 5~km away, or none for a device that never states it",
         "months removed", f"{len(moved)} months, {len({x['id'] for x in moved})} photometers"),
        ("Placeholder position", "one position shared by photometers whose skies differ", "photometer excluded",
         f"{len(placeholder)} photometers"),
        ("Position typing error", "latitude or longitude evidently mistyped in the archive", "position corrected",
         f"{len(fixes)} photometers"),
        ("No clear-sky baseline", "no ERA5-Land cell at the site, or no clear candidate bin", "photometer excluded",
         f"{len(no_baseline)} photometers"),
        ("Weak clear-sky baseline", rf"held-out clear-sky SD above the network's upper Tukey fence ({fence:.2f}~K)",
         "scores given with and without", f"{weak} photometers"),
    ]
    rows = [" & ".join(x) + r" \\" for x in items]
    write("tab_qc", rows, [" | ".join(x) for x in items], log)


def tab_metar(log):
    c = pd.read_csv(ROOT / "results" / "classify_metar.csv")
    v = {(r.method, r.metric): r for r in c.itertuples()}
    rows, plain = [], []
    for m, label in METHODS:
        if m in GROUP_START:
            rows += [r"\middlehline", rf"\multicolumn{{7}}{{l}}{{\textit{{{GROUP_START[m]}}}}} \\"]
        ba, sp = v[(m, "balanced_accuracy")], v[(m, "balanced_accuracy_splits")]
        hit, fa = v[(m, "hit_rate")].value, v[(m, "false_alarm")].value
        auc = v.get((m, "auc"))
        cu = v[(m, "balanced_accuracy_cluster")]
        fixed = m in ("dashboard", "alarcon", "always_clear")
        ba_s = f3(ba.value) if (fixed and f3(ba.ci_lo) == f3(ba.ci_hi)) else f"{f3(ba.value)} {ci(ba.ci_lo, ba.ci_hi)}"
        cu_s = "--" if (fixed and f3(cu.ci_lo) == f3(cu.ci_hi)) else cl(cu.ci_lo, cu.ci_hi)
        sp_s = f3(sp.value) if f3(sp.ci_lo) == f3(sp.ci_hi) else f"{f3(sp.value)}, {f3(sp.ci_lo)}--{f3(sp.ci_hi)}"   # mean, range
        auc_s = f3(auc.value) if auc is not None else "--"
        rows.append(f"{label} & {ba_s} & {cu_s} & {sp_s} & {f3(hit)} & {f3(fa)} & {auc_s} \\\\")
        plain.append(f"{m}: BA {ba_s} cluster {cu_s} splits {sp_s} hit {f3(hit)} FA {f3(fa)} AUC {auc_s}")
    a = pd.read_csv(ROOT / "results" / "sat_score_dem.csv")
    rows += [r"\middlehline",                    # the CAVOK/NSC block, not comparable with the rows above
             r"\multicolumn{7}{l}{\textit{Two-test classifier, CAVOK and NSC counted clear where the satellite is clear}} \\"]
    for sub, label in (("cavok_all", "All {} photometers"),
                       ("cavok_left", "{} photometers with under 100 clear reports")):
        r = a[(a["reference"] == "metar") & (a["subset"] == sub) & (a["method"] == "two_test")]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(f"{label.format(r.photometers)} & {f3(r.balanced_accuracy)} {ci(r.ci_lo, r.ci_hi)} & "
                    f"{cl(r.ci_lo_phot, r.ci_hi_phot)} & -- & {f3(r.hit)} & {f3(r.false_alarm)} & -- \\\\")
        plain.append(f"{sub}: photometers {r.photometers} reports {r.references} BA {f3(r.balanced_accuracy)} "
                     f"{ci(r.ci_lo, r.ci_hi)} cluster {cl(r.ci_lo_phot, r.ci_hi_phot)} hit {f3(r.hit)} FA {f3(r.false_alarm)}")
    t = (ROOT / "results" / "classify_metar.txt").read_text().splitlines()
    plain += [l for l in t if l.startswith(("thinned", "per airport"))]
    write("tab_metar", rows, plain, log)


def tab_agree(log):
    a = pd.read_csv(ROOT / "results" / "sat_vs_metar.csv")
    order = ["satellite vs METAR", "classifier vs METAR", "classifier vs satellite", "classifier vs consensus"]
    rows, plain = [], []
    for ref, name, _ in REFS:
        g = a[a["reference"] == ref].set_index("comparison")
        cells = [f"{f3(g.loc[k, 'balanced_accuracy'])} {ci(g.loc[k, 'ci_lo'], g.loc[k, 'ci_hi'])}" for k in order]
        clus = [cl(g.loc[k, "ci_lo_phot"], g.loc[k, "ci_hi_phot"]) for k in order]
        rows.append(f"{SHORT[ref]} & {n(g.loc[order[0], 'reports'])} & " + " & ".join(cells) + r" \\")
        rows.append(r"\quad cluster interval & & " + " & ".join(clus) + r" \\")
        plain.append(f"{ref} reports {int(g.loc[order[0], 'reports'])} consensus {int(g.loc[order[3], 'reports'])}: "
                     + " | ".join(cells) + " ; cluster " + " | ".join(clus))
    write("tab_agree", rows, plain, log)


SUBSETS = [("primary", "lowland"), ("primary_qc", r"\quad flag out"), ("metar", r"\quad METAR"),
           ("snow", "snow"), ("mountain", "mountain")]


def tab_sat(log):
    a = pd.read_csv(ROOT / "results" / "sat_score_dem.csv")
    v = {(r.reference, r.subset, r.method): r for r in a.itertuples()}
    rows, plain = [], []
    for ref, name, _ in REFS:
        for k, (sub, label) in enumerate(SUBSETS):
            t, z, db = v[(ref, sub, "two_test")], v[(ref, sub, "z_brunt")], v[(ref, sub, "dashboard")]
            cells = [SHORT[ref] if k == 0 else "", label, str(t.photometers), n(t.references),
                     f"{f3(t.balanced_accuracy)} {ci(t.ci_lo, t.ci_hi)}", cl(t.ci_lo_phot, t.ci_hi_phot), f3(t.hit),
                     f3(t.false_alarm), f3(z.balanced_accuracy), f3(db.balanced_accuracy)]
            rows.append(" & ".join(cells) + r" \\")
            plain.append(f"{ref} {sub}: photometers {t.photometers} scans {t.references} two-test "
                         f"{f3(t.balanced_accuracy)} {ci(t.ci_lo, t.ci_hi)} cluster {cl(t.ci_lo_phot, t.ci_hi_phot)} "
                         f"hit {f3(t.hit)} FA {f3(t.false_alarm)} baseline {f3(z.balanced_accuracy)} dashboard {f3(db.balanced_accuracy)}")
    write("tab_sat", rows, plain, log)


def counts(log):
    m = pd.read_csv(ROOT / "results" / "metar_labels.csv")
    country = pd.read_csv(ROOT / "data" / "metar" / "pairs.csv").drop_duplicates("icao").set_index("icao")["country"]
    g = m.assign(country=m["icao"].map(country)).groupby("country").agg(photometers=("id", "size"),
                                                                       airports=("icao", "nunique"))
    g = g.sort_values(["photometers", "airports"], ascending=False)
    log.append(f"counts: METAR countries {len(g)}: " + ", ".join(f"{c} {r.photometers} photometers {r.airports} airports"
                                                                  for c, r in g.iterrows()))
    q = pd.read_csv(ROOT / "results" / "qc_baseline.csv")
    log.append(f"counts: photometers with a clear-sky baseline {len(q)}, of which weak-baseline flag {int(q['weak'].sum())}")
    sd = q.set_index("id")["sd_brunt"]
    ratios = [(a, b, sd[a] / sd[b]) for a, b in COLOCATED if a in sd and b in sd]
    log.append("counts: co-located pairs, flagged over unflagged clear-sky SD: " + ", ".join(
        f"{a}/{b} {sd[a]:.2f}/{sd[b]:.2f} = {r:.2f}" for a, b, r in ratios)
               + (f"; ratio {min(r for *_, r in ratios):.1f}-{max(r for *_, r in ratios):.1f}" if ratios else ""))
    pos = pd.read_csv(ROOT / "data" / "tess_archive" / "selected.csv").set_index("id").loc[sd.index, ["lat", "lon"]]
    wk = q.set_index("id")["weak"].astype(int)
    la, lo = np.radians(pos["lat"].to_numpy()), np.radians(pos["lon"].to_numpy())
    ids = list(pos.index)
    pairs = []
    for i in range(len(ids)):
        dkm = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la[i + 1:] - la[i]) / 2) ** 2
                                           + np.cos(la[i]) * np.cos(la[i + 1:]) * np.sin((lo[i + 1:] - lo[i]) / 2) ** 2))
        pairs += [(ids[i], ids[i + 1 + j]) for j in np.flatnonzero(dkm < COLOCATED_KM)]
    kind = {2: "both flagged", 1: "one flagged", 0: "neither flagged"}
    by = {k: [p for p in pairs if wk[p[0]] + wk[p[1]] == n] for n, k in kind.items()}
    log.append(f"counts: all pairs of record photometers within {COLOCATED_KM} km ({len(pairs)}): " + "; ".join(
        f"{k} {len(v)}" + (f" ({', '.join(f'{a}/{b} {sd[a]:.2f}/{sd[b]:.2f}' for a, b in v)})" if k != "neither flagged" else "")
        for k, v in by.items()))
    pole = pos["lat"].abs() >= 45
    log.append(f"counts: latitude 45 deg or more (north or south): weak-flagged {int((pole & (wk == 1)).sum())} of {int(wk.sum())} "
               f"({(pole & (wk == 1)).sum() / wk.sum():.2f}); all record photometers {int(pole.sum())} of {len(pole)} ({pole.mean():.2f})")
    units = pd.read_csv(ROOT / "results" / "features_units.csv")
    log.append("counts: photometers with no clear-sky baseline " + ", ".join(sorted(set(units.loc[units["sd_brunt"].isna(), "id"]) - excluded_ids())))


def main():
    log = []
    for t in (tab_data, tab_qc, tab_metar, tab_agree, tab_sat, counts):
        t(log)
    TXT.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
