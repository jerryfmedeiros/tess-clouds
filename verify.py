"""verify.py — check that the shipped outputs still match what the code produces.

Each target is a script run in a scratch copy of the code: the scripts, results/, tables/ and an empty figures/
are copied, and data/ is a folder of links to the shipped data. The data folders a target writes (data/features,
data/metar, data/sat) are copied instead of linked, or created empty for data/features, so the shipped files are
never written. The files the target writes are then compared with the shipped copies: byte for byte, except that
gzip files are compared after decompression (the gzip header carries the time it was written). Exit status 0 only
if every target matches.

Targets run from the derived data: the night bins (data/nights, from build_nights.py), the extracted satellite rows
(data/goes, data/msg, data/viirs), the METAR reports (data/metar), the ERA5 series (data/era5_ts, data/mountain,
data/mountain_pl) and the caches of the cloud-top tests (data/cloudtop, data/viirs_why_*). A target whose inputs
are absent is skipped and named. mountain_below.py also reads the raw ERA5 pressure-level files
(era5pl_arco and, at the Canary photometers, era5pl_arco_canary, under TESS_CLOUDS_RAW, default data/raw). Scripts that read the raw TESS-W archive, download satellite
granules or call a web service (fetch_*.py, *_extract.py, viirs_local.py, viirs_v2_backfill.py, viirs_added_sites.py, audit_*.py other than
audit_country.py,
build_nights.py, mountain_pl.py, cloud_top.py --ref, terrain_boxes.py, arco_check.py) are not targets; run_all.py
--raw lists them. features.py and sat_score.py are slow (marked; skip them with --quick).

Usage:
  python verify.py              # check every target, each on the shipped outputs of the stages before it
  python verify.py --only sat   # check targets whose name contains a substring
  python verify.py --list       # list targets, their outputs and inputs, and stop
  python verify.py --quick      # leave out the slow targets
  python verify.py --diff       # show a unified diff of text files on mismatch
  python verify.py --tmp DIR    # put the scratch copies in DIR
  python verify.py --chain      # run every target in order in one scratch copy, so each stage reads the outputs
                                # regenerated before it, then compare every target's outputs"""
from __future__ import annotations

import argparse
import difflib
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
RAW = Path(os.environ.get("TESS_CLOUDS_RAW", ROOT / "data" / "raw"))   # raw files, outside git
WRITTEN = ("features", "metar", "sat", "record")   # data folders that targets write
COPIED = ("viirs_why_box.csv", "viirs_why_era5.csv.gz")   # caches viirs_why.py extends when incomplete

# (label, argv, outputs (globs relative to the code folder), inputs that must exist, slow)
TARGETS: list[tuple[str, list[str], list[str], list[str], bool]] = [
    ("qc_ir", ["qc_ir.py"], ["results/qc_ir.csv", "results/qc_ir_months.csv"], ["data/nights"], False),
    ("features", ["features.py"], ["data/features/*.csv.gz", "results/features_units.csv"],
     ["data/nights", "data/mountain_pl", "results/audit_devices.csv", "results/qc_ir_months.csv"], True),
    ("qc_baseline", ["qc_baseline.py"], ["results/qc_baseline.csv"], [], False),
    ("die_air", ["die_air.py"], ["results/die_air.txt"], ["data/nights"], False),
    ("audit_country", ["audit_country.py"], ["results/audit_country.csv"], ["data/tess_archive/selected.csv"], False),
    ("baseline_explore", ["baseline_explore.py"], ["results/baseline_explore.csv", "results/baseline_explore.txt"],
     ["data/nights", "results/qc_ir_months.csv"], False),
    ("metar_labels", ["metar_labels.py"], ["data/metar/labels/*.csv.gz", "results/metar_labels.csv"],
     ["data/features", "data/metar"], False),
    ("classify_metar", ["classify_metar.py"],
     ["results/classify_metar.csv", "results/classify_metar_thresholds.csv", "results/classify_metar_repeats.csv",
      "results/classify_metar_airports.csv", "results/classify_metar.txt", "data/metar/scored.csv.gz"],
     ["data/features", "data/metar/labels"], False),
    ("breakdown_metar", ["breakdown_metar.py"],
     ["results/breakdown_metar.csv", "results/breakdown_metar_sites.csv", "results/breakdown_metar_distance.csv",
      "results/breakdown_metar.txt"],
     ["data/metar/scored.csv.gz", "data/nights"], False),
    ("sat_score", ["sat_score.py"],
     ["data/sat/*.csv.gz", "data/sat/by_id/*/*.csv.gz", "results/sat_score.csv", "results/sat_score_sites.csv",
      "results/sat_score.txt"], ["data/features", "data/goes", "data/msg", "data/viirs"], True),
    ("sat_score_dem", ["sat_score.py", "--dem"],
     ["results/sat_score_dem.csv", "results/sat_score_sites_dem.csv", "results/sat_score_dem.txt"],
     ["data/sat", "results/qc_ir_months.csv"], False),
    ("sat_sensitivity", ["sat_sensitivity.py"], ["results/sat_sensitivity.csv", "results/sat_sensitivity.txt"],
     ["data/sat", "data/features", "data/goes", "data/msg"], True),
    ("sat_vs_metar", ["sat_vs_metar.py"], ["results/sat_vs_metar.csv", "results/sat_vs_metar.txt"],
     ["data/metar/scored.csv.gz", "data/sat"], False),
    ("cloud_record", ["cloud_record.py"], ["data/record/*.csv.gz", "results/cloud_record.csv", "results/cloud_record.txt"],
     ["data/features", "results/sat_score_sites_dem.csv", "data/elevation.csv", "results/qc_ir_months.csv"], False),
    ("nsb_selection", ["nsb_selection.py"],
     ["data/metar/nsb_nights.csv.gz", "results/nsb_selection.csv", "results/nsb_selection.txt", "results/nsb_thresholds.csv"],
     ["data/nights", "data/features", "data/metar/labels", "data/tess_archive/stars721"], False),   # raw minutes of the METAR photometers
    ("cloud_base", ["cloud_base.py"],
     ["data/metar/cloud_base.csv.gz", "results/cloud_base.csv", "results/cloud_base.txt"],
     ["data/features", "data/metar/labels"], False),
    ("mountain_baseline", ["mountain_baseline.py"], ["results/mountain_baseline.csv", "results/mountain_baseline.txt"],
     ["data/nights", "data/mountain", "data/mountain_pl", "data/era5_ts", "results/qc_ir_months.csv"], False),
    ("mountain_network", ["mountain_baseline.py", "--network"],
     ["results/mountain_network.csv", "results/mountain_network.txt"],
     ["data/nights", "data/mountain_pl", "data/era5_ts", "results/qc_ir_months.csv"], False),
    ("mountain_below", ["mountain_below.py"], ["results/mountain_below.csv.gz", "results/mountain_below.txt"],
     ["data/sat", str(RAW / "era5pl_arco")], False),
    ("cloud_top", ["cloud_top.py", "--summary"], ["results/cloud_top.txt"], ["data/cloudtop"], False),
    ("viirs_why", ["viirs_why.py"], ["results/viirs_why.csv", "results/viirs_why.txt"],
     ["data/features", "data/cloudtop/viirs.csv", "data/viirs_why_box.csv", "data/viirs_why_era5.csv.gz"], False),
    ("tables", ["tables.py"], ["tables/*.tex", "results/tables.txt"], ["data/metar/scored.csv.gz", "data/sat", "results/qc_ir_months.csv"], False),
    ("fig_roc", ["fig_roc.py"], ["figures/fig_roc.png", "results/fig_roc.txt"], ["data/metar/scored.csv.gz"], False),
    ("fig_breakdown", ["fig_breakdown.py"], ["figures/fig_conditions.png", "results/fig_breakdown.txt"], [], False),
    ("fig_example_night", ["fig_example_night.py"], ["figures/fig_example_night.png", "results/fig_example_night.txt"],
     ["data/features", "data/metar/scored.csv.gz", "data/sat/msg.csv.gz", "results/classify_metar_thresholds.csv"], False),
    ("fig_sat_sites", ["fig_sat_sites.py"], ["figures/fig_sat_sites.png", "results/fig_sat_sites.txt"], [], False),
]


def missing(target) -> list[str]:
    """The inputs of a target that are absent."""
    return [p for p in target[3] if not (Path(p) if p.startswith("/") else ROOT / p).exists()]


def scratch_copy(work: Path, written: set[str]) -> Path:
    """The code in `work`: scripts, results/ and tables/ copied, figures/ empty, data/ linked entry by entry,
    except the data folders in `written` (copied; data/features created empty) and the caches in COPIED."""
    for f in ROOT.glob("*.py"):
        shutil.copy2(f, work / f.name)
    for d in ("results", "tables"):
        shutil.copytree(ROOT / d, work / d)
    (work / "figures").mkdir()
    if (ROOT / "fonts").exists():
        (work / "fonts").symlink_to(ROOT / "fonts")
    (work / "data").mkdir()
    for e in (ROOT / "data").iterdir():
        dst = work / "data" / e.name
        if e.name == "features" and "features" in written:
            dst.mkdir()
        elif e.name in written or e.name in COPIED:
            (shutil.copytree if e.is_dir() else shutil.copy2)(e, dst)
        else:
            dst.symlink_to(e)
    return work


def written_by(targets) -> set[str]:
    return {o.split("/")[1] for t in targets for o in t[2] if o.startswith("data/")} & set(WRITTEN)


def run(argv: list[str], cwd: Path) -> tuple[bool, str]:
    """Run a script in cwd; its own output goes nowhere, the tail of stderr is kept for errors."""
    p = subprocess.run([sys.executable, "-W", "ignore", *argv], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       cwd=cwd, text=True)
    return p.returncode == 0, p.stderr[-600:] if p.returncode else ""


def content(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def compare(target, work: Path, show_diff: bool) -> tuple[int, list[str]]:
    """Files of the target's outputs that match, and a report line for each that does not."""
    ok, bad = 0, []
    for pattern in target[2]:
        want = {p.relative_to(ROOT) for p in ROOT.glob(pattern)}
        got = {p.relative_to(work) for p in work.glob(pattern)}
        for rel in sorted(want - got):
            bad.append(f"not regenerated  {rel}")
        for rel in sorted(got - want):
            bad.append(f"not shipped      {rel}")
        for rel in sorted(want & got):
            a, b = content(ROOT / rel), content(work / rel)
            if a == b:
                ok += 1
                continue
            bad.append(f"differs          {rel}")
            if show_diff and rel.suffix in (".txt", ".csv", ".tex", ".gz") and rel.suffix != ".png":
                for line in list(difflib.unified_diff(a.decode().splitlines(), b.decode().splitlines(), "shipped",
                                                      "regenerated", lineterm=""))[:30]:
                    bad.append(f"    {line}")
    return ok, bad


def report(label: str, ok: int, bad: list[str], secs: float) -> None:
    if bad:
        print(f"  DIFFERS  {label:18s} {ok} file(s) match, {sum(not b.startswith('    ') for b in bad)} do not "
              f"({secs:.0f} s)")
        for b in bad:
            print(f"           {b}")
    else:
        print(f"  ok       {label:18s} {ok} file(s) ({secs:.0f} s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="check only targets whose label contains this substring")
    ap.add_argument("--list", action="store_true", help="list targets and stop")
    ap.add_argument("--quick", action="store_true", help="leave out the slow targets (features, sat_score, sat_sensitivity)")
    ap.add_argument("--diff", action="store_true", help="show a unified diff of text files on mismatch")
    ap.add_argument("--chain", action="store_true", help="run every target in order in one scratch copy")
    ap.add_argument("--tmp", type=Path, help="folder for the scratch copies (default: the system's; data/features "
                                             "regenerated needs about 2 GB)")
    a = ap.parse_args()
    if a.tmp:
        Path(a.tmp).mkdir(parents=True, exist_ok=True)

    chosen = [t for t in TARGETS if (not a.only or a.only in t[0]) and not (a.quick and t[4])]
    if a.list:
        for t in chosen:
            note = (" (slow)" if t[4] else "") + (f"  skipped: needs {', '.join(missing(t))}" if missing(t) else "")
            print(f"  {t[0]:18s} python {' '.join(t[1])}{note}\n  {'':18s} -> {', '.join(t[2])}")
        return 0
    targets = [t for t in chosen if not missing(t)]
    skipped = [t for t in chosen if missing(t)]
    failed = 0

    if a.chain:
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            work = scratch_copy(Path(tmp), written_by(targets))
            for t in targets:
                t0 = time.time()
                ran, err = run(t[1], work)
                if not ran:
                    print(f"  ERROR    {t[0]:18s} python {' '.join(t[1])}\n           {err.strip()}", flush=True)
                    failed += 1
                    continue
                ok, bad = compare(t, work, a.diff)
                report(t[0], ok, bad, time.time() - t0)
                failed += bool(bad)
    else:
        for t in targets:
            t0 = time.time()
            with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
                work = scratch_copy(Path(tmp), written_by([t]))
                ran, err = run(t[1], work)
                if not ran:
                    print(f"  ERROR    {t[0]:18s} python {' '.join(t[1])}\n           {err.strip()}", flush=True)
                    failed += 1
                    continue
                ok, bad = compare(t, work, a.diff)
            report(t[0], ok, bad, time.time() - t0)
            failed += bool(bad)

    for t in skipped:
        print(f"  skip     {t[0]:18s} needs {', '.join(missing(t))}")
    print(f"\n{len(targets) - failed} matched, {failed} differed or errored, {len(skipped)} skipped.")
    if failed:
        print("A mismatch means the shipped outputs no longer match the code. Find what moved before trusting either.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
