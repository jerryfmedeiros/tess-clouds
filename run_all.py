"""run_all.py — rebuild the data, results, tables and figures, in pipeline order.

Stages 1-3 read inputs that are not shipped (the raw STARS4ALL archive, satellite granules, a EUMETSAT and an
Earthdata account, the Copernicus CDS and ARCO-ERA5, Open Topo Data, the raw files under TESS_CLOUDS_RAW); they are listed, and
run only with --raw. Stage 4 runs the targets of verify.py in order, in place: each writes its outputs into data/,
results/, tables/ and figures/. Review the diff before committing.

Usage:
  python run_all.py                  # stage 4 from the shipped data
  python run_all.py --list           # print every stage and command, run nothing
  python run_all.py --raw            # stages 1-3 first (see the README for their inputs and accounts)
  python run_all.py --from tables    # resume stage 4 at a target
  python run_all.py --quick          # stage 4 without the slow targets (features, sat_score)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import verify

KEEP = str(verify.RAW)
# Photometers added to the selection after the first VIIRS extraction (no VIIRS row before 2023-08 although they
# have readings): viirs_added_sites.py fills in their earlier months.
ADDED_SITES = ["stars1036", "stars1051", "stars219", "stars220", "stars221", "stars238", "stars247", "stars268", "stars270", "stars276", "stars29", "stars290", "stars356", "stars359", "stars391", "stars397", "stars509", "stars511", "stars517", "stars610", "stars727", "stars992"]
RAW = [
    ("1. Archive and audits (raw STARS4ALL archive)", [
        ["fetch_tess_archive.py", "--scan"], ["fetch_tess_archive.py", "--select"],
        ["fetch_tess_archive.py", "--download"], ["fetch_tess_archive.py", "--splits"],
        ["audit_ir.py"], ["audit_devices.py"], ["audit_positions.py"], ["audit_sequence.py"], ["audit_placeholder.py"],
        ["fetch_elevation.py"],
    ]),
    ("2. References and meteorology (IEM, NOAA, EUMETSAT, LAADS, CDS, ARCO-ERA5, NOAA ISD)", [
        ["fetch_metar.py", "--pairs"], ["fetch_metar.py", "--download"], ["fetch_metar.py", "--auto"],
        ["fetch_era5_ts.py"],
        ["goes_extract.py", "--keep", KEEP + "/goes"], ["msg_extract.py", "--keep", KEEP + "/msg"],
        ["viirs_extract.py", "--keep", KEEP + "/viirs"], ["viirs_local.py", "--keep", KEEP],
        ["viirs_v2_backfill.py", "--keep", KEEP],
        ["viirs_added_sites.py", "--keep", KEEP, "--only", *ADDED_SITES],
        ["fetch_mountain_met.py", "--isd"], ["fetch_mountain_met.py", "--era5", "--keep", KEEP],
        ["fetch_mountain_met.py", "--era5-sites", "--arco", "--keep", KEEP],
        ["fetch_mountain_met.py", "--arco-canary", "--keep", KEEP],
        ["mountain_pl.py", "--keep", KEEP],
    ]),
    ("3. Night bins (raw archive and ERA5-Land series)", [
        ["build_nights.py"],
    ]),
]
# Raw-input steps that follow a stage-4 target, because they read its outputs.
AFTER = {
    "mountain_below": [["cloud_top.py", "--ref", r, "--keep", KEEP] for r in ("goes", "msg", "viirs")]
                      + [["terrain_boxes.py"]],
}


def run(argv: list[str]) -> bool:
    ok = subprocess.run([sys.executable, "-W", "ignore", *argv], cwd=verify.ROOT).returncode == 0
    print(f"  {'ran' if ok else 'ERROR':6s} python {' '.join(argv)}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the stages and commands and stop")
    ap.add_argument("--raw", action="store_true", help="run stages 1-3 too")
    ap.add_argument("--from", dest="start", metavar="TARGET", help="resume stage 4 at this verify target")
    ap.add_argument("--quick", action="store_true", help="leave out the slow targets")
    a = ap.parse_args()
    targets = [t for t in verify.TARGETS if not (a.quick and t[4])]

    if a.list:
        for title, steps in RAW:
            print(title)
            for argv in steps:
                print(f"    python {' '.join(argv)}")
        print("4. Analysis (the targets of verify.py, in order)")
        for t in targets:
            need = verify.missing(t)
            print(f"    python {' '.join(t[1])}" + (f"   (skipped: needs {', '.join(need)})" if need else ""))
            for argv in AFTER.get(t[0], []):
                print(f"    python {' '.join(argv)}   (with --raw)")
        return 0

    bad = 0
    if a.raw and not a.start:
        for title, steps in RAW:
            print(title)
            for argv in steps:
                bad += not run(argv)
    print("4. Analysis")
    started = a.start is None
    for t in targets:
        started = started or t[0] == a.start
        if not started:
            continue
        need = verify.missing(t)
        if need:
            print(f"  skip   python {' '.join(t[1])} (needs {', '.join(need)})")
            continue
        bad += not run(t[1])
        if a.raw:
            for argv in AFTER.get(t[0], []):
                bad += not run(argv)
    print(f"\n{'done' if not bad else f'{bad} step(s) failed'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
