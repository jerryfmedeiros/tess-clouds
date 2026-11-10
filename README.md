# Cloud detection from the TESS-W infrared sensor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24114227.svg)](https://doi.org/10.5281/zenodo.24114227)

Code and data release for "Night cloud detection from the infrared sensor of a low-cost light-pollution photometer
network" (Fernandes Medeiros, submitted to Atmospheric Measurement Techniques). A clear/cloudy classifier built from
the MLX90614 infrared channel of the STARS4ALL TESS-W photometers, trained on automated ceilometer reports and
tested against three night satellite cloud masks, and the cloud record it produces at 384 photometers.

Every number, table and figure of the paper is written by a script here from the files in `data/` and `results/`.
`verify.py` checks that the shipped outputs still match what the code produces.

## Quick start (the analysis from the shipped derived data)

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python verify.py --list          # the 26 targets, what each writes and needs
    python verify.py --quick         # every target except the three slow ones; about 35 min, 2 GB of scratch
    python verify.py                 # all targets (features ~30 min, sat_score ~10 min on 12 cores)
    python run_all.py                # the same targets run in place, rewriting results/, tables/ and figures/

`verify.py` runs each target in a scratch copy and compares the files it writes with the shipped copies byte for
byte (gzip files after decompression). A target whose inputs are absent is skipped and named: `mountain_below`
reads raw ERA5 pressure-level files (about 54 GB, not shipped; fetch_mountain_met.py --arco and --arco-canary rebuild them) and is skipped without them; its shipped outputs stand.
Scripts that read raw files take them from the folder named by the environment variable `TESS_CLOUDS_RAW`
(default `data/raw/`), or from `--keep`.

## Getting the data

The git repository holds the code, the results, the tables, the figures and the small data files (the selection,
the exclusions, the elevations, the frozen thresholds, the ERA5 pressure-level series). The derived data that
`verify.py` reads are in the data archive released with the paper (about 4.1 GB; Zenodo,
[10.5281/zenodo.24114227](https://doi.org/10.5281/zenodo.24114227)), one tar file per folder: `data_features.tar` (2.3 GB), `data_nights.tar` (1.0 GB), `data_record.tar`
(0.4 GB), `data_era5_ts.tar`, `data_metar.tar`, `data_msg.tar`, `data_sat.tar`, `data_mountain.tar`, `data_goes.tar`
and `data_viirs.tar`, plus `data_tess_archive_metar.tar.gz` (the unchanged STARS4ALL monthly files of the 58
METAR-paired photometers, read by `nsb_selection.py`), with their checksums in `SHA256SUMS`. Unpack them in the repository folder
(`for f in data_*.tar*; do tar -xf "$f"; done`) so that the folders sit under `data/`; without them
`verify.py --list` marks 17 of the 26 targets as skipped.

## What is where

| Folder | Content |
|---|---|
| `data/tess_archive/selected.csv` | the 406 photometers used (position, months), with the positions of `data/station_fixes.csv` applied |
| `data/station_fixes.csv` | the five mistyped archive positions and their corrections, with the evidence for each |
| `data/excluded_ir.csv`, `excluded_positions.csv`, `excluded_months.csv` | the exclusions of Table 1 with their evidence |
| `data/elevation.csv` | ASTER GDEM elevation of every photometer (fetch_elevation.py) |
| `data/frozen_thresholds.csv` | the classifier thresholds fixed before the satellite scoring (2026-10-30) |
| `data/features/<id>.csv.gz` | per photometer, every 5-min night bin: sky temperature, clear-sky baseline and residual, variability, flags (features.py) |
| `data/metar/` | airport pairs, automated METAR labels per photometer, the held-out scored reports (`scored.csv.gz`) |
| `data/goes`, `data/msg`, `data/viirs` | satellite mask boxes around each photometer, monthly (the extraction scripts) |
| `data/sat/` | satellite scans matched to photometer bins with the classifier's calls (sat_score.py) |
| `data/record/<id>.csv.gz` | **the cloud record**: the call for every night bin with its flags (cloud_record.py; column key in `results/cloud_record.txt`) |
| `data/mountain_pl`, `data/mountain`, `data/era5_ts`, `data/cloudtop`, `data/viirs_why_*` | ERA5 pressure-level series, station data and cloud-top samples for the mountain tests |
| `results/` | every number quoted in the paper, as csv and a plain-text summary per stage |
| `tables/` | the rows of the paper's tables (tables.py); `figures/` the figures (fig_*.py) |

## Pipeline

Stages 1-3 download and extract the raw inputs and are not needed to reproduce the paper (`python run_all.py --list`
prints every command). They need the raw STARS4ALL archive (fetch_tess_archive.py), an IEM METAR download
(fetch_metar.py), the Copernicus CDS and the ARCO-ERA5 copy on Google Cloud (fetch_era5_ts.py,
fetch_mountain_met.py, mountain_pl.py), NOAA's GOES archive on AWS (goes_extract.py), a EUMETSAT Data Store account
(msg_extract.py), a NASA Earthdata token (viirs_extract.py, viirs_local.py, viirs_v2_backfill.py,
viirs_added_sites.py), Open Topo Data (fetch_elevation.py) and NOAA ISD (fetch_mountain_met.py --isd); the audits of
the raw archive (audit_ir.py, audit_devices.py, audit_positions.py, audit_sequence.py, audit_placeholder.py) and the night bins
(build_nights.py) read the archive.

Stage 4, the analysis, in the order of `verify.py --list`:

| Target | Paper |
|---|---|
| features, qc_ir, qc_baseline, die_air, audit_country, baseline_explore | Sect. 2.1, 2.2, 3.1, 4.1, Table 1 |
| metar_labels, classify_metar, breakdown_metar | Sect. 2.4, 4.1, 4.3, Table 3, Fig. 2, Fig. 4 |
| sat_score, sat_score_dem, sat_sensitivity | Sect. 4.2, Table 4, Fig. 3 |
| sat_vs_metar | Sect. 4.2, Table 5 |
| cloud_record | the released record (Sect. 5.2) |
| nsb_selection, cloud_base | Sect. 4.5, 4.3 |
| mountain_baseline, mountain_network, mountain_below, cloud_top, viirs_why | Sect. 4.4 |
| tables, fig_roc, fig_breakdown, fig_example_night, fig_sat_sites | Tables 2-5, Figs. 1-4 |

The classifier and the satellite matching rules were fixed before the satellite scoring:
`data/frozen_thresholds.csv` (2026-10-30). What was added afterwards is listed in Sect. 3.5 of the paper and in the docstring
of each script. Scripts that read the raw archive or the raw granules, or call a web service, are not targets
(`terrain_boxes.py` and `cloud_top.py --ref` among them); `verify.py` names them and `run_all.py --raw` runs them.

## Data and licences

TESS-W readings: STARS4ALL network archive, ODbL 1.0; the derived data here are released under the same licence.
METAR: Iowa Environmental Mesonet (public domain). ERA5 and ERA5-Land: Copernicus Climate Change Service, CC BY 4.0
("The results contain modified Copernicus Climate Change Service information 2026"). GOES: NOAA open data. Meteosat:
EUMETSAT Data Store. VIIRS and ASTER: NASA. Code: MIT licence (`LICENSE`).

The STARS4ALL device histories that `audit_devices.py` cross-checks (name and MAC history lists) are not
redistributed; they are available from STARS4ALL on request. Their result, `results/audit_devices.csv`, is shipped,
and `audit_devices.py` runs without them (the category column is then left empty).
