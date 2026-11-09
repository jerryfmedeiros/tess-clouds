## Cloud-base information: design fixed before the run (2026-10-31)
A quick look (single-layer BKN/OVC METAR, 139k reports, simple least squares, airport CV: median error 0.54 km
against 0.91 km for the typical height; rank correlation 0.45) showed height information. The design below is
fixed before the one confirmatory run; the quick look is not reported as a result.

- Reports: automated METAR at the 58 scored photometers, cloudy (BKN or OVC), exactly one layer reported (lowest
  base = highest base), snow-free (ERA5-Land snow depth <= 0.01 m), not at the firmware floor, classifier inputs
  defined. Photometer bins: the 30 minutes before the report (as classify_metar.py), medians.
- Target: ceilometer base above the airport, as log10(km) (bases below 30 m set to 30 m), and as three classes:
  low < 2000 ft, mid 2000-6500 ft, high >= 6500 ft.
- Inputs, fixed: air minus sky (t2m - sky); cloud warming fraction R = (sky - clear baseline) / (t2m - clear
  baseline), clipped to [-0.5, 1.5], with the out-of-sample Brunt baseline of features.py; dew-point depression
  (t2m - d2m); t2m; overcast indicator (OVC vs BKN).
- Model A: least squares on log10 height with those inputs and an intercept. Model B (non-linear check): median
  log height in training cells of R deciles x dew-point-depression quintiles x OVC/BKN (training cells with
  fewer than 50 reports fall back to the R decile).
- Validation: 5-fold cross-validation grouped by airport, folds as in classify_metar.py (seed 20261030).
  Comparison: the training-fold median height. Scores: median absolute error (km), share within a factor 2,
  Spearman rank correlation, 3-class balanced accuracy and the confusion table; 95% night-block bootstrap
  within photometers (B 1000) for median error and balanced accuracy.
- Reported as is. Also the subset of non-US airports (US ASOS ceilometers report bases only to 12,000 ft).
  The claim follows the held-out numbers.
