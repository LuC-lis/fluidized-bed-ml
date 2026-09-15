# File map

The analysis files that came with the project are archived under `legacy/` and are
mapped below to the modules and scripts that carry the same work now. Nothing was
deleted.

## Notebooks and scripts → current modules

| Archived file | What it does | Where it lives now |
|---|---|---|
| `XC_AC preformances.ipynb` | wide-ANN surrogate `(r/R, H, DT, H0, U_g) → (εs, C/C0)`, 5-fold CV, SHAP, activation study | `src/fbfml/model_wide_ann.py`, `src/fbfml/train_cv.py`, `scripts/06` |
| `XC_regime.ipynb` | first operating-condition model (vibration features → `U_g`, `Temp`) | `scripts/05`, `scripts/12` |
| `XC_regime_multi_activation.ipynb` | same task, mixed activations per layer | `src/fbfml/model_wide_ann.py` (`ACTIVATIONS`) |
| `XC_regime_with_split.ipynb` | adds an explicit 80/20 split | `src/fbfml/train_cv.py` |
| `XC_regime_with_split_fixed.ipynb` | target standardisation | `src/fbfml/train_cv.py` |
| `XC_regime_with_split_fixed1/2/3.ipynb` | network-width variants | `src/fbfml/model_wide_ann.py` |
| `XC_regime_with_split_fixed3_optuna.ipynb` | Optuna hyper-parameter search | `optuna` in `requirements.txt`, `scripts/06 --activations` |
| `XC_regime_最好的.ipynb` | residual blocks, BatchNorm, dropout, `SmoothL1Loss`, polynomial features, XGBoost baseline | `src/fbfml/model_wide_ann.py` (`ResidualMLP`), `results/tables/surrogate_recorded_reference.csv` |
| `OFP-PVD.ipynb` | `.pvd` → spreadsheet, voltage→`εs` calibration | `src/fbfml/data_ofp.py`, `scripts/04` |
| `OFP_results_batch.ipynb` (was `计算2.ipynb`) | batch version of the above | `src/fbfml/data_ofp.py`, `scripts/04` |
| `vibration_postprocess.m` (was `20250609 LMF/code.m`) | per-condition channel statistics and plots | `src/fbfml/data_vibration.py`, `src/fbfml/features_stats.py`, `scripts/05` |
| `1.ipynb`, `1.py`, `8.py`, `j.py`, `main.py` | PyTorch / PyCharm smoke tests | — |

## Quantities that were re-derived here

| Quantity | Source | Module |
|---|---|---|
| band energies | stored `Pressure_Energy/energy_*.xlsx` tables | `src/fbfml/features_wavelet.py` (verified, `scripts/03`) |
| gas velocity of the vibration sessions | folder acquisition timestamps, validated against the labelled folders | `src/fbfml/data_vibration.py::velocity_assignments` |
| probe coordinates (`H`, `r/R`, `U_g`) | `.pvd` folder tree | `src/fbfml/data_ofp.py::parse_location` |
| probe voltage → solids fraction | two constant pairs recoverable from the analysis scripts | `src/fbfml/data_ofp.py` (calibration; see `docs/data_quality.md` §4) |

## Datasets used by the modelling notebooks

`工作簿1.xlsx` (the table behind the 40-sample labelled set) and
`dataset/dataset_training.xlsx` (the 530-row surrogate dataset) are not present on
this machine. The 40-sample set survives as
`training & validation/{train,validation}_set.xlsx`; the surrogate results are
preserved with provenance in `results/tables/surrogate_recorded_reference.csv`.
