# Legacy code map

The original project folder contained eleven near-duplicate notebooks, three toy
scripts and one MATLAB script. Nothing was deleted: everything is archived under
`legacy/` and mapped below to what replaced it.

## 1. Notebooks

| Legacy file | What it actually does | Status | Replaced by |
|---|---|---|---|
| `XC_regime.ipynb` | first attempt: MLP with a `Sigmoid` output on **un-standardised** targets → CV R² −16…−37 | broken | kept as evidence; the fix is in `train_cv.py` |
| `XC_regime_multi_activation.ipynb` | same data, mixed activations per layer | broken (same bug) | `model_wide_ann.ACTIVATIONS` |
| `XC_regime_with_split.ipynb` | adds an explicit 80/20 split | broken (same bug) | `train_cv.cross_validate` |
| `XC_regime_with_split_fixed.ipynb` | target standardisation introduced | superseded | same |
| `XC_regime_with_split_fixed1/2/3.ipynb` | scalar-encoder variants; the saved run reaches mean CV R² ≈ 0.60 | superseded | `analysis_vibration.py` + `train_cv.py` |
| `XC_regime_with_split_fixed3_optuna.ipynb` | adds Optuna hyper-parameter search | superseded | `optuna` remains in `requirements.txt` |
| `XC_regime_最好的.ipynb` ("the best") | residual blocks, BatchNorm, dropout, `SmoothL1Loss`, `PolynomialFeatures`, XGBoost baseline; single-fold R² 0.86 (MLP) and 0.89 (XGBoost on `Ug`, 0.31 on `Temp`) | **source of the recorded reference numbers** | `scripts/06`, `results/tables/surrogate_recorded_reference.csv` |
| `XC_AC preformances.ipynb` | ⭐ Wide-ANN surrogate `(r/R,H,DT,H0,Ug) → (εs,C/C0)`, 5-fold CV, SHAP, activation ablation. Mean CV R² = 0.9455 | **the strongest result in the project** | `scripts/06_train_surrogate.py` (architecture + harness) |
| `OFP-PVD.ipynb` | `.pvd` → xlsx + the voltage→`εs` calibration | lab pipeline, re-implemented | `data_ofp.py`, `scripts/04` |
| `计算2.ipynb` | batch version of the above, writes the mean into the filename | lab pipeline, re-implemented | `data_ofp.py`, `scripts/04` |
| `1.ipynb`, `1.py`, `8.py`, `j.py`, `main.py` | PyCharm/PyTorch smoke tests (`print(torch.__version__)`, a 4→8→2 toy net) | discarded | — |

## 2. MATLAB

| Legacy file | What it does | Replaced by |
|---|---|---|
| `20250609 LMF/code.m` (also `untitled.m`, `untitled.asv`) | walks `<root>/<condition>/*.mat`, subtracts 9.8 from channel 1, takes `abs()`, plots the three axes per condition to PNG, writes `实验结果汇总.xlsx` | `data_vibration.load_vibration` + `scripts/05` — the same quantities (`Average`, `StandardDeviation`) are computed in `features_stats.basic_stats`, but with the correct axis names (X/Y/Z, not "Y / X-Y / Y-X") |

## 3. What was **not** recoverable

| Missing artefact | Why it matters |
|---|---|
| the script that produced `Pressure_Energy/energy_*.xlsx` | the feature definition was lost → **reverse-engineered and verified** in `features_wavelet.py` |
| `工作簿1.xlsx` | the 40-sample labelled table's upstream source; no longer on the machine |
| `dataset/dataset_training.xlsx` | the 530-row surrogate dataset (group property, not redistributable) |
| the `εs` calibration constants that actually belong to this probe | blocks absolute solids-fraction values → `docs/data_quality.md` §4 |

## 4. Legacy notebooks and stale outputs

Several archived notebooks contain cell outputs that do not match their code
(e.g. `XC_AC preformances.ipynb` carries a per-fold metric of −16.16 that belongs
to an earlier regression experiment). They were executed on different machines
over several weeks; the outputs were never regenerated after the code changed.
This is one of the reasons the pipeline was rewritten as numbered scripts whose
outputs are produced by a single `python scripts/NN_*.py` call.
