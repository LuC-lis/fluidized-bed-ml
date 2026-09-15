# Contributions: what is real, what is provable, what is not

Written to be used directly when describing this project (CV, statement of
purpose, interview, portfolio). Every entry states **where the evidence lives** and
**how strong it is**.

Evidence scale:

* **Verified** — reproduced by code in this repository, from raw data.
* **Recorded** — the number exists in the archived legacy notebooks, with saved
  outputs, but cannot be regenerated here because the underlying dataset is not
  redistributable.
* **Stated** — appears only in the project's poster/deck or in a draft CV, with no
  code, table or figure behind it.
* **Retracted** — appears in the materials and is contradicted by the evidence.

---

## 1. Verified contributions (the backbone of any claim)

### C1 · Recovered a lost feature definition, and proved the recovery is exact
*Evidence: `src/fbfml/features_wavelet.py`, `scripts/03_wavelet_validation.py`,
`tests/test_wavelet_matches_vendor.py`*

The script that produced the group's wavelet feature tables was gone; the
model inputs therefore had no definition. I recovered it
(**db4, 12 levels, `mode="symmetric"`, DC removed, `Level_i = Σ cD_i²`,
`TotalEnergy` = sum of the 12 detail bands, approximation band excluded**) and
verified it against the group's own stored tables:

> **54 / 54 channel–velocity comparisons match; worst relative error 6 × 10⁻¹².**

This converts an undocumented artefact into an executable specification with a
regression test. It is the single most defensible technical contribution in the
project.

### C2 · Turned 434 raw probe records into a documented solids-fraction field
*Evidence: `src/fbfml/data_ofp.py`, `src/fbfml/analysis_concentration.py`,
`scripts/04`, `results/tables/ofp_conditions.csv`*

* reverse-engineered the vendor `.pvd` container (GBK header + two-column sample
  block) and its metadata;
* built a reusable reader for **434 records / 86 conditions / 108 folders**,
  including the folder→coordinate parsing (bed height, `r/R`, `U_g`);
* produced the time-averaged field with **repeat-to-repeat error bars**;
* ran a **radial-gradient test** per condition: **14 / 17 conditions show the
  physically expected core–annulus sign, sign test p = 0.006**;
* computed an explicit **error budget**: repeat spread 85 mV against a spatial
  range of 949 mV → **noise/signal = 0.09**;
* wrote a **plausibility diagnostic that rejects the available calibration** and
  prevented an unphysical `εs ≈ 0.4` field from being reported as a result.

### C3 · Independently re-derived the flow-regime transition from the pressure taps
*Evidence: `src/fbfml/analysis_regime.py`, `scripts/02`,
`results/tables/pressure_regime_transition.json`*

Three instrument-only estimators (steepest rise of the topmost tap, steepest rise
of the top/bottom fluctuation ratio, plateau-decline velocity of the lowest tap)
give **0.55 – 0.65 m/s**, overlapping the 0.6–0.8 m/s quoted in the project
poster. I also documented a negative result that the poster omitted: the classic
"peak of the pressure-fluctuation intensity" criterion is **not usable** in this
campaign, because the summed standard deviation is still rising at the highest
velocity tested.

### C4 · Recovered the gas velocity of eight unlabelled folders — with validation
*Evidence: `src/fbfml/data_vibration.py::velocity_assignments`,
`tests/test_data_readers.py::test_timestamp_rule_matches_the_labelled_folders`*

Only 2 of 10 condition folders carried a velocity label. I recovered a
nine-point `0.1 – 0.9 m/s` sweep from the acquisition timestamps and **validated
the rule against both labelled folders**. The obvious alternative (rank by
vibration intensity) *fails* on this campaign because one condition is missing a
sensor; that failure is documented rather than hidden, and the physical check is
done **per sensor** (median Spearman ρ = 0.78; 8/18 channels > 0.8).

### C5 · Identified which vibration features actually carry velocity information
*Evidence: `results/tables/vibration_monotonicity.csv`*

Ranked by Spearman ρ against the assigned velocity: `Level9_ratio` 0.87,
dominant frequency 0.83, `Level6_ratio` 0.82, `std` 0.78 — i.e. the finest
wavelet scales and the dominant frequency, not the broadband energy.

### C6 · A complete data-quality audit of the campaign
*Evidence: `docs/data_quality.md`, `scripts/01`, `results/tables/inventory_*.csv`*

Four truncated vibration recordings, one aborted run, a permanently missing
sensor, 22 empty probe conditions (30 with fewer than five repeats, concentrated
at the near-wall position), signal dropouts at the negative rail, an unknown
pressure unit, three unsynchronised acquisition systems, and a sampling-rate
inconsistency — all quantified, all reported.

### C7 · A modelling harness without the leakage the original code had
*Evidence: `src/fbfml/train_cv.py`, `tests/test_models_and_cv.py`,
`src/fbfml/model_wide_ann.py`*

Seeded end-to-end; scaler fitted **inside** each fold; early stopping on the
validation split; per-fold and aggregate metrics; a test that proves a
pure-noise target cannot score. The legacy notebooks fitted their scaler on the
full dataset before splitting, so **every cross-validated score in the archived
notebooks is optimistically biased** — this is stated wherever those numbers are
reused.

---

## 2. Recorded contributions (real work, but the data is not redistributable)
### C8 · Wide-ANN surrogate with SHAP and an activation ablation
*Evidence: `results/tables/surrogate_recorded_reference.csv`,
`legacy/notebooks/XC_AC preformances.ipynb`, reimplemented in `scripts/06`*

A deep-and-wide feed-forward network mapping
`(r/R, H, DT, H0, U_g) → (εs, C/C0)`, trained with 5-fold cross-validation and
gradient-based SHAP explanations. Recorded results:

| Activation (hidden layers) | mean 5-fold CV R² |
|---|---|
| LeakyReLU | **0.9455** |
| ELU | 0.9335 |
| GELU | 0.9303 |
| *no* output sigmoid | 0.9074 |
| ReLU | 0.8136 |
| Tanh | 0.6705 |

Mean CV MSE 0.00094. The finding worth keeping: **the activation function moves
the score far more than the depth or width of the network** — a 0.28 spread on
identical splits.

### C9 · Operating-condition models from vibration signatures
*Evidence: `results/tables/surrogate_recorded_reference.csv`,
`legacy/notebooks/XC_regime_最好的.ipynb`*

26 wavelet features → `(U_g, Temp)` on 40 labelled samples: XGBoost hold-out
**R² = 0.89 for gas velocity**, 0.31 for temperature; a tuned MLP reaches
~0.60 mean CV R². The honest reading is a **negative result with a clean cause**
(40 samples is far too few), plus the debugging path from a −27 R² failure
(sigmoid output + unstandardised targets) to a working configuration.

---

### C10 · Solved the project's headline claim properly (the strongest *new* result)
*Evidence: `src/fbfml/features_probe.py`, `scripts/07_regime_classification.py`,
`results/tables/probe_classification_results.csv`,
`results/figures/probe_confusion_matrices.png`*

The poster claims the network "accurately distinguishes between different flow
regimes", but no classifier, accuracy or confusion matrix ever existed. I turned
that claim into a measured one: **36 features extracted from 10.5 s of
optical-probe signal**, 434 records, three model families, two evaluation
protocols, and a majority-class baseline reported next to every score.

| Task | Best accuracy | Majority baseline | Macro-F1 |
|---|---|---|---|
| 9-class gas velocity | 0.29 | 0.13 | 0.30 | (0.58-0.64 within one velocity step) |
| 3-class regime (bubbling / transition / turbulent) | 0.70 | 0.52 | 0.50 |
| **2-class bubbling vs turbulent** | **0.82** | 0.59 | **0.81** |

The binary result **survives the harsher protocol** — training on five radial
positions and testing on a sixth that the model has never seen (accuracy 0.80,
macro-F1 0.79) — so it is not an artefact of repeating the same measurement.

The two negative results are reported with equal prominence: the *transition*
regime is not separable as a discrete class (the classifier almost never predicts
it, macro-F1 0.50), which is what one expects for a continuum; and the exact gas
velocity is not recoverable from a local probe signal, although the errors are
strongly ordinal (58-64 % of predictions land within one velocity step).

---

## 3. What is **not** attributable to this work

| Statement | Status |
|---|---|
| "TensorFlow" | **Retracted.** No TensorFlow code exists; the framework is PyTorch (+ XGBoost, Optuna, SHAP). |
| ">95 % classification accuracy" | **Retracted.** No classifier, no accuracy metric, no confusion matrix exists in the project. The only ≈0.95 number is a **regression R² on a different dataset**. |
| "±0.05 m/s accuracy for U_mf" | **Stated only.** No code, table or figure computes `U_mf` or its error. Not usable without new work (the pressure data would in fact support it — see the plan). |
| Authorship of, or acknowledgement in, Chen *et al.* (2026), *Eng. Appl. Comput. Fluid Mech.* 20(1):2670348 | **None.** The author of this repository is not an author and is not named in the acknowledgements. The paper is the group's independent descendant of this method family and must be cited as context only. |
| The `.pvd`/OFP processing pipeline (`OFP-PVD.ipynb`, `计算2.ipynb`) authored from scratch | **Shared.** Those scripts were written on the laboratory's Windows machine (paths like `C:\Users\cbi20\...`) and are the group's existing pipeline; what this repository adds is the reusable, documented re-implementation and the calibration audit. |
| The 40-sample and 530-row datasets | **Group property.** Used, but not owned or redistributable. |

---

## 4. Skills this project actually demonstrates

* **Signal processing** — discrete wavelet decomposition (db4, 12 levels), PSD,
  band-energy ratios, fluctuation statistics, calibration of an optical probe.
* **Software engineering** — packaging a research folder into a layered library,
  removing hard-coded paths, writing regression tests, streaming ~4.6 GB of
  recordings without exhausting memory, caching derived tables.
* **Statistics and rigour** — leakage-free cross-validation, sign tests,
  error budgets, explicit separation of measured vs inferred quantities.
* **Scientific judgement** — reproducing a peer's numbers instead of trusting
  them; refusing to publish an unphysical calibration; reporting negative results
  (no interior fluctuation peak; 40 samples is too few).
* **Documentation** — data dictionary, data-quality report, provenance table,
  legacy mapping.

The narrative that these facts support is *not* "I built a model that predicts
fluidized beds". It is: **"I took an undocumented, partially corrupted
experimental dataset from someone else's project, recovered the lost pieces,
reproduced the claims that held up, disproved the ones that did not, and left
behind a repository that reruns end to end."** For a research-master application
that is a stronger and much rarer signal.
