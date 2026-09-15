# Datasheet: pressure, vibration and optical-probe measurements in a 3-D gas-solid fluidized bed across the bubbling-to-turbulent transition

## 1. Motivation

Measurements were taken in a 3-D gas-solid fluidized bed to characterise the
transition between the bubbling and the turbulent regime and to support
data-driven surrogates for the local solids fraction. The published results of
this project are summarised in the companion code repository (see the Zenodo
record's "related identifiers").

## 2. Composition

| Dataset | Instrument | Rate | Channels | Records | Size |
|---|---|---|---|---|---|
| pressure | see docs/data_dictionary.md | - | - | 18 | 27 MB |
| vibration | see docs/data_dictionary.md | - | - | 120 | 629 MB |
| ofp | see docs/data_dictionary.md | - | - | 1403 | 3182 MB |

Full column/field documentation: see `docs/data_dictionary.md` in the code
repository.

## 3. Collection process

* one continuous session per subsystem; the three acquisition systems are
  **not synchronised** with each other;
* gas velocity 0.1-0.9 m/s, static bed height 1.0 m, FCC catalyst particles;
* per-record lengths: pressure 120 s, vibration ~1000 s, optical probe 10.49 s.

## 4. Known defects (please read before using)

# Data quality report

Everything that is wrong, missing or uncertain about the raw data collected for
this project — found by running the code in this repository, not by memory.

Severity legend: **A** = blocks a scientific claim · **B** = limits a claim or
needs to be disclosed · **C** = cosmetic / housekeeping.

---

## 1. Damaged and incomplete vibration recordings (A)

`scripts/01_data_inventory.py` audits all 60 accelerometer files:

| Condition folder | Sensor | Problem |
|---|---|---|
| `0.9 20250609_114111` | 1 | `time_axis` empty |
| `20250609_112223` | 1 | `time_axis` empty |
| `20250609_112851` | 1 | `time_axis` empty |
| `20250609_113508` | 1 | `time_axis` empty |
| `20250609_104155` | all | aborted run: median record length **1.8 s** instead of ~1000 s |

Consequences

* The affected conditions have **5 sensors instead of 6**, which biases any
  sensor-averaged statistic. This is exactly why the naive
  "rank conditions by vibration intensity" approach fails on this campaign
  (see §3) and why the per-sensor check exists in `analysis_vibration.py`.
* The aborted run is excluded from the velocity sweep by the documented
  60 s minimum-duration rule.

## 2. A sensor is missing from the archive (B)

Seven accelerometers were installed (poster and deck) but files exist only for
sensors `1, 2, 3, 4, 5, 7` — **sensor 6 has no recording in any condition**.
Any 7-sensor claim cannot be reproduced.

## 3. Eight of ten vibration folders have no gas-velocity label (A)

Only `0.1 …` and `0.9 …` carry a velocity. **The velocity used in
`results/tables/vibration_features.csv` is inferred**, by this rule:

1. order the folders by the acquisition timestamp embedded in their names;
2. drop runs shorter than 60 s;
3. assign `0.1, 0.2, …, 0.9 m/s` to the remaining nine, in order.

Validation: the rule reproduces **both** labelled folders exactly
(`0.1 → 0.1`, `0.9 → 0.9`), and the resulting ordering is physically consistent —
the per-sensor Spearman correlation between wall-vibration intensity and the
assigned velocity has a **median of ρ = 0.78**, with 8 of 18 sensor/axis channels
above 0.8.

Cross-check that *fails* and is therefore reported: ordering the folders by the
mean vibration intensity across sensors (the obvious approach) puts the `0.9 m/s`
folder in 6th place, because it is missing sensor 1. The failure is itself
evidence that the timestamp rule is the more reliable key — and that the missing
sensor matters.

Every derived table keeps both `velocity_assigned` and `velocity_inferred`, so an
inferred velocity can never be mistaken for a measured one.

## 4. The absolute solids-fraction calibration is not usable (A)

Applying the two constant sets recovered from the legacy scripts gives:

| Diagnostic | Value | Verdict |
|---|---|---|
| `εs` range over 86 conditions | 0.288 – 0.507 | a bubbling bed must span roughly 0.02 – 0.6 |
| span | 0.218 | too narrow |
| radial gradient direction | core < wall in all comparable conditions | ✓ correct sign |
| **usable?** | **no** | `calibration_usable = false` |

The constants belong to a different probe/gain setting. A bubbling bed with a
time-averaged `εs ≈ 0.4` everywhere, independently of `r/R` and of `U_g`, is not
physical.

**Handling.** The pipeline reports the calibration-free **probe voltage** as the
primary measurement (monotonically related to `εs`) and flags the calibrated
value automatically. `analysis_concentration.calibration_plausibility()` will
start emitting `eps_s` figures again as soon as a correct `(a, b)` pair is
supplied — the code is not hard-wired to the wrong constants.

## 5. Incomplete probe coverage (B)

108 condition folders exist (2 heights × 6 radial positions × 9 velocities), but:

| repeats per condition | number of conditions |
|---|---|
| 0 (empty folder) | 22 |
| 1–4 | 8 |
| 5–6 | 74 |
| 7–10 | 4 |

**434 usable records out of a nominal 540.** The empty folders are concentrated at
`r/R = 0.95` (both bed heights) and at `r/R = 0.837`, i.e. the near-wall region is
the least sampled — unfortunate, because that is where the core–annulus contrast
is largest.

## 6. Signal artefacts in the probe records (C)

Raw probe channels occasionally hit the negative rail (`counts_min` down to
−5000) although the instrument range is 0–5 V. The audit quantifies this exactly:

* **5 of 434 records** contain at least one negative sample;
* in those records the affected fraction is **≈ 2 × 10⁻⁶** (a single sample out of
  524 288), and `frac_rails` (|counts| ≥ 4990) never exceeds 3.8 × 10⁻⁶.

This is a benign, sub-ppm blip rather than a saturation problem, so no filtering
is applied. Per-record values are nevertheless exposed in
`results/tables/ofp_records.csv` as `counts_min`, `counts_max`, `frac_below_zero`
and `frac_rails` so that a user who wants a stricter filter can apply one.

## 7. The three acquisition systems are not synchronised (B)

| System | Rate | Record length |
|---|---|---|
| pressure | 500 Hz | 120 s |
| vibration | 1666 Hz | ~1000 s |
| optical probe | 50 kHz | 10.49 s |

Different instruments, different clocks, no shared trigger. No simultaneous
pressure–vibration–probe analysis is possible from this archive. Any statement
coupling two of them is a comparison of *conditions*, not of *instants*.

## 8. Sampling-rate inconsistency (B)

The materials quote 500 Hz for the campaign; the vibration files store
`data_rate_hz = 1666` (a non-integer DAQ default). The recorded value is used
throughout, and every derived frequency (dominant frequency, PSD axis) inherits a
possible scale error of an unknown factor if 1666 is wrong.

## 9. Unknown pressure unit (B)

`Pressure_Energy/*.xlsx` has no unit. Values sit around 1.3 (low taps) and 0.0–0.6
(high tap) — volts and kPa are both plausible. All pressure results are therefore
presented as **relative** quantities; no absolute pressure drop or bed weight is
claimed.

## 10. Missing upstream datasets (A for modelling)

| File | What it held | Impact |
|---|---|---|
| `工作簿1.xlsx` | the 40-sample labelled table (10 velocities × 4 temperatures), 26 wavelet features | the operating-condition models cannot be re-run |
| `dataset/dataset_training.xlsx` | the 530-row surrogate dataset (r/R, H, DT, H0, Ug → εs, C/C0) | the Wide-ANN results cannot be re-run |

Neither file is present anywhere on the machine. The 40-sample table survives as
`training & validation/{train,validation}_set.xlsx` (features + targets), but
**its raw-data trail does not**: no recording in this archive carries a
temperature axis. The temperature dimension is documented as an open question.

## 11. Legacy code-quality issues, and how they are handled (B/C)

| Issue | Consequence | Resolution here |
|---|---|---|
| `StandardScaler` fitted on the full dataset **before** the k-fold split | every cross-validated score in the legacy notebooks is optimistically biased | `train_cv.py` fits inside the fold; the bias is documented in `README.md` §6 |
| `Sigmoid` output layer combined with un-standardised targets (range 600–900 °C) | validation R² of **−27** across all folds | kept as evidence of the debugging path; the fixed version standardises the targets |
| Absolute paths (`/Users/luchi/...`, `C:\Users\cbi20\...`, `F:\XC_OFP&2D camera\...`, `D:\desktop\zhendong`) | code unrunnable anywhere else | all paths via `fbfml.paths`, overridable with `FBFML_DATA_ROOT` |
| 11 near-duplicate notebooks (`fixed`, `fixed1`, `fixed2`, `fixed3`, …) with stale cell outputs | impossible to tell which result is real | archived under `legacy/` with a mapping table; the pipeline was rewritten as numbered scripts |
| No version control, no README, no requirements | nothing reproducible, nothing reviewable | this repository |
| 198 MB `matlab_R2025a_macOSAppleSilicon.dmg` inside the code folder | accidental commits | excluded by `.gitignore`; the file was left where it was, not deleted |
| Chinese column names in figures | missing-glyph warnings, unreadable labels | ASCII labels (`T1`…`T6`) via `data_pressure.channel_label` |

## 12. Claims in the project's own materials that the data does not support (A)

| Claim | Finding |
|---|---|
| "TensorFlow" neural network | no TensorFlow anywhere; the code is PyTorch |
| ">95 % classification accuracy" in the draft CV text | no classifier, no accuracy metric, no confusion matrix exists in the project folder. The only ~0.95 figure is a **regression R²** on a different (non-redistributable) dataset |
| "U_mf within ±0.05 m/s" | stated on the poster, but no code, table or figure in the folder computes `U_mf` or its error |
| "transition velocity 0.6–0.8 m/s" | **independently reproduced here** as **0.55–0.65 m/s** (three estimators). The bands overlap; the re-analysis is slightly lower |
| "12-level wavelet energy" features | **fully reproduced**, machine-exact (§ `docs/data_dictionary.md` §1.2) |



## 5. Uses

Intended: signal-processing benchmarks, regime-classification studies,
surrogate-model training, methods for quality control of experimental archives.

Not intended: absolute solids-fraction values without a probe-specific
calibration; any analysis that assumes the three subsystems are time-aligned;
any claim about the 7th accelerometer (no data exists for it).

## 6. Licence and attribution

Raw experimental data: CC BY 4.0 (pending written confirmation) (to be confirmed by the Fluidization
Laboratory). Code: MIT. Please cite both the dataset DOI and the code
repository. **The data were collected by the Fluidization Laboratory,
Beacons of Excellence Research and Innovation Institute, University of
Nottingham Ningbo China; the release is prepared by Chi Lu.**

## 7. Checksums

See `MANIFEST.csv` (SHA-256 per file).
