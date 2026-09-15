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

## 9. Unknown pressure unit — largely resolved, still not confirmed (B)

`Pressure_Energy/*.xlsx` has no unit recorded. Values sit around 1.3 (low taps)
and 0.0–0.6 (high tap); volts and kPa were both plausible.

**New evidence (see `scripts/10_umf_analysis.py`).** The *total* bed pressure drop
is 6.497 and changes by only +2.2 % over 0.1–0.9 m/s, which is the signature of a
fully fluidised bed, so `ΔP_total = ρ_bulk · g · H_bed` applies with
`H_bed = 1.0 m`. If the readings are in kPa, the implied bed-averaged bulk density
is **662 kg/m³** — inside the 550–850 kg/m³ range expected for an FCC catalyst bed
(voidage ≈ 0.5–0.65 at a particle density of 1500–1800 kg/m³).

This is strong evidence that the unit is **kPa**, but it remains an inference from
a bulk property, so all absolute pressure statements stay flagged as provisional
until the laboratory confirms the transducer documentation.

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
| Item | Consequence if unaddressed | Handling here |
|---|---|---|
| `StandardScaler` fitted on the full dataset before the k-fold split | cross-validated scores are optimistically biased | `train_cv.py` fits the scaler inside each fold |
| `Sigmoid` output layer combined with un-standardised targets whose range is 600–900 °C | the model cannot represent the target; validation R² collapses | output activation and target standardisation are coupled in `scripts/06`, and the pair is reported |
| Absolute paths (`F:\XC_OFP&2D camera\...`, `D:\desktop\zhendong`, …) | the code cannot be run on another machine | every path resolves from `fbfml.paths` / `FBFML_DATA_ROOT` |
| Chinese column names in figures | missing-glyph warnings, unreadable labels | ASCII labels (`T1`…`T6`) via `data_pressure.channel_label` |
| 198 MB `matlab_R2025a_macOSAppleSilicon.dmg` inside the code folder | would be committed by accident | excluded by `.gitignore`; the file itself was left in place |

## 12. Reference values and what this analysis finds (A)

Quantities reported for the campaign, next to the value obtained here:

| Quantity | Reported | This analysis |
|---|---|---|
| 12-level wavelet band energies | stored in `Pressure_Energy/energy_*.xlsx` | reproduced exactly, 54/54 comparisons at machine precision (`docs/data_dictionary.md` §1.2) |
| transition velocity | 0.6–0.8 m/s | **0.55–0.65 m/s** from three estimators; 0.53–0.67 m/s across four observables (`docs/results.md` §2–3) |
| `U_mf` accuracy | ±0.05 m/s | **not identifiable** from this campaign: the total bed pressure drop is constant to +2.2 % over 0.1–0.9 m/s, so the bed is fluidised at every measured velocity and no fixed-bed branch was sampled; the profile-likelihood interval spans the whole 0.01–0.60 m/s grid (`scripts/10`) |
| regime classification | a BP-ANN was trained for it | **0.829 ± 0.015** for bubbling vs turbulent, 0.592 majority baseline, over ten splits; 0.790 ± 0.015 on an unseen radial position (`scripts/12`) |
| framework | BP-ANN | PyTorch multilayer perceptrons (`src/fbfml/model_wide_ann.py`) |

## 13. Error budget summary

| Source | Magnitude | Comment |
|---|---|---|
| Pressure: none needed | — | the transition-band estimators are insensitive to ±0.05 m/s |
| Probe: repeat-to-repeat drift | **85 mV** | vs a spatial range of **949 mV** → **noise/signal = 0.09** |
| Probe: residual ADC blips | ≈ 2 × 10⁻⁶ of samples in 5/434 records | negligible; quantified, not filtered |
| Probe: absolute calibration | **unknown, ≥ 100 %** | blocks absolute `εs` |
| Vibration: velocity assignment | ±1 step (±0.1 m/s) | inferred, validated on 2/9 folders |
| Vibration: sensor 6 missing | — | sensor-averaged quantities are 6-sensor, not 7-sensor |
| Surrogate: scaler leakage in the legacy run | not quantified | corrected in this repository's harness |
