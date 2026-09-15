# Experimental setup

> **Source policy.** Everything in this document is either (a) taken verbatim
> from the project's own poster and 34-slide progress deck, (b) taken from the
> raw data itself, or (c) explicitly marked *inferred/unconfirmed*. Where a
> documented statement conflicts with the raw data, both are reported.

## 1. Apparatus

| Item | Value | Source |
|---|---|---|
| Bed type | 3-D column, with a 2-D bed and a small bed used as comparative setups | poster |
| Bed dimensions | 200 cm (height) × 30 cm (width) × 1.9 cm (depth) | poster — *the dimension/label pairing looks inconsistent (1.9 cm depth is a quasi-2-D column); needs confirmation* |
| Bed material | FCC catalyst, ~110 µm | inferred from the folder name `FCC110` in the legacy probe pipeline |
| Static bed height | H₀ = 1 m | poster |
| Superficial gas velocity | 0.1 – 0.9 m/s | poster **and** the folder names of all three datasets |
| Measurement heights (pressure) | 10–30, 30–50, 50–70, 70–90, 90–110, 110–130 cm | poster; matches the six pressure columns |
| Number of accelerometers | 7 (sensors 1, 5, 7 vertically aligned, 2–4 and 6 distributed laterally) | poster + deck |
| Probe positions | bed heights 60 and 100 cm; radial `r/R` = 0, 0.316, 0.548, 0.707, 0.837, 0.95 | folder tree + deck |

## 2. Instrumentation and sampled data

| Channel | Instrument | Rate | Record length | Channels | Stored as |
|---|---|---|---|---|---|
| Differential pressure | 6 transducers | **500 Hz** | 120 s | 6 | `Pressure_Energy/<U>.xlsx` |
| Wall vibration | 7 tri-axial accelerometers (6 present) | **1666 Hz** | ~1000 s (16.7 min) | 3 per sensor | `20250609 LMF/<condition>/<n>.mat` |
| Local solids fraction | dual-tip optical fibre probe (PV6D family) | **50 kHz** | 524 288 samples ≈ 10.49 s | 2 | `20250609 LC/<H>/<r/R>/<U>/*.pvd` |

Notes

* The poster quotes "sampling frequency: 500 Hz, sampling time: 2 min" — that
  describes the **pressure** channels.  The vibration and probe channels run at
  their own rates; the three systems are **not synchronised** with each other.
* Vibration records store `data_rate_hz = 1666`, a non-integer value consistent
  with a DAQ default rather than a deliberate choice.  It is used as recorded and
  flagged in `docs/data_quality.md`.
* Accelerometer range is `full_scale_g = 2`; `processed_data` is already scaled to
  g, `raw_data` holds the int16 ADC counts.

## 3. Campaign structure

Three campaigns are visible in the folder timestamps.

| Date | Dataset | Contents |
|---|---|---|
| 2025-06-05 / 06-06 | `20250609 LC` | optical-probe records for the 60 cm and 100 cm beds |
| 2025-06-09 | `20250609 LMF` | one vibration session containing the 0.1–0.9 m/s sweep |
| 2025-07-01 | `Pressure_Energy` | pressure logs for 0.1–0.9 m/s |

The vibration session ran from 10:41:55 to 11:41:11 local time and produced ten
condition folders: nine complete runs plus one 1.8 s aborted run.  Ordering the
folders by their timestamps reproduces the labels of the only two folders that
carry one, giving a clean nine-point velocity sweep (`0.1 … 0.9 m/s`) — see
`docs/data_quality.md` §3.

## 4. Processing reported by the project itself

* Signals were analysed both **vertically** (height-dependence) and
  **horizontally** (cross-bed).
* A **db4 discrete wavelet transform with 12 levels** was used to extract
  time–frequency features; the per-level energy and its proportion, plus the mean
  and standard deviation of the signal, formed the model inputs (28 inputs on the
  poster).
* A **BP-ANN** ("back-propagation artificial neural network") was trained with an
  80/20 train–test split to predict pressure drop and a flow-regime class.
* The poster reports the transition velocity as **0.6–0.8 m/s** and states that
  U_mf is identified to within **±0.05 m/s**.

Two corrections that this repository documents rather than repeats:

1. The materials state the network was built in **TensorFlow**; the actual code is
   **PyTorch** (plus XGBoost/Optuna/SHAP in the exploration notebooks). No
   TensorFlow code exists anywhere in the project folder.
2. No classification model, confusion matrix or accuracy score exists in the
   project folder. The numerical results that do exist are **regression** scores.
   See `docs/CONTRIBUTIONS.md`.

## 5. Estimators used in this repository

Everything below is recomputed from the raw signals; nothing is copied from the
poster.

| Quantity | Definition used here | Module |
|---|---|---|
| Transition band | span of three instrument-only estimators: steepest rise of the topmost tap mean, steepest rise of the top/bottom fluctuation ratio, and the velocity at which the lowest tap drops 2 % below its plateau | `analysis_regime.py` |
| Band energies | `Level_i = Σ cD_i²`, db4, 12 levels, `mode="symmetric"`, DC removed | `features_wavelet.py` |
| Radial gradient | least-squares slope of the probe signal vs `r/R` per condition, plus a sign test over conditions | `analysis_concentration.py` |
| Error budget | repeat-to-repeat spread divided by the spatial range of the probe signal | `analysis_concentration.py` |
| Velocity assignment | order condition folders by acquisition timestamp, drop incomplete runs, assign `0.1 … 0.9 m/s` | `data_vibration.py` |
