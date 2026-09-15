# Data dictionary

Field-by-field description of the raw files and of every derived table, so the
data can be read without reading the code.

---

## 1. `Pressure_Energy/` — differential pressure

### 1.1 Raw logs `0.1.xlsx … 0.9.xlsx`

One file per superficial gas velocity (the file name *is* `U_g` in m/s).

| Column | Meaning | Unit |
|---|---|---|
| `时间` | time from the start of the record | s |
| `通道9` … `通道14` | differential-pressure transducers, ordered by mounting height | **unknown** |

| Property | Value |
|---|---|
| rows | 60 000 (`dt` = 0.002 s) |
| sampling rate | 500 Hz (verified from the time column) |
| duration | 120 s |
| channels | 6 |

Transducer heights (cm above the distributor):

| Column | label used in this repo | height band |
|---|---|---|
| `通道9` | `T1` | 10–30 |
| `通道10` | `T2` | 30–50 |
| `通道11` | `T3` | 50–70 |
| `通道12` | `T4` | 70–90 |
| `通道13` | `T5` | 90–110 |
| `通道14` | `T6` | 110–130 |

> **Open question.** The physical unit of the readings is not recorded anywhere.
> Absolute bed-inventory statements are therefore avoided; the analysis uses
> relative changes only.

### 1.2 Stored feature tables `energy_0.1.xlsx … energy_0.9.xlsx`

The group's own derived features, one row per pressure channel (6 rows):

| Column | Meaning |
|---|---|
| `Level1` … `Level12` | energy of the db4 detail band at that level |
| `TotalEnergy` | `Σ Level1..Level12` (the level-12 **approximation band is excluded**) |

These tables are what `scripts/03_wavelet_validation.py` reproduces exactly.

---

## 2. `20250609 LMF/` — wall vibration

```
20250609 LMF/
├── 0.1 20250609_104345/          # folder name may carry the velocity, often not
│   ├── 1.mat 1.bin               # accelerometer #1 (+ its raw binary blob)
│   ├── 2.mat 2.bin
│   ├── 3.mat 3.bin
│   ├── 4.mat 4.bin
│   ├── 5.mat 5.bin
│   └── 7.mat 7.bin               # sensor #6 has no file
└── …
```

### 2.1 `*.mat` keys

| Key | Shape / type | Meaning |
|---|---|---|
| `processed_data` | `(N, 3) float64` | acceleration in **g**, columns = X, Y, Z |
| `raw_data` | `(N, 3) int16` | ADC counts |
| `time_axis` | `(N, 1) float64` | absolute epoch seconds (UTC) |
| `start_time` | scalar float | epoch seconds of the first sample |
| `data_rate_hz` | scalar int | 1666 |
| `full_scale_g` | scalar int | 2 |
| `dev_id` | scalar int | device id |

`N ≈ 1.65 – 1.73 million`, i.e. ≈ 1000 s per record.  All six files of a folder
share the same `start_time` → they are **simultaneous recordings**, one per
sensor, not repetitions.

> The legacy MATLAB script labelled the columns "Y axis / X-Y axis / Y-X axis".
> They are the **X, Y and Z** axes; this repository uses the correct names.

### 2.2 Derived tables

| Table | One row per | Key columns |
|---|---|---|
| `vibration_features.csv` | condition × sensor × axis | `Level1..12`, `Level1_ratio..12_ratio`, `stat_*`, `dominant_freq_hz`, `velocity_assigned`, `velocity_inferred`, `condition_rank` |
| `vibration_intensity_by_condition.csv` | condition | mean wall-vibration std vs assigned velocity |
| `vibration_band_ratios.csv` | condition | median band-energy ratio per level |
| `vibration_monotonicity.csv` | feature | Spearman ρ against the assigned velocity |
| `vibration_per_sensor_check.csv` | sensor × axis | Spearman ρ of intensity vs velocity |

---

## 3. `20250609 LC/` — optical fibre probe

```
20250609 LC/
└── <bed height cm>/<r/R>/<U_g m/s>/
    ├── <timestamp>,1A,1B.pvd     # the samples
    ├── <timestamp>-st.txt        # per-channel statistics
    ├── <timestamp>-st.xls        # same, as a spreadsheet
    └── <timestamp>.ini           # full instrument configuration
```

### 3.1 `.pvd` format

Plain text, GBK-encoded header, then whitespace-separated integer samples, two
columns per row (probe channels `1A`, `1B`).

Header fields (Chinese key → meaning):

| Key | Meaning | Typical value |
|---|---|---|
| `文件名` | record id | `20250606_110010` |
| `采样时间` | acquisition time | `2025-06-06 11:00:10` |
| `通道数` | number of channels | 2 |
| `数据量` | number of samples | 524288 |
| `采样频率(kHz)` | sampling rate | 50 |
| `滤波频率(Hz)` | anti-alias filter | 2 kHz |
| `测点间距` | probe tip spacing | 1.7 mm |
| `采样通道` | channel ids | `1A,1B` |
| `传感器编号` | probe id | `P1` |
| `传感器量程(V)` | input range | 5 |
| `最大值` / `最小值` / `平均值` / `标准差` | per-channel statistics | counts (mV) |

Per record: 524 288 samples at 50 kHz → **10.49 s**, two near-identical channels
(Pearson r = 0.999; the two tips are 1.7 mm apart).

### 3.2 Voltage → solids fraction

The instrument reports ADC counts in **millivolts**.  The legacy pipeline used an
exponential calibration

```
eps_s = a * exp(b * V)        with V in volts
```

Two constant sets were recovered:

| `(a, b)` | Where it appears in the legacy code |
|---|---|
| `(0.0164, 0.7759)` | `OFP-PVD.ipynb`, `计算2.ipynb` (h102 / r0.95) |
| `(0.01511, 0.79292)` | `计算2.ipynb` (h40 / r0) |

**Neither is applicable to this campaign** — applying them yields a flat field in
0.29–0.51 with no radial structure, which is physically impossible for a
bubbling bed.  The pipeline therefore reports the **probe voltage** (mV) as the
primary measurement and automatically flags the calibrated value as unusable
(`analysis_concentration.calibration_plausibility`).  Both constant sets remain
available so that the correct probe-specific calibration can simply be plugged in.

### 3.3 Derived tables

| Table | One row per | Key columns |
|---|---|---|
| `ofp_records.csv` | raw record | `bed_height_cm`, `radial_position`, `velocity`, `counts_mean`, `eps_mean`, `eps_std`, `duration_s` |
| `ofp_conditions.csv` | height × r/R × U_g | `n_records`, `counts_mean`, `eps_mean`, repeat-spread and within-record spread |
| `ofp_core_annulus.csv` | height × U_g | core (`r/R=0`) over wall (`r/R=0.95`) ratio |
| `ofp_radial_gradient_test.csv` | height × U_g | slope vs `r/R`, R², p-value, significance |
| `ofp_noise_budget.json` | — | repeat spread vs spatial range |
| `ofp_calibration_diagnostic.json` | — | plausibility verdict |

---

## 4. `training & validation/` — the 40-sample labelled table

| Item | Value |
|---|---|
| rows | 32 train + 8 validation = 40 |
| design | **10 gas velocities (0.1–1.0 m/s) × 4 temperatures (600/700/800/900 °C)** |
| features (26) | `Level1..Level12`, `Level1_ratio..level12_ratio`, `Average`, `StandardDeviation` |
| targets (2) | `Ug` (m/s), `Temp` (°C) |

> **Source.** These are the features and targets of the operating-condition
> models. The upstream table they were extracted from (`工作簿1.xlsx`) is not on
> this machine, and none of the recordings archived here carries a temperature
> axis; the file is included as the labelled table it is.

---

## 5. `results/tables/surrogate_recorded_reference.csv`

Numbers recorded during the original work, preserved verbatim with a
`source_notebook` column.  `scripts/06` cannot regenerate them because the
underlying dataset belongs to the laboratory and is not redistributed here, so any
use of these values keeps the provenance column attached.
