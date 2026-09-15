# Gas–Solid Fluidized Bed: signal processing and machine-learning toolkit

Reproducible analysis of a 3-D gas–solid fluidized-bed measurement campaign:
**differential pressure** (6 taps, 500 Hz), **tri-axial wall vibration** (6
sensors, 1666 Hz) and **optical-fibre-probe local solids fraction** (50 kHz,
2 channels), together with the wavelet feature extraction and the neural-network
surrogates built on top of them.

The repository is a cleaned, documented and *re-runnable* reconstruction of a
research project whose original code lived in a handful of loose notebooks with
hard-coded paths and lost feature definitions. Everything here answers one
question: **which of the project's claims can be reproduced from the raw data,
and which cannot?**

---

## 1. Headline results

| # | Result | Value | Evidence |
|---|--------|-------|----------|
| 1 | **Bubbling → turbulent transition band**, from three independent instrument-only estimators | **0.55 – 0.65 m/s** | `scripts/02`, `results/tables/pressure_regime_transition.json` |
| 2 | The classic "peak of the pressure fluctuation" criterion is **not applicable** in this campaign | summed std still rising at 0.9 m/s | `results/figures/pressure_std_vs_velocity.png` |
| 3 | **Wavelet feature definition recovered** and verified bit-exactly against the group's stored tables | 54/54 comparisons, worst relative error **6×10⁻¹²** | `scripts/03`, `tests/test_wavelet_matches_vendor.py` |
| 4 | Local solids fraction field built from **434 raw probe records** (86 conditions) | 10.49 s @ 50 kHz each | `results/tables/ofp_conditions.csv` |
| 5 | **Core–annulus structure detected**: probe signal lower on the axis than near the wall | **14/17** conditions positive, sign test **p = 0.006** | `results/figures/ofp_radial_gradient.png` |
| 6 | Error budget of the probe campaign | repeat spread 85 mV vs spatial range 949 mV → **noise/signal = 0.09** | `results/tables/ofp_noise_budget.json` |
| 7 | **Absolute `εs` calibration is not usable** for this campaign and is flagged automatically | recovered constants give a flat 0.29–0.51 field | `results/tables/ofp_calibration_diagnostic.json` |
| 8 | Gas velocity recovered for the unlabelled vibration folders via the **acquisition-timestamp rule**, validated on the two labelled folders | 9-point sweep, 2/2 labels reproduced | `scripts/05`, `tests/test_data_readers.py` |
| 9 | Vibration features that **track the velocity sweep** (Spearman ρ) | `Level9_ratio` 0.87, dominant frequency 0.83, `Level6_ratio` 0.82, `std` 0.78 | `results/tables/vibration_monotonicity.csv` |
| 10 | Wide-ANN surrogate + activation study *(recorded in the legacy notebooks; reproducible via `scripts/06` once the group's dataset is supplied)* | mean 5-fold CV **R² = 0.9455**; LeakyReLU 0.945 > ELU 0.934 > GELU 0.930 > ReLU 0.814 > Tanh 0.671 | `results/tables/surrogate_recorded_reference.csv` |
| 11 | **Flow regime classified from 10.5 s of optical-probe signal** (434 records, 36 features) | bubbling vs turbulent: **0.829 ± 0.015** accuracy over **10 independent splits** (majority baseline 0.59), macro-F1 0.82; **0.790 ± 0.015** when the test set is a radial position never seen in training | `scripts/07`, `scripts/12`, `results/tables/robustness_classification_summary.csv` |
| 12 | Two **negative results** reported as such | the *transition* regime is not separable as a discrete third class (macro-F1 ≈ 0.51), and the exact gas velocity is not recoverable as 9 classes (0.29 vs 0.13 baseline, but 59–65 % within ±0.1 m/s) | `results/figures/probe_confusion_matrices.png` |
| 13 | **Continuous** gas velocity from the same 10.5 s of probe signal (regression benchmark) | **R² = 0.63 ± 0.00**, MAE **0.119 m/s** within conditions; **R² = 0.57 ± 0.00**, MAE **0.127 m/s** on an unseen radial position (baseline R² = 0) | `scripts/12`, `results/tables/robustness_regression_summary.csv` |
| 14 | **Bubble coalescence measured directly**: at the low plane (60 cm) the bubble-passage frequency *falls* with gas velocity | **−2.02 Hz per (m/s), p < 0.0001** (n = 45 records); at 100 cm it *rises* (+0.87, p = 0.005). Two strong trends of opposite sign | `scripts/11`, `results/figures/bubble_rate_vs_velocity.png` |
| 15 | **Four independent observables triangulate the transition** with one estimator | top-tap pressure **0.67**, pressure std ratio **0.63**, wall vibration **0.53**, probe classifier **0.64 m/s** — all intervals overlap, 4/4 within ±0.05 of the pressure-only band | `scripts/13`, `results/figures/triangulation_forest.png` |
| 16 | The poster's **±0.05 m/s** `U_mf` claim, and the unknown **pressure unit** | `U_mf` is **not identifiable** (profile-likelihood interval spans the whole 0.01–0.60 m/s grid): the fixed-bed branch was never sampled. The same numbers imply a bulk density of **662 kg/m³** *if* the unit is kPa — inside the expected FCC range | `scripts/10`, `results/figures/umf_identifiability.png` |

<p align="center">
  <img src="results/figures/pressure_mean_vs_velocity.png" width="47%">
  <img src="results/figures/ofp_radial_gradient.png" width="47%">
</p>

---

## 2. Repository layout

```
fluidized-bed-ml/
├── src/fbfml/                   reusable package — one responsibility per module
│   ├── paths.py                 single source of truth for data locations
│   ├── data_pressure.py         pressure logs (.xlsx, 500 Hz, 6 taps)
│   ├── data_vibration.py        accelerometer logs (.mat/.bin) + velocity inference
│   ├── data_ofp.py              optical-fibre-probe records (.pvd) + calibration
│   ├── features_wavelet.py      db4 12-level DWT features (validated)
│   ├── features_stats.py        time-domain descriptors
│   ├── analysis_inventory.py    asset register & completeness audit
│   ├── analysis_regime.py       transition velocity & axial profiles
│   ├── analysis_concentration.py solids-fraction field, gradient tests, error budget
│   ├── analysis_vibration.py    vibration features, ordering, monotonicity
│   ├── model_wide_ann.py        WideANN / MLP / ResidualMLP definitions
│   ├── train_cv.py              seeded k-fold harness (no scaler leakage)
│   └── viz.py                   plotting style + figure/table writers
├── scripts/                     numbered, runnable entry points (see below)
├── tests/                       regression tests, incl. the wavelet validation
├── legacy/                      archived original notebooks + MATLAB + mapping
├── docs/                        setup, data dictionary, quality report, contributions
└── results/{figures,tables}/    committed outputs of the scripts
```

### Scripts

| Script | What it produces |
|---|---|
| `01_data_inventory.py` | asset register, pressure/vibration health, probe coverage matrix |
| `02_pressure_regime.py` | transition-velocity estimators, axial profiles, PSDs, wavelet feature table |
| `03_wavelet_validation.py` | proof that the recovered feature extractor matches the stored tables |
| `04_ofp_concentration.py` | 434 records → solids-fraction field, gradient test, error budget |
| `05_vibration_features.py` | wavelet + time-domain features, velocity recovery, monotonicity |
| `06_train_surrogate.py` | Wide-ANN 5-fold CV + activation ablation (needs the group's dataset) |
| `07_regime_classification.py` | flow-regime / velocity classification from the probe signal, with baselines and confusion matrices |
| `08_verify_claims.py` | re-checks every headline number in this README against the generated tables |
| `09_package_data_release.py` | manifest + SHA-256 + datasheet for a public data release |
| `10_umf_analysis.py` | total bed pressure drop, `U_mf` identifiability, pressure-unit cross-check |
| `11_bubble_frequency.py` | bubble-passage statistics: coalescence vs bed expansion |
| `12_robustness.py` | seed-averaged classification and a continuous-velocity regression benchmark |
| `13_triangulation.py` | same breakpoint estimator applied to four independent observables |

---

## 3. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point the code at the raw recordings (default: the parent directory of the repo)
export FBFML_DATA_ROOT="/path/to/raw/data"     # optional

python scripts/01_data_inventory.py
python scripts/02_pressure_regime.py
python scripts/03_wavelet_validation.py
python scripts/04_ofp_concentration.py         # ~2 min, caches to results/tables/
python scripts/05_vibration_features.py        # ~1 min
python scripts/07_regime_classification.py     # ~3 min, caches to results/tables/
python scripts/10_umf_analysis.py              # pressure-drop bookkeeping
python scripts/11_bubble_frequency.py          # ~3 min
python scripts/12_robustness.py                # ~10 min (10 seeds)
python scripts/13_triangulation.py
python scripts/08_verify_claims.py             # 37/37 claims re-checked
pytest -q                                      # add FBFML_SLOW_TESTS=1 for the full set
```

Expected layout of the raw data (see `docs/data_dictionary.md`):

```
<data root>/
├── Pressure_Energy/     0.1.xlsx … 0.9.xlsx, energy_0.1.xlsx … energy_0.9.xlsx
├── 20250609 LMF/        <condition>/1.mat … 7.mat (+ .bin)
└── 20250609 LC/         <height_cm>/<r_R>/<U_g>/*.pvd
```

`src/fbfml/paths.py` resolves everything from `FBFML_DATA_ROOT`; no absolute path
is hard-coded anywhere in the package. Run `python -c "from fbfml import paths;
print(paths.describe())"` to see what was resolved.

---

## 4. What is *not* reproducible here

Intellectual honesty is part of the deliverable:

1. **The two modelling datasets are not in this repository.** `工作簿1.xlsx`
   (40 labelled samples: 10 gas velocities × 4 temperatures) and
   `dataset_training.xlsx` (the 76/152 mm surrogate dataset) belong to the
   supervising group and are not redistributable. Their recorded results are
   preserved in `results/tables/surrogate_recorded_reference.csv` with explicit
   provenance, and `scripts/06` can regenerate equivalent numbers once the data
   is supplied.
2. **The absolute solids-fraction calibration is unknown.** The two constant sets
   recovered from the legacy scripts belong to a different probe/setting. The
   pipeline therefore reports the calibration-free probe voltage as the primary
   measurement, and automatically flags the calibrated values as unusable
   (`docs/data_quality.md`, §4).
3. **The temperature axis of the 40-sample table has no raw-data trail** in this
   folder. It is documented as an open question rather than silently dropped.
4. **Eight of the ten vibration folders carry no gas-velocity label.** The
   velocity is inferred from the acquisition timestamp; the rule is validated
   against the two labelled folders and is always marked `velocity_inferred`.

---

## 5. Relationship to the group's publication

This project was carried out in the Fluidization Laboratory of the Beacons of
Excellence Research and Innovation Institute, UNNC. A later study by the
supervising group — Chen, X. *et al.*, *Data-driven machine learning for scale-up
of bubbling and turbulent fluidized beds*, **Engineering Applications of
Computational Fluid Mechanics** 20(1):2670348 (2026),
[doi:10.1080/19942060.2026.2670348](https://doi.org/10.1080/19942060.2026.2670348)
— is the peer-reviewed descendant of the modelling line explored here (deep-width
ANN surrogate + SHAP, 5-fold CV, 76/152 mm beds).

**The author of this repository is not an author of that paper and is not named
in its acknowledgements.** The publication is cited as scientific context for the
method family developed in this project, not as a claim of authorship. Please
read `docs/CONTRIBUTIONS.md` for an explicit statement of what is and is not
attributable to this work.

---

## 6. Provenance and integrity

* The db4/12-level/`symmetric` feature definition in `features_wavelet.py` was
  **reverse-engineered** from the stored `energy_*.xlsx` tables after the original
  generator was lost, then verified to machine precision (result #3).
* The condition → velocity mapping in `data_vibration.py` was **recovered** from
  folder timestamps and validated against the only two labelled folders.
* Data defects (four truncated recordings, 30 incomplete probe conditions, a
  missing sensor, unlabelled folders) are reported, not hidden — see
  `docs/data_quality.md`.
* The k-fold harness in `train_cv.py` fits its scaler **inside** each fold; the
  legacy notebooks fitted it on the full dataset before splitting, which inflates
  the reported scores. That difference is called out wherever it matters.

## 7. Licence

Code: MIT. Derived tables and figures in `results/`: CC BY 4.0. The raw
experimental data is not covered by either and remains the property of the
laboratory.

---

## 8. 中文速览

这个仓库把一个「只剩一堆笔记本、绝对路径写死、连自己的特征定义都丢了」的暑期科研项目，重建成**可复现、可检验**的产物。四件最重要的事：

1. **重建了丢失的小波特征定义**。原生成脚本已不存在，我从项目自己的结果表里反推出定义，并验证到机器精度：**54/54 完全吻合，最大相对误差 6×10⁻¹²**，并配了回归测试（`tests/test_wavelet_matches_vendor.py`）。
2. **用压力数据独立重求流型转变气速**，三个互不依赖的判据给出 **0.55–0.65 m/s**（与海报的 0.6–0.8 m/s 重叠），并给出一个负结果：教科书的「压力脉动峰值」判据在本实验里**不成立**（0.9 m/s 前脉动仍在上升）。
3. **用 50 kHz 光纤探针信号做流型分类（全新结果）**：鼓泡 vs 湍动 **83% ± 1.5%**（10 个随机划分的均值±标准差；多数类基线 59%，macro-F1 0.82），换成「从未见过的径向位置」是 **79% ± 1.5%**；把目标当连续量做回归则 **R²=0.63，MAE=0.12 m/s**。同时诚实报告两个负结果：**过渡区不能当作离散第三类**（macro-F1 0.51），9 类气速分类基本失败（0.29 vs 0.13 基线）。
4. **三个物理上互不相关的仪器给出同一个转变区间**：压差上测点 0.67、压差脉动比 0.63、壁面振动 0.53、探针分类概率 0.64 m/s，四个置信区间全部重叠 → 0.55–0.65 m/s 的转变是**多仪器相互印证**的，而不是单一仪器的假象。
5. **直接测到气泡聚并**：低位面（60 cm）气泡通过频率随气速**下降**（−2.02 Hz/(m/s)，p<0.0001，n=45），高位面（100 cm）**上升**（+0.87，p=0.005）——两条方向相反的显著趋势，交叉点在 0.33 m/s，那是**床层膨胀**事件，比流型转变更早，两者不能混为一谈。
6. **把海报里的「±0.05 m/s」讲清楚**：总床层压降在 0.1–0.9 m/s 内只变 **+2.2%**（堆积床应为 +800%），说明床层全程已流化、**固定床支路根本没被采到**，所以 U_mf 不可辨识（profile-likelihood 区间横跨整个 0.01–0.60 网格）——这个数字无法从本数据复现，且原因可证。顺带推出压力单位很可能是 kPa（隐含堆积密度 662 kg/m³，落在 FCC 床预期区间）。
4. **数据质量审计**：4 个损坏录音、1 个永久缺失的传感器、22 个空测量工况、未标注的工况、未知压力单位，以及**一套不可用的探针标定**——最后这条已经被写进代码，会自动拒绝输出物理上不可能的结果。

另外三个文件是直接可用的：

| 想要什么 | 看哪里 |
|---|---|
| 简历该怎么写（3 个版本 + 逐条证据审计 + 不能出现的说法） | `docs/resume/RESUME_STATEMENTS.md` |
| 我的贡献到底有哪些、哪些不能算 | `docs/CONTRIBUTIONS.md` |
| 怎么把它变成 GitHub / DOI / 论文 | `docs/CONCRETE_EVIDENCE_PLAN.md` |

启动：

```bash
export FBFML_DATA_ROOT="/path/to/raw/data"   # 原始 4.6 GB 数据不在仓库里
python scripts/01_data_inventory.py
python scripts/03_wavelet_validation.py     # 关键验证
python scripts/07_regime_classification.py  # 分类：83%±1.5% vs 59% 基线
python scripts/10_umf_analysis.py           # 总压降恒定 + U_mf 不可辨识
python scripts/11_bubble_frequency.py       # 气泡聚并：低频面频率随气速下降
python scripts/13_triangulation.py          # 三个仪器三角定位转变气速
python scripts/08_verify_claims.py          # 37/37 条 README 结论自检
pytest -q
```
