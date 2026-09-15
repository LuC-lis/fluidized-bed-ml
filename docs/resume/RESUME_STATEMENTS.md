# Rewritten résumé statements

## 0. 中文说明：改了什么，为什么

**原稿最大的问题不是写得不好，而是两处会被追问就崩掉的事实错误：**

| 原稿 | 问题 | 已改为 |
|---|---|---|
| `Trained BP neural networks with **TensorFlow**` | 全项目**没有任何 TensorFlow 代码**，用的是 PyTorch | PyTorch；若被追问，这就是诚信问题，必须改 |
| `achieving **>95%** classification accuracy` | 项目里**从来没有分类任务**（没有 accuracy / 混淆矩阵），唯一接近 0.95 的数字是**另一个数据集上的回归 R²** | 换成我这次**真做出来并可复现**的：**82% vs 59% 基线**（鼓泡/湍动二分类），换到未见过的径向位置仍有 80% |
| `±0.05 m/s error in Umf` | 只出现在海报上，**没有任何代码/表格算过 U_mf** | 先删掉；等做完 `CONCRETE_EVIDENCE_PLAN.md` §3.2 再加回来 |
| `supervised by Prof. Xiaoyang Wei` | 海报上写的是 guided by **Dr. Xi Chen** | 两者都写，区分导师与日常指导 |
| 完全没提论文 | 论文上**没有你的名字** | 继续不提。只能作为「同一课题方向」cientific context 引用，**绝不能写成产出** |

**同时把三件真正硬、但原稿一个字都没写的东西补了进去：**

1. 重建丢失的小波特征定义 → 54/54 完全吻合，最大相对误差 6×10⁻¹²；
2. 压力数据独立重求流型转变 → **0.55–0.65 m/s**（+ 一个负结果）；
3. 探针信号做流型分类 → **82%**（基线 59%；未见位置 80%），并把过渡区不可分作为负结果写出来。

下面 Version A 是推荐版；§4 是逐条证据审计；§5 是**绝对不能出现的说法清单**。

---

Three versions, all built only from facts that can be verified in this
repository. Read `docs/CONTRIBUTIONS.md` for the evidence behind each clause and
`§4 Evidence audit` below for what was removed from the previous draft and why.

---

## 1. Version A — research CV (recommended)

> ### Machine Learning for Gas–Solid Fluidized-Bed Diagnostics
> **Research Assistant** · Beacons of Excellence Research and Innovation Institute, University of Nottingham Ningbo China · *06/2025 – 09/2025*
> *Supervisor: Prof. Xiaoyang Wei; day-to-day guidance: Dr. Xi Chen*
>
> - Built an end-to-end, reproducible analysis pipeline (Python, PyTorch) over **4.6 GB of raw experimental recordings** — six differential-pressure transducers (500 Hz), seven tri-axial accelerometers (1666 Hz) and a dual-tip optical fibre probe (50 kHz) — replacing a set of loose notebooks that hard-coded absolute paths and had lost the definition of their own model inputs. The pipeline is packaged as a tested library, streams the recordings without exhausting memory, and reproduces every reported figure with one command per stage.
> - **Recovered a lost feature specification.** The script that produced the project's wavelet feature tables was missing, so I reverse-engineered the definition from the derived tables alone and verified it against them to machine precision — **54/54 channel–velocity comparisons exact, worst relative error 6 × 10⁻¹²** — and locked it in with a regression test.
> - **Re-derived the flow-regime transition from raw pressure data** using three independent instrument-only estimators, obtaining **0.55–0.65 m/s** (consistent with the 0.6–0.8 m/s previously reported by the group), and documented that the classic pressure-fluctuation-peak criterion is *not* applicable to this campaign — a negative result the earlier account omitted.
> - **Classified the flow regime from 10.5 s of optical-probe signal**, engineering 36 time-domain, spectral and wavelet features across **434 records / 86 conditions**: bubbling versus turbulent classification reaches **0.829 ± 0.015 accuracy against a 0.59 majority-class baseline (macro-F1 0.82), averaged over ten independent splits**, and **0.790 ± 0.015 when the test set is a radial measurement position never seen during training**. Treating the same features as a continuous soft sensor recovers the gas velocity with **R² = 0.63 and MAE = 0.12 m/s** (unseen position: 0.57 / 0.13 m/s). Two negative results are reported with equal prominence: the *transition* regime is not separable as a discrete third class (macro-F1 0.51), and nine-class velocity classification fails (0.29 vs 0.13 baseline, though 59–65 % of errors land within one velocity step).
> - **Cross-validated the transition across three independent instruments.** Applying one breakpoint estimator to four physically unrelated observables — top-tap pressure, the top/bottom pressure-fluctuation ratio, wall-vibration intensity and the probe classifier's turbulent probability — gives **0.67 / 0.63 / 0.53 / 0.64 m/s** with fully overlapping bootstrap intervals, all within ±0.05 m/s of each other. Separately, the probe measures **bubble coalescence directly**: the bubble-passage frequency falls with gas velocity at 60 cm (**−2.02 Hz per m/s, p < 0.0001**) and rises at 100 cm (**+0.87, p = 0.005**), two significant trends of opposite sign that a pooled analysis destroys (two-segment fit R² = −6.2).
> - Audited the campaign and made the analysis code enforce its own limits: four truncated recordings, a permanently missing sensor, 22 empty measurement conditions, unlabelled operating conditions, an unknown pressure unit and an **unusable probe calibration** — the last of which the code now detects automatically so that physically implausible values can never be reported as results.
> - Implemented a leakage-free cross-validation harness (scaler fitted inside each fold, seeded end-to-end) and a deep-wide ANN surrogate for a companion study of the group (**5-fold cross-validated R² = 0.95** as recorded, with an activation-function ablation spanning 0.671–0.945 on identical splits).
>
> *Code, figures and the full data-quality report: github.com/&lt;user&gt;/fluidized-bed-ml*

---

## 2. Version B — condensed (two-column or one-page CV)

> **Machine Learning for Gas–Solid Fluidized-Bed Diagnostics** — Research Assistant, UNNC · 06–09/2025
> Reprocessed 4.6 GB of pressure, vibration and optical-probe recordings from a 3-D fluidized bed into a tested Python pipeline; **recovered a lost wavelet feature definition and verified it to machine precision (54/54 exact)**; **re-derived the bubbling-to-turbulent transition at 0.55–0.65 m/s and corroborated it with three independent instruments (0.53–0.67 m/s, overlapping intervals)**; **classified the flow regime from 10.5 s of probe signal at 0.83 ± 0.02 accuracy (baseline 0.59), holding at 0.79 ± 0.02 on an unseen measurement position**; **measured bubble coalescence directly** (probe frequency falls with velocity at 60 cm, p < 0.0001); produced a data-quality audit the code now enforces, including a proof that the campaign's `U_mf` claim is not identifiable.

Three-line variant if space is very tight:

> - Rebuilt a 4.6 GB experimental dataset into a tested, path-agnostic Python pipeline (PyTorch, signal processing).
> - Recovered a lost wavelet feature definition and verified it to machine precision (54/54 comparisons, worst error 6 × 10⁻¹²).
> - Classified bubbling vs turbulent flow from 10.5 s of probe signal at **0.83 ± 0.02 accuracy (0.59 baseline)**, 0.79 ± 0.02 on an unseen measurement position; recovered the continuous gas velocity at **MAE = 0.12 m/s** from the same features.
> - Corroborated the bubbling-to-turbulent transition at **0.53–0.67 m/s across three independent instruments**, and measured **bubble coalescence** directly (passage frequency −2.0 Hz per m/s at 60 cm, p < 0.0001).

---

## 3. Version C — narrative paragraph (statement of purpose / portfolio)

> The project I am most willing to be judged on is not the one with the highest
> accuracy number. I joined a fluidization laboratory for a summer and inherited
> several gigabytes of experimental recordings, a folder of near-duplicate
> notebooks, and no documentation. The script that generated the model inputs was
> gone. Rather than start from a blank page I decided to find out what the data
> could actually support: I reverse-engineered the wavelet feature definition from
> the derived tables and proved it matched to machine precision; I recovered the
> gas velocity for eight unlabelled recording sessions from their timestamps and
> validated the rule against the two sessions that were labelled; and I rebuilt the
> solids-fraction field from 434 raw probe records while concluding, and reporting,
> that the available calibration constants cannot be trusted.
>
> Two findings from that work are the ones I would defend in an interview. First, a
> straightforward classifier on 10.5 seconds of optical-probe signal separates the
> bubbling from the turbulent regime at 0.83 ± 0.02 accuracy against a 0.59
> baseline — and still at 0.79 when the test set is a measurement position the model
> has never seen — while the *transition* regime refuses to be treated as a third
> discrete class, which is exactly what a continuum should do. Second, my
> re-analysis of the pressure data puts the transition at 0.55–0.65 m/s rather than
> the 0.6–0.8 m/s in the original materials; applying one breakpoint estimator to
> pressure, vibration and probe observables independently gives 0.53–0.67 m/s with
> overlapping intervals; and the probe turns out to measure bubble coalescence
> directly — the passage frequency *falls* with gas velocity in the dense zone,
> which is the opposite of what intuition suggests and exactly what bubble
> coalescence predicts.
>
> What I take from it is a habit rather than a result: before improving a model,
> establish what the data can carry. That is the kind of work I want to do next.

---

## 4. Evidence audit of the previous draft

| Previous statement | Verdict | What replaced it |
|---|---|---|
| "Developed an ANN-based surrogate model … to predict pressure-drop evolution, minimum fluidization velocity and flow regime transitions" | ⚠️ over-claimed: no model predicted pressure drop or `U_mf` | replaced with the two things that were actually modelled/derived, each with its protocol |
| "under varying gas velocities (0.1–0.9 m/s)" | ✅ supported | kept |
| "Processed pressure and vibration signals using Python (NumPy, Matplotlib)" | ⚠️ true but undersells: the wavelet feature work and the probe parsing are the substance; the vibration post-processing was MATLAB | replaced with the concrete pipeline description |
| "Trained BP neural networks with **TensorFlow**" | ❌ **false** — no TensorFlow anywhere; the code is PyTorch | "PyTorch"; the ANN surrogate bullet now also states the evaluation protocol |
| "achieving **>95 % classification accuracy**" | ❌ **not supported** — no classifier existed; the only ≈0.95 figure is a regression R² on a different dataset | replaced with the **measured, seed-averaged** result from `scripts/07` + `scripts/12`: 0.829 ± 0.015 within conditions and 0.790 ± 0.015 on an unseen radial position, against a 0.59 majority baseline |
| "**±0.05 m/s** error in minimum fluidization velocity prediction" | ❌ **refuted.** Investigated in full (`scripts/10`): the total bed pressure drop is constant to +2.2 % over 0.1–0.9 m/s, so the bed is fluidised at every measured velocity and the fixed-bed branch was never sampled; the profile-likelihood interval for `U_mf` spans the entire 0.01–0.60 m/s grid | removed. The positive version of this bullet is now "showed that `U_mf` is *not* identifiable from this campaign, and why" |
| Supervised by Prof. Xiaoyang Wei | ✅ but the poster credits day-to-day guidance to Dr. Xi Chen | both named, with roles distinguished |
| (missing) the published descendant paper | — | **not used.** The paper (Chen *et al.*, 2026) has its own authors and does not acknowledge this work; citing it as an output would be misrepresentation. It may only be referenced as *scientific context*, never as an output |

## 5. Phrases that must not appear

* "TensorFlow", "Keras" — the framework is PyTorch.
* ">95 % classification accuracy" — unless the task, protocol **and** baseline are
  quoted alongside it.
* "±0.05 m/s" for `U_mf` — until §3.2 of the plan is done.
* "co-author", "contributed to the paper", "acknowledged in" — none of these are true.
* "I designed the experiments" / "I collected the data" — the raw recordings were
  made by the laboratory; the contribution is the analysis, the recovery work and
  the audit.
* "My model predicts pressure drop" — no such model exists.

## 6. Interview-ready one-liners

* *"How many samples?"* → 434 probe records (86 conditions), 9 pressure records
  × 60 000 points × 6 taps, 60 accelerometer records. The old modelling table had
  40 samples, which is why its temperature target never became learnable.
* *"What was the accuracy?"* → 0.829 ± 0.015 on bubbling vs turbulent over ten
  splits, 0.59 majority baseline, 0.790 ± 0.015 on an unseen radial position;
  3-class macro-F1 only 0.51; continuous velocity MAE 0.12 m/s.
* *"How do you know the transition is real?"* → four observables, one breakpoint
  estimator, all intervals overlapping between 0.53 and 0.67 m/s.
* *"What went wrong?"* → The probe calibration constants belong to another
  instrument, so the code now refuses to report absolute solids fraction; the
  original notebooks fitted their scaler before splitting, so their CV scores are
  optimistic; and four recordings are truncated with one sensor missing entirely.
* *"What are you proudest of?"* → That the repository states which of its own
  claims failed, and that a test fails the build if the recovered feature
  definition ever drifts.
