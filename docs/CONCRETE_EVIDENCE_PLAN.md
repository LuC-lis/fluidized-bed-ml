# How to make this work concrete and checkable

The gap to close is simple: right now the project is a folder on a laptop. The
following turns it into artefacts a stranger can open, run and verify. Ordered by
return on effort, with the honest cost of each step.

---

## Tier 1 — do it this week (no permissions needed, no new science)

| Step | Action | Output |
|---|---|---|
| 1.1 | Publish `fluidized-bed-ml/` as a **public GitHub repository** (code + `results/` + `docs/`; raw data excluded by `.gitignore`) | a URL to put on a CV |
| 1.2 | Enable the **GitHub → Zenodo** integration and mint a DOI from a tagged release | a citable artefact: `doi:10.5281/zenodo.XXXXXXX` |
| 1.3 | Add three **CI badges** (GitHub Actions: install, run `pytest`, run `scripts/01`+`03`) | proof the repo runs, not just looks runnable |
| 1.4 | Write one **800-word technical note** ("Recovering a lost wavelet feature definition from derived tables") as a repo `docs/` page or a blog post and link it from the README | a written sample that shows the *reasoning*, which is what admissions readers cannot get from a CV line |
| 1.5 | Add the repo link to the CV entry for this project | the CV line becomes verifiable in one click |

Why this is the highest-value step: it converts every claim into something
inspectable. A reader who clicks through sees a validation table reporting
`54/54 exact matches, worst relative error 6e-12`, a data-quality report that
volunteers the project's own defects, and a README that says which claim did *not*
survive. That combination is rare in an undergraduate portfolio.

**Cost:** 1–2 days. **Risk:** none.

---

## Tier 2 — needs one email to the group (do it in parallel)

Send the supervisor / PhD student a short, specific message asking for:

| # | Question | Why it unlocks something |
|---|---|---|
| 2.1 | Confirm the velocity ↔ condition-folder mapping (the timestamp rule) | converts `velocity_inferred = True` into a measured label |
| 2.2 | The **probe-specific calibration constants** `(a, b)` for this campaign | unlocks the absolute solids-fraction figures (`analysis_concentration.py` already accepts them) |
| 2.3 | Where the `Temp = 600–900 °C` axis of the 40-sample table came from | resolves the last provenance gap in `docs/data_quality.md` §10 |
| 2.4 | Permission to release the **derived** tables publicly, and — if possible — the raw recordings | a dataset release needs this in writing |
| 2.5 | Whether the group wants this repository cited/linked from anywhere | avoids any awkwardness around the published paper |

**Important:** do **not** ask to be added to the paper, and do not claim
authorship. Ask about data facts only. The value here is scientific accuracy, not
credit.

**Cost:** one email. **Risk:** low; worst case is no reply, and Tier 1 already
stands alone.

---

## Tier 3 — new analysis that only this data can support (the strongest move)

### 3.1 Regime / gas-velocity classification from the probe signal ⭐ *(done — `scripts/07`)*
`scripts/07_regime_classification.py` trains three classifiers on **36 features
extracted from 10.5 s of optical-probe signal** per record, over 434 records, for
three labelling schemes (9-class velocity, 3-class regime, binary
bubbling-vs-turbulent), under two honest protocols (leave-one-replicate-out;
test on an unseen radial position), each reported next to its majority-class
baseline.

This is the analysis that **replaced the unsupported ">95 % classification
accuracy"** with a real, defensible, reproducible number. The results that came
out of it:

| Task | Best accuracy | Majority baseline | Macro-F1 |
|---|---|---|---|
| 9-class gas velocity | 0.29 | 0.13 | 0.30 (0.59–0.65 within one velocity step) |
| 3-class regime | 0.70 | 0.52 | 0.50 — the *transition* class is not separable |
| **2-class bubbling vs turbulent** | **≈0.82** | 0.59 | **≈0.81** — still 0.80 on an unseen radial position |

Why it was worth doing:

* the probe dataset has a genuinely dense design (434 records, 86 conditions) —
  unlike the 40-sample table, it can support a classifier;
* the poster's own claim ("the ANN distinguishes bubbling / transition /
  turbulent") becomes testable;
* it is a *new* result that nobody has published, obtained from data that was
  already sitting in the folder.

### 3.2 A working `U_mf` estimate, with an error bar *(open — the remaining place where
the poster's "±0.05 m/s" could become a real number)*
The pressure campaign is at a single temperature, so the classic
`ΔP`-versus-`U_g` intersection method can be applied: fit the fixed-bed line and
the fluidized line, intersect, repeat per tap, and report the spread across taps.
That turns the poster's "±0.05 m/s" into a measured quantity (or shows it is not
achievable — which is also a result). **Cost: half a day.**

### 3.3 A leave-one-condition-out soft-sensor benchmark
Use the probe features to predict the *continuous* gas velocity
(`GroupKFold` by cell, regression). Report `R²` and `RMSE` per protocol. This is
the honest version of "the ANN predicts the hydrodynamic state", with error bars
and a baseline. **Cost: half a day; reuses `scripts/07`.**

### 3.4 Uncertainty-aware reporting
Repeat every classification/regression result over ≥10 seeds and report mean ± std
plus a bootstrap confidence interval. Cheap, and it makes the numbers
interview-proof. **Cost: a few hours** (the harness already takes `seed`).

---

## Tier 4 — the "real research artefact" option (only if it is what you want)

If the goal is a *publication* rather than a portfolio piece, the honest target is
a **data/methods paper** (e.g. *Data in Brief*, *SoftwareX*, *Journal of Open
Research Software*) on:

> an annotated, quality-audited dataset of pressure, vibration and optical-probe
> measurements in a 3-D gas–solid fluidized bed across the bubbling-to-turbulent
> transition, with a reproducible feature-extraction pipeline.

Requirements before that is realistic: 2.1–2.4 from Tier 2 (especially data
release permission and the calibration), plus a co-author from the laboratory
(a data paper without the data owner is not viable). Do not start here.

---

## What to put where

| Artefact | Where | What it proves |
|---|---|---|
| Repository | GitHub, linked from the CV | you can build software, not just notebooks |
| DOI | Zenodo, from a release tag | citability; shows awareness of research practice |
| `docs/data_quality.md` | in the repo | you find and disclose problems in data you did not collect |
| `scripts/03` + its test | in the repo | you can recover a lost specification and prove it |
| `scripts/07` results | in the repo | you can design an honest evaluation protocol |
| Technical note | repo `docs/` or blog | you can write about method |
| One-paragraph summary of the negative results | in the statement of purpose | intellectual maturity |

---

## Red lines (do not do these)

1. **Do not claim authorship of, or acknowledgement in, Chen et al. (2026).** It is
   a real paper with real authors; you are not among them.
2. **Do not repeat ">95 % classification accuracy"** unless Tier 3.1 produces that
   number for a stated task, protocol and baseline — and if it does, quote the
   protocol too.
3. **Do not say TensorFlow.** The code is PyTorch.
4. **Do not quote a `U_mf` error** until Tier 3.2 is done.
5. **Do not release the raw recordings** without written permission from the
   laboratory (2.4).
