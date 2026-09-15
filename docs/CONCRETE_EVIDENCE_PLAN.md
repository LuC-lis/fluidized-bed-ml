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
### 3.0 Bonus results that were not in the original plan *(done — `scripts/11`, `scripts/13`)*
Two of the strongest findings in the repository came out of simply asking what else
these recordings contain:

* **bubble coalescence measured directly** (`scripts/11`): the bubble-passage
  frequency falls with gas velocity at 60 cm (−2.02 Hz per m/s, p < 0.0001) and
  rises at 100 cm (+0.87, p = 0.005) — opposite trends that a pooled analysis
  destroys (R² = −6.2). The two curves cross at 0.33 m/s, a *bed-expansion* event
  distinct from the regime transition;
* **triangulation** (`scripts/13`): one breakpoint estimator applied to pressure,
  vibration and probe observables gives 0.67 / 0.63 / 0.53 / 0.64 m/s, all
  intervals overlapping — the transition is corroborated by three independent
  measurement principles.
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

### 3.2 A working `U_mf` estimate, with an error bar *(done — `scripts/10`)*
**Outcome: the claim is refuted, provably.** The total bed pressure drop moves by
only +2.2 % over 0.1–0.9 m/s (a packed bed would move by +800 %), so the bed is
fluidised at every measured velocity and the fixed-bed branch is absent. The
profile-likelihood interval for `U_mf` covers the whole 0.01–0.60 m/s grid. Bonus:
the same arithmetic implies a bulk density of 662 kg/m³ if the unit is kPa,
largely resolving the unknown pressure unit.
The pressure campaign is at a single temperature, so the classic
`ΔP`-versus-`U_g` intersection method can be applied: fit the fixed-bed line and
the fluidized line, intersect, repeat per tap, and report the spread across taps.
That turns the poster's "±0.05 m/s" into a measured quantity (or shows it is not
achievable — which is also a result). **Cost: half a day.**

### 3.3 A leave-one-condition-out soft-sensor benchmark *(done — `scripts/12`)*
**Outcome:** regression of the continuous gas velocity from probe features gives
R² = 0.63 (within conditions) and 0.57 (unseen radial position), MAE 0.12–0.13 m/s,
against an R² = 0 baseline. Classification, seed-averaged over 10 splits, is
0.829 ± 0.015 (2-class, within conditions) and 0.790 ± 0.015 (unseen position).
Use the probe features to predict the *continuous* gas velocity
(`GroupKFold` by cell, regression). Report `R²` and `RMSE` per protocol. This is
the honest version of "the ANN predicts the hydrodynamic state", with error bars
and a baseline. **Cost: half a day; reuses `scripts/07`.**

### 3.4 Uncertainty-aware reporting *(done — `scripts/12`, `scripts/13`)*
Every classification and regression number is now a mean ± std over **10
independent splits**, and every breakpoint carries a bootstrap interval. Worth
noting what this changed: the original single-split accuracy (0.821) turned out to
be the optimistic end of a 0.790 ± 0.015 distribution — reporting one split would
have overstated the result.

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
