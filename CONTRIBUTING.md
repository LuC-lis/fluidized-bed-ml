# Contributing / maintenance notes

Short, practical notes for whoever touches this repository next (most likely a
future version of its author).

## Ground rules

1. **Never commit raw data.** `.gitignore` blocks `.mat`, `.bin`, `.pvd`, `.xlsx`,
   `.dmg`, `.pptx`, `.pdf`. The recordings are ~4.6 GB and belong to the
   laboratory. Derived tables and figures in `results/` *are* committed on
   purpose — they are the evidence.
2. **Never invent an operating condition.** If a velocity, height or radial
   position is not in the folder name, it must be marked `*_inferred` and its
   recovery rule must be documented and tested.
3. **Never report a physical quantity whose calibration is unverified.** The
   probe case is the worked example: `calibration_plausibility()` exists to block
   it. If you change the calibration, change the diagnostic too.
4. **A new headline number requires three things**: the script that computes it,
   the table/figure it lands in, and a check in `scripts/08_verify_claims.py`.
   Otherwise it does not go in the README.
5. **Negative results are results.** Both of the current ones (no interior
   fluctuation peak; the transition class is not separable) are in the README.

## How to add an analysis

1. Put reusable logic in `src/fbfml/features_*.py` or `analysis_*.py` — never in
   the script. Scripts only orchestrate and plot.
2. Add a numbered `scripts/NN_name.py` with a docstring listing its outputs.
3. Add a test in `tests/`. If it needs the raw data, mark it `@pytest.mark.slow`
   so the default test run stays fast.
4. Add the claim to `scripts/08_verify_claims.py` **before** adding it to the
   README.
5. Regenerate the affected tables/figures and commit them.

## Things that are still open

| Item | Where it is tracked |
|---|---|
| `U_mf` estimate with an error bar | `docs/CONCRETE_EVIDENCE_PLAN.md` §3.2 |
| correct probe calibration constants | `docs/data_quality.md` §4 |
| provenance of the `Temp = 600–900 °C` axis | `docs/data_quality.md` §10 |
| permission to release the raw data | `docs/CONCRETE_EVIDENCE_PLAN.md` §2.4 |
| seed-averaged error bars for the classification results | `docs/CONCRETE_EVIDENCE_PLAN.md` §3.4 |

## Environment

Verified on Python 3.9 with `torch 2.5.1`, `scikit-learn 1.6.1`,
`PyWavelets 1.6.0`, `pandas 2.3.1`, `numpy 2.0.1`, `scipy 1.13.1`,
`matplotlib 3.9.2`, `openpyxl 3.1.5`, `optuna 4.4.0`, `xgboost 2.1.4`.

`shap` is only needed if the explainability part of the surrogate work is
revisited; it is commented out in `requirements.txt` because the dataset it needs
is not redistributable.
