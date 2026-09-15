#!/usr/bin/env python
"""
08 -- Verify the headline claims in the README against the generated results.

Every number quoted in `README.md` is re-checked here from the result tables.
If a claim and an artefact ever disagree, this script fails. It is the cheapest
possible defence against a README that drifts away from its own evidence.

Run after the analysis scripts:

    python scripts/08_verify_claims.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import paths

FAILURES: list = []
CHECKS: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    tables = paths.TABLES_DIR
    if not tables.is_dir():
        print("no results/tables directory - run scripts 01-07 first")
        return 1

    # --- 3. wavelet feature definition recovered exactly -------------------
    wv = pd.read_csv(tables / "wavelet_validation.csv")
    check("wavelet: all comparisons match", bool(wv["matches"].all()),
          f"{int(wv['matches'].sum())}/{len(wv)} matched")
    check("wavelet: worst relative error < 1e-9", float(wv["max_rel_error"].max()) < 1e-9,
          f"worst = {wv['max_rel_error'].max():.3e}")

    # --- 1-2. transition velocity ------------------------------------------
    transition = json.loads((tables / "pressure_regime_transition.json").read_text())
    band = transition["recommended_band"]
    check("transition: band within 0.5-0.7 m/s", 0.5 <= band[0] and band[1] <= 0.7,
          f"band = {band}")
    check("transition: no interior fluctuation peak (negative result holds)",
          transition["fluctuation_peak_interior"] is False,
          f"fluctuation_peak_interior = {transition['fluctuation_peak_interior']}")

    # --- 4-7. probe field ---------------------------------------------------
    conditions = pd.read_csv(tables / "ofp_conditions.csv")
    records = pd.read_csv(tables / "ofp_records.csv")
    check("probe: 434 records parsed", len(records) == 434, f"got {len(records)}")
    check("probe: >= 80 conditions", len(conditions) >= 80, f"got {len(conditions)}")

    gradient = pd.read_csv(tables / "ofp_radial_gradient_test.csv")
    positive = int(gradient["slope_positive"].sum())
    check("probe: majority of radial gradients positive",
          positive > len(gradient) / 2, f"{positive}/{len(gradient)}")

    budget = json.loads((tables / "ofp_noise_budget.json").read_text())
    check("probe: noise/signal < 0.5", budget["noise_to_signal"] < 0.5,
          f"noise/signal = {budget['noise_to_signal']:.3f}")

    diagnostic = json.loads((tables / "ofp_calibration_diagnostic.json").read_text())
    check("probe: calibration flagged unusable (as documented)",
          diagnostic["calibration_usable"] is False,
          f"calibration_usable = {diagnostic['calibration_usable']}")

    # --- 8-9. vibration velocity recovery ----------------------------------
    vib = pd.read_csv(tables / "vibration_features.csv")
    sweep = vib.dropna(subset=["condition_rank"])["velocity_assigned"].nunique()
    check("vibration: nine-point velocity sweep recovered", sweep == 9, f"got {sweep}")
    labels = vib.dropna(subset=["velocity_label"])[
        ["condition", "velocity_label", "velocity_assigned"]
    ].drop_duplicates()
    agree = bool(np.allclose(labels["velocity_label"], labels["velocity_assigned"]))
    check("vibration: timestamp rule reproduces the labelled folders", agree,
          f"{len(labels)} labelled folder(s)")

    sensor_check = pd.read_csv(tables / "vibration_per_sensor_check.csv")
    median_rho = float(sensor_check["spearman_rho"].median())
    check("vibration: median per-sensor Spearman rho > 0.6", median_rho > 0.6,
          f"median rho = {median_rho:.3f}")

    # --- 11-12. classification (single-split) ------------------------------
    cls = pd.read_csv(tables / "probe_classification_results.csv")
    two = cls[cls["task"] == "regime_2class"]
    baseline = float(two["majority_baseline"].max())
    check("classification: bubbles-vs-turbulent beats the majority baseline by >= 0.15",
          float(two["accuracy"].max()) - baseline >= 0.15,
          f"{two['accuracy'].max():.3f} vs {baseline:.3f}")

    three = cls[cls["task"] == "regime_3class"]
    check("classification: transition class stays poorly separable (negative result)",
          float(three["macro_f1"].max()) < 0.6,
          f"best macro-F1 = {three['macro_f1'].max():.3f}")

    nine = cls[cls["task"] == "velocity_9class"]
    check("classification: 9-class velocity accuracy stays low (negative result)",
          float(nine["accuracy"].max()) < 0.4,
          f"best = {nine['accuracy'].max():.3f}")
    check("classification: but errors are strongly ordinal",
          float(nine["within_one_step"].max()) >= 0.55,
          f"within +/-1 step = {nine['within_one_step'].max():.3f}")

    # --- 13. robustness over ten seeds ------------------------------------
    robust = tables / "robustness_classification_summary.csv"
    if robust.exists():
        r = pd.read_csv(robust)
        two_r = r[r["task"] == "regime_2class"]
        within = two_r[two_r["protocol"] == "replicate"]["accuracy_mean"].max()
        across = two_r[two_r["protocol"] == "radial"]["accuracy_mean"].max()
        check("robustness: 2-class accuracy within conditions >= 0.80 over 10 seeds",
              within >= 0.80, f"mean = {within:.3f}")
        check("robustness: 2-class accuracy on an unseen radial position >= 0.75",
              across >= 0.75, f"mean = {across:.3f}")
        spread = float(two_r["accuracy_std"].max())
        check("robustness: seed-to-seed spread is small (< 0.03)", spread < 0.03,
              f"max std = {spread:.3f}")
        three_r = r[r["task"] == "regime_3class"]
        check("robustness: transition class still poorly separable over 10 seeds",
              float(three_r["macro_f1_mean"].max()) < 0.6,
              f"macro-F1 = {three_r['macro_f1_mean'].max():.3f}")

    regression = tables / "robustness_regression_summary.csv"
    if regression.exists():
        reg = pd.read_csv(regression)
        best_r2 = float(reg["r2_mean"].max())
        best_mae = float(reg["mae_mean"].min())
        check("regression: gas velocity recovered with R2 >= 0.5", best_r2 >= 0.5,
              f"best R2 = {best_r2:.3f}")
        check("regression: MAE <= 0.15 m/s", best_mae <= 0.15, f"MAE = {best_mae:.3f} m/s")
        check("regression: the mean-prediction baseline scores 0",
              abs(float(reg["r2_baseline"].mean())) < 1e-6,
              f"baseline R2 = {reg['r2_baseline'].mean():.2e}")

    # --- 14. bubble coalescence -------------------------------------------
    slopes = tables / "bubble_slopes.csv"
    if slopes.exists():
        sl = pd.read_csv(slopes)
        low = sl[sl["bed_height_cm"] == 60]
        check("bubble: passage rate falls with velocity at 60 cm at every position",
              bool((low["slope"] < 0).all()),
              f"{(low['slope'] < 0).sum()}/{len(low)} negative")
        tests = tables / "bubble_pooled_slope_tests.json"
        if tests.exists():
            import json as _json
            pooled = _json.loads(tests.read_text())
            low_test = next(t for t in pooled if t["height_cm"] == 60)
            high_test = next(t for t in pooled if t["height_cm"] == 100)
            check("bubble: 60 cm slope significant and negative (p < 0.01)",
                  low_test["slope"] < 0 and low_test["p_value"] < 0.01,
                  f"slope = {low_test['slope']:+.2f}, p = {low_test['p_value']:.2e}")
            check("bubble: 100 cm slope significant and positive (opposite sign)",
                  high_test["slope"] > 0 and high_test["p_value"] < 0.01,
                  f"slope = {high_test['slope']:+.2f}, p = {high_test['p_value']:.2e}")
        pooled_fit = tables / "bubble_pooled_fit.json"
        if pooled_fit.exists():
            import json as _json
            fit = _json.loads(pooled_fit.read_text())
            check("bubble: pooling the heights destroys the signal (negative result)",
                  fit["two_segment_r2"] < 0,
                  f"pooled R2 = {fit['two_segment_r2']:.2f}")

    # --- 15. triangulation -------------------------------------------------
    triang = tables / "triangulation_breakpoints.csv"
    if triang.exists():
        tri = pd.read_csv(triang)
        check("triangulation: four observables give four breakpoints",
              len(tri) == 4, f"got {len(tri)}")
        check("triangulation: all breakpoints inside 0.45-0.75 m/s",
              bool(((tri["breakpoint"] >= 0.45) & (tri["breakpoint"] <= 0.75)).all()),
              f"range {tri['breakpoint'].min():.2f}-{tri['breakpoint'].max():.2f}")
        lower = float(tri["ci_low"].max())
        upper = float(tri["ci_high"].min())
        check("triangulation: the four intervals share a common overlap",
              lower < upper, f"overlap {lower:.2f}-{upper:.2f} m/s")

    # --- 16. U_mf identifiability and the pressure unit --------------------
    umf_profile = tables / "umf_identifiability_profile.csv"
    if umf_profile.exists():
        prof = pd.read_csv(umf_profile)
        admissible = prof[prof["identifiable"]]
        check("U_mf: not identifiable (profile interval spans the whole grid)",
              len(admissible) >= 0.9 * len(prof),
              f"{len(admissible)}/{len(prof)} grid points admissible")
        check("U_mf: the admissible set reaches well below the measured range",
              float(admissible["umf"].min()) <= 0.05,
              f"lowest admissible U_mf = {admissible['umf'].min():.2f} m/s")
    plateau = tables / "umf_plateau_statistics.json"
    if plateau.exists():
        import json as _json
        st = _json.loads(plateau.read_text())
        check("U_mf: total bed pressure drop changes by < 5 % over the range",
              abs(st["relative_change_across_range"]) < 0.05,
              f"{st['relative_change_across_range'] * 100:+.2f} %")
        check("U_mf: a packed bed would have changed far more",
              st["packed_bed_expectation_ratio"] >= 5,
              f"ratio = {st['packed_bed_expectation_ratio']:.0f}x")
    unit = tables / "umf_unit_crosscheck.json"
    if unit.exists():
        import json as _json
        u = _json.loads(unit.read_text())
        check("unit cross-check: implied bulk density inside the expected FCC range",
              u["within_expected_range"], f"{u['implied_bulk_density']:.0f} kg/m3")

    # --- 10. recorded surrogate reference ----------------------------------
    reference = tables / "surrogate_recorded_reference.csv"
    check("surrogate reference preserved with provenance", reference.exists(),
          str(reference))
    if reference.exists():
        ref = pd.read_csv(reference)
        best = ref[(ref["what"] == "Wide-ANN surrogate, 5-fold CV") &
                   (ref["metric"] == "mean validation R2")]
        check("surrogate: recorded CV R2 = 0.9455",
              bool(np.isclose(float(best["value"].iloc[0]), 0.9455, atol=1e-4)),
              f"value = {best['value'].iloc[0]}")

    # --- report -------------------------------------------------------------
    print("=" * 78)
    print("CLAIM VERIFICATION")
    print("=" * 78)
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {detail}")
    print("=" * 78)
    print(f"{len(CHECKS) - len(FAILURES)}/{len(CHECKS)} claims verified")
    if FAILURES:
        print("\nMISMATCHES:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 2
    print("All README claims are consistent with the generated results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
