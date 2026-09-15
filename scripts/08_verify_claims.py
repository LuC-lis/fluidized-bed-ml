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

    # --- 11-12. classification ---------------------------------------------
    cls = pd.read_csv(tables / "probe_classification_results.csv")
    two = cls[cls["task"] == "regime_2class"]
    best_acc = float(two["accuracy"].max())
    baseline = float(two["majority_baseline"].max())
    radial = two[two["protocol"] == "radial"]
    check("classification: bubbling-vs-turbulent accuracy >= 0.80", best_acc >= 0.80,
          f"best = {best_acc:.3f}")
    check("classification: beats the majority baseline by >= 0.15",
          best_acc - baseline >= 0.15, f"{best_acc:.3f} vs {baseline:.3f}")
    check("classification: holds up on an unseen radial position",
          bool((radial["accuracy"] >= 0.78).any()),
          f"radial best = {radial['accuracy'].max():.3f}")

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
