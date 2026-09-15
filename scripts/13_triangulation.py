#!/usr/bin/env python
"""
13 -- Triangulation: do three independent instruments see the same transition?

The bed carries four different observables, and each is fitted with the *same*
two-segment breakpoint estimator (`fbfml.analysis_change_point`), so the
comparison is apples-to-apples:

======================  ====================================================
curve                   instrument
======================  ====================================================
top-tap mean pressure   differential pressure, 110-130 cm
top/bottom std ratio    differential pressure, scale-free
wall-vibration intensity tri-axial accelerometers
turbulent probability   optical probe, out-of-fold classifier output
======================  ====================================================

The result is a forest plot: four breakpoint estimates with bootstrap intervals.
Where they agree, the transition is corroborated by four physically unrelated
measurements; where they disagree, the disagreement is reported.

Outputs
-------
results/tables/triangulation_curves.csv
results/tables/triangulation_breakpoints.csv
results/figures/triangulation_curves.png
results/figures/triangulation_forest.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_change_point as cp
from fbfml import analysis_regime as reg
from fbfml import data_pressure, paths, viz
from fbfml.threading import single_threaded

TRANSITION_PATH = paths.TABLES_DIR / "pressure_regime_transition.json"
TRANSITION = json.loads(TRANSITION_PATH.read_text()) if TRANSITION_PATH.exists() else None
BUBBLE_CROSSING = paths.TABLES_DIR / "bubble_crossing.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    curves = build_curves()
    viz.write_table(curves, "triangulation_curves")

    rows = []
    for name, group in curves.groupby("curve", sort=False):
        group = group.sort_values("velocity")
        fit = cp.fit_breakpoint(group["velocity"], group["value"], min_segment=3)
        ci = cp.bootstrap_breakpoint_ci(group["velocity"], group["value"], n_boot=800)
        rows.append(dict(
            curve=name, instrument=group["instrument"].iloc[0],
            breakpoint=ci["x_break"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
            slope_low=fit.slope_low, slope_high=fit.slope_high, r_squared=fit.r2,
            exchange_rate=fit.slope_high - fit.slope_low,
        ))
    breakpoints = pd.DataFrame(rows)
    viz.write_table(breakpoints, "triangulation_breakpoints")

    print("=== breakpoint estimated from four independent observables ===")
    for _, row in breakpoints.iterrows():
        print(f"  {row['curve']:26s} {row['instrument']:22s} "
              f"{row['breakpoint']:.2f} m/s  (95% CI {row['ci_low']:.2f} - {row['ci_high']:.2f})")

    values = breakpoints["breakpoint"].to_numpy()
    print(f"\n  spread of the four estimates : {values.min():.2f} - {values.max():.2f} m/s")
    if TRANSITION:
        band = TRANSITION["recommended_band"]
        inside = int(((values >= band[0] - 0.05) & (values <= band[1] + 0.05)).sum())
        print(f"  pressure-only band           : {band[0]:.2f} - {band[1]:.2f} m/s")
        print(f"  estimates within +/-0.05 of it: {inside}/{len(values)}")

    # --- the second, earlier event -----------------------------------------
    if BUBBLE_CROSSING.exists():
        crossing = json.loads(BUBBLE_CROSSING.read_text())
        print("\n=== a second, earlier event (different physics) ===")
        print(f"  bubble-rate height crossing  : {crossing['velocity']:.2f} m/s "
              f"(95% CI {crossing['ci_low']:.2f} - {crossing['ci_high']:.2f})")
        print("  -> bed expansion, not the regime transition: the two must not be conflated")

    _figure_curves(curves, breakpoints)
    _figure_forest(breakpoints)

    print("\nDone. See results/figures.")
    return 0


def build_curves() -> pd.DataFrame:
    """Assemble the four observables onto a common long-format table."""
    frames = []

    stats = reg.channel_statistics()
    top = (
        stats[stats["channel"] == data_pressure.CHANNELS_LOW_TO_HIGH[-1]]
        .sort_values("velocity")[["velocity", "mean"]]
        .rename(columns={"mean": "value"})
    )
    frames.append(top.assign(curve="top-tap mean", instrument="pressure"))

    contrast = reg.axial_contrast(stats).sort_values("velocity")
    frames.append(contrast[["velocity", "std_top_over_bottom"]]
                  .rename(columns={"std_top_over_bottom": "value"})
                  .assign(curve="top/bottom std ratio", instrument="pressure"))

    vib_path = paths.TABLES_DIR / "vibration_features.csv"
    if vib_path.exists():
        vib = pd.read_csv(vib_path)
        vib = vib[vib["status"] == "ok"].dropna(subset=["velocity_assigned"])
        intensity = (vib.groupby("velocity_assigned")["stat_std"].mean()
                     .rename("value").reset_index()
                     .rename(columns={"velocity_assigned": "velocity"}))
        frames.append(intensity.assign(curve="wall vibration intensity", instrument="vibration"))

    probe_path = paths.TABLES_DIR / "probe_features.csv"
    if probe_path.exists():
        probability = turbulent_probability(pd.read_csv(probe_path))
        frames.append(probability.assign(curve="turbulent probability", instrument="probe"))

    return pd.concat(frames, ignore_index=True)


def turbulent_probability(table: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold P(turbulent) per record, averaged per gas velocity.

    Using out-of-fold predictions is essential: an in-sample probability would be
    ~0 or ~1 for every record and the curve would carry no information.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    feature_columns = [c for c in table.columns
                       if c not in {"file", "bed_height_cm", "radial_position", "velocity",
                                    "velocity_label", "n_samples", "error", "regime"}]
    X = table[feature_columns].to_numpy(dtype=float)
    binary = table[table["velocity"] != 0.6].copy()
    Xb = binary[feature_columns].to_numpy(dtype=float)
    yb = (binary["velocity"] > 0.6).astype(int).to_numpy()
    groups = binary.groupby(["bed_height_cm", "radial_position"]).ngroup().to_numpy()

    probability = np.full(len(binary), np.nan)
    splitter = GroupKFold(n_splits=6)
    for train, test in splitter.split(Xb, yb, groups):
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        with single_threaded():
            model.fit(Xb[train], yb[train])
        probability[test] = model.predict_proba(Xb[test])[:, 1]
    binary["p_turbulent"] = probability
    return (
        binary.dropna(subset=["p_turbulent"])
        .groupby("velocity")["p_turbulent"].mean()
        .rename("value").reset_index()
    )


def _figure_curves(curves: pd.DataFrame, breakpoints: pd.DataFrame) -> None:
    names = list(curves["curve"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 3.9), squeeze=False)
    for ax, name in zip(axes[0], names):
        group = curves[curves["curve"] == name].sort_values("velocity")
        x = group["velocity"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        row = breakpoints[breakpoints["curve"] == name].iloc[0]
        fit = cp.fit_breakpoint(x, y, min_segment=3)
        xs = np.linspace(x.min(), x.max(), 200)
        pred = np.where(xs <= fit.x_break,
                        fit.slope_low * xs + fit.intercept_low,
                        fit.slope_high * xs + fit.intercept_high)
        ax.plot(x, y, "o", ms=5, color=viz.PALETTE[0])
        ax.plot(xs, pred, "-", color=viz.PALETTE[1], lw=1.8)
        ax.axvline(row["breakpoint"], ls="--", color=viz.PALETTE[1], lw=1.2)
        ax.axvspan(row["ci_low"], row["ci_high"], color=viz.PALETTE[1], alpha=0.15)
        if TRANSITION:
            ax.axvspan(*TRANSITION["recommended_band"], color="grey", alpha=0.15)
        ax.set_title(f"{name}\n$x_b$ = {row['breakpoint']:.2f} m/s "
                     f"({row['ci_low']:.2f}-{row['ci_high']:.2f})", fontsize=9)
        ax.set_xlabel("$U_g$ (m/s)")
    axes[0][0].set_ylabel("normalised observable")
    fig.suptitle("Same breakpoint estimator applied to four physically unrelated observables",
                 fontweight="bold")
    viz.save(fig, "triangulation_curves")


def _figure_forest(breakpoints: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    order = breakpoints.sort_values("breakpoint")
    y = np.arange(len(order))
    ax.errorbar(order["breakpoint"], y,
                xerr=np.vstack([order["breakpoint"] - order["ci_low"],
                                order["ci_high"] - order["breakpoint"]]),
                fmt="o", ms=8, capsize=4, lw=1.6, color=viz.PALETTE[0])
    ax.set_yticks(y, [f"{r['curve']}\n({r['instrument']})" for _, r in order.iterrows()],
                  fontsize=8)
    if TRANSITION:
        ax.axvspan(*TRANSITION["recommended_band"], color="grey", alpha=0.25,
                   label=f"pressure-only band {TRANSITION['recommended_band'][0]:.2f}-"
                         f"{TRANSITION['recommended_band'][1]:.2f} m/s")
    if BUBBLE_CROSSING.exists():
        crossing = json.loads(BUBBLE_CROSSING.read_text())
        ax.axvline(crossing["velocity"], ls=":", color=viz.PALETTE[1], lw=1.8,
                   label=f"bed-expansion event {crossing['velocity']:.2f} m/s")
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_title("Triangulation: where each observable sees the change")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    viz.save(fig, "triangulation_forest")


if __name__ == "__main__":
    raise SystemExit(main())
