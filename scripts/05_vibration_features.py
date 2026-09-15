#!/usr/bin/env python
"""
05 -- Vibration feature extraction and velocity-sweep trends.

Computes the db4 12-level band energies and time-domain descriptors for every
sensor / axis / condition, then ranks the conditions by vibration intensity and
tests which features actually track the gas-velocity sweep.

Only two of the ten condition folders carry a velocity label, so the ordering is
inferred and labelled as such in every output.

Outputs
-------
results/tables/vibration_features.csv
results/tables/vibration_intensity_by_condition.csv
results/tables/vibration_band_ratios.csv
results/tables/vibration_monotonicity.csv
results/figures/vibration_intensity_rank.png
results/figures/vibration_band_ratios.png
results/figures/vibration_spectrum_example.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_vibration as vib
from fbfml import data_vibration, paths, viz

CACHE = paths.TABLES_DIR / "vibration_features.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the cached feature table")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="use only the first N samples of each record (faster, for smoke tests)")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if CACHE.exists() and not args.force and args.max_samples is None:
        print(f"reusing cached feature table {CACHE} (use --force to recompute)")
        features = pd.read_csv(CACHE)
    else:
        print(f"extracting features from {paths.vibration_dir()} ...")
        features = vib.condition_feature_table(max_samples=args.max_samples)
        if args.max_samples is None:
            features.to_csv(CACHE, index=False)

    ok = features[features["status"] == "ok"]
    print(f"  conditions        : {features['condition'].nunique()}")
    print(f"  feature rows      : {len(ok)}  (sensor x axis x condition)")
    print(f"  samples per record: {ok['n_samples'].min():,} - {ok['n_samples'].max():,}")
    viz.write_table(features, "vibration_features")

    intensity = vib.intensity_vs_velocity(features)
    viz.write_table(intensity, "vibration_intensity_by_condition")

    validation = data_vibration.validate_velocity_rule(data_vibration.velocity_assignments())
    print("\n=== validation of the timestamp -> velocity rule ===")
    for row in validation["details"]:
        print(f"  {row['condition']:24s} label={row['label']:.1f}  "
              f"assigned={row['assigned']:.1f}  {'OK' if row['ok'] else 'MISMATCH'}")
    print(f"  rule consistent with all labelled folders: {validation['rule_consistent']}")

    sensor_check = vib.per_sensor_velocity_check(features)
    viz.write_table(sensor_check, "vibration_per_sensor_check")
    print("\n=== per-sensor physical cross-check (intensity vs velocity) ===")
    print(f"  median Spearman rho : {sensor_check['spearman_rho'].median():.3f}")
    print(f"  sensors with rho>0.8 : {int((sensor_check['spearman_rho'] > 0.8).sum())}/{len(sensor_check)}")

    ratios = vib.band_ratio_table(features)
    viz.write_table(ratios, "vibration_band_ratios")

    monotonicity = pd.DataFrame(
        [
            dict(feature="stat_std", spearman_rho=vib.monotonicity_score(ok, "stat_std")),
            dict(feature="intensity", spearman_rho=vib.monotonicity_score(ok, "intensity")),
            dict(feature="dominant_freq_hz", spearman_rho=vib.monotonicity_score(ok, "dominant_freq_hz")),
            *[
                dict(feature=f"Level{i}_ratio", spearman_rho=vib.monotonicity_score(ok, f"Level{i}_ratio"))
                for i in range(1, 13)
            ],
        ]
    ).sort_values("spearman_rho", ascending=False)
    viz.write_table(monotonicity, "vibration_monotonicity")
    print("\n=== which features track the (inferred) velocity sweep? ===")
    print(monotonicity.head(6).to_string(index=False))

    _figure_intensity(intensity)
    _figure_ratios(ratios)
    _example_spectrum()

    print("\nDone. See results/figures.")
    return 0


def _figure_intensity(intensity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = intensity["velocity_assigned"].to_numpy()
    y = intensity["intensity"].to_numpy()
    ax.plot(x, y, "-o", color=viz.PALETTE[0], lw=1.8, ms=6)
    for _, row in intensity.iterrows():
        colour = viz.PALETTE[2] if pd.notna(row["velocity_label"]) else "#555555"
        ax.annotate("labelled" if pd.notna(row["velocity_label"]) else "inferred",
                    (row["velocity_assigned"], row["intensity"]),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7, color=colour)
    ax.set_xlabel("gas velocity $U_g$ (m/s)")
    ax.set_ylabel("mean wall-vibration std across sensors (g)")
    ax.set_title("Wall-vibration intensity grows with gas velocity\n"
                 "velocity recovered from the acquisition timestamp; 2 of 9 folders carry a label")
    viz.save(fig, "vibration_intensity_rank")


def _figure_ratios(ratios: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    cols = [f"Level{i}_ratio" for i in range(1, 13)]
    half = 6
    for ax, group, name in ((axes[0], cols[:half], "coarse scales (Level1-6)"),
                            (axes[1], cols[half:], "fine scales (Level7-12)")):
        cmap = plt.get_cmap("viridis")
        for i, col in enumerate(group):
            ax.plot(ratios["velocity_assigned"], ratios[col], "-o", ms=3, lw=1.3,
                    color=cmap(i / max(1, len(group) - 1)), label=col.replace("_ratio", ""))
        ax.set_xlabel("$U_g$ (m/s)")
        ax.set_ylabel("median band-energy ratio (-)")
        ax.set_title(name)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Band-energy distribution shifts towards the fine scales as the bed "
                 "becomes more vigorous", fontweight="bold")
    viz.save(fig, "vibration_band_ratios")


def _example_spectrum() -> None:
    """PSD of one sensor at the lowest and highest inferred condition."""
    conditions = data_vibration.list_conditions()
    if len(conditions) < 2:
        return
    picks = [conditions[0], conditions[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, condition in zip(axes, picks):
        records = data_vibration.load_condition(condition)
        rec = records.get(1)
        if rec is None or not rec.is_valid:
            ax.set_axis_off()
            continue
        fs = float(rec.data_rate_hz or 1.0)
        for axis_name, color in zip(data_vibration.AXES, viz.PALETTE):
            x = rec.axis(axis_name)
            x = x - x.mean()
            freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
            psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            psd = psd / psd.max()
            ax.semilogy(freqs[1:], psd[1:], lw=0.7, color=color, label=axis_name)
        ax.set_xlim(0, 200)
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(condition.name)
    axes[0].set_ylabel("normalised PSD (sensor 1)")
    axes[0].legend(fontsize=8)
    viz.save(fig, "vibration_spectrum_example")


if __name__ == "__main__":
    raise SystemExit(main())
