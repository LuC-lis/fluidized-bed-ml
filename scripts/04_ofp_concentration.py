#!/usr/bin/env python
"""
04 -- Optical-fibre-probe analysis: the local solids-fraction field.

Turns the 400+ raw ``.pvd`` records (50 kHz, dual-channel probe) into the
time-averaged, spatially resolved local solids fraction of the bed.

Design decision worth reading
-----------------------------
Absolute solids fraction requires a **probe-specific** two-parameter voltage
calibration.  The only constants recoverable from the legacy scripts belong to a
different probe/setting, and applying them here yields a physically implausible
field (a nearly flat, uniformly high 0.22-0.55).  Rather than publish a number
that cannot be defended, this script therefore reports

* **primary**: the time-averaged probe voltage (ADC counts, calibration-free) --
  monotonically related to the local solids fraction, and fully reproducible;
* **secondary**: the calibrated ``eps_s`` under the recovered constants, clearly
  flagged as *not validated* by an automatic plausibility diagnostic.

Outputs
-------
results/tables/ofp_records.csv
results/tables/ofp_conditions.csv
results/tables/ofp_core_annulus.csv
results/tables/ofp_calibration_diagnostic.json
results/figures/ofp_voltage_profiles.png
results/figures/ofp_velocity_trends.png
results/figures/ofp_contour.png
results/figures/ofp_timeseries_example.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_concentration as conc
from fbfml import data_ofp, paths, viz

CACHE = paths.TABLES_DIR / "ofp_records.csv"
PRIMARY = "counts_mean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the cached per-record table")
    parser.add_argument("--limit", type=int, default=None, help="process at most N records (debug)")
    parser.add_argument("--channel", type=int, default=1, choices=(0, 1),
                        help="probe channel column (default 1 = '1B', as in the legacy scripts)")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if CACHE.exists() and not args.force and args.limit is None:
        print(f"reusing cached per-record table {CACHE} (use --force to recompute)")
        records = pd.read_csv(CACHE)
    else:
        print(f"parsing probe records from {paths.ofp_dir()} ...")
        records = conc.process_records(channel=args.channel, limit=args.limit)
        if args.limit is None:
            records.to_csv(CACHE, index=False)

    viz.write_table(records, "ofp_records")
    ok = records.dropna(subset=["eps_mean"])
    print(f"  records parsed   : {len(records)}")
    print(f"  mean duration    : {ok['duration_s'].mean():.2f} s")
    print(f"  probe voltage    : {ok['counts_mean'].min():.0f} - {ok['counts_mean'].max():.0f} mV")

    summary = conc.aggregate_by_condition(records)
    viz.write_table(summary, "ofp_conditions")
    print(f"  conditions       : {len(summary)}")

    # calibration-free radial structure
    contrast_v = conc.core_annulus_ratio(summary, PRIMARY)
    contrast_e = conc.core_annulus_ratio(summary, "eps_mean")
    contrast = contrast_v.merge(
        contrast_e[["bed_height_cm", "velocity", "core_over_wall"]],
        on=["bed_height_cm", "velocity"], suffixes=("_voltage", "_eps"),
    )
    viz.write_table(contrast, "ofp_core_annulus")

    diagnostic = conc.calibration_plausibility(summary)
    (paths.TABLES_DIR / "ofp_calibration_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2)
    )
    print("\n=== calibration plausibility (recovered legacy constants) ===")
    for key, value in diagnostic.items():
        print(f"  {key:32s} = {value}")
    if not diagnostic["calibration_usable"]:
        print("  -> NOT USABLE for this campaign: absolute eps_s must not be quoted.")

    _figure_profiles(summary, PRIMARY,
                     "Time-averaged probe voltage (radial profiles)",
                     "probe voltage (mV)", "ofp_voltage_profiles")
    _figure_trends(summary, PRIMARY, "ofp_velocity_trends")
    _figure_contour(summary, PRIMARY, "ofp_contour")
    if diagnostic["calibration_usable"]:
        _figure_profiles(summary, "eps_mean",
                         "Local solids fraction (radial profiles)",
                         "$\\varepsilon_s$ (-)", "ofp_radial_profiles_eps")
    _example_timeseries()

    # --- quantitative statements about the field ---------------------------
    gradient = conc.radial_gradient_test(summary, PRIMARY)
    viz.write_table(gradient, "ofp_radial_gradient_test")
    budget = conc.noise_to_signal(summary, PRIMARY)
    (paths.TABLES_DIR / "ofp_noise_budget.json").write_text(json.dumps(budget, indent=2))
    n_pos = int(gradient["slope_positive"].sum())
    sign_p = float(sp_stats.binomtest(n_pos, len(gradient), 0.5, alternative="greater").pvalue)
    print("\n=== core-annulus (radial) gradient test ===")
    print(f"  conditions fitted      : {len(gradient)}")
    print(f"  positive slope         : {n_pos}/{len(gradient)}  (sign test p = {sign_p:.3f})")
    print(f"  individually sig. (5%) : {int(gradient['significant'].sum())}")
    print("=== error budget ===")
    print(f"  repeat spread  : {budget['mean_repeat_spread_mv']:.0f} mV")
    print(f"  spatial range  : {budget['spatial_range_mv']:.0f} mV")
    print(f"  noise/signal   : {budget['noise_to_signal']:.3f}")
    _figure_gradient(gradient, sign_p)

    print("\nDone. See results/figures.")
    return 0


def _figure_gradient(gradient, sign_p: float) -> None:
    if gradient.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels, values, colors = [], [], []
    for _, row in gradient.iterrows():
        labels.append(f"{row['bed_height_cm']:g} cm, {row['velocity']:g} m/s")
        values.append(row["slope_per_unit_rR"])
        colors.append(viz.PALETTE[2] if row["slope_positive"] else viz.PALETTE[1])
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("d(probe voltage) / d($r/R$)  (mV per unit $r/R$)")
    ax.set_title("Core-annulus gradient: positive slope = denser near the wall\n"
                 f"{sum(1 for v in values if v > 0)}/{len(values)} positive "
                 f"(sign test p = {sign_p:.3f})")
    ax.invert_yaxis()
    viz.save(fig, "ofp_radial_gradient")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figure_profiles(summary, value_col, title, ylabel, name) -> None:
    heights = sorted(summary["bed_height_cm"].dropna().unique())
    fig, axes = plt.subplots(1, len(heights), figsize=(6.2 * len(heights), 4.4),
                             sharey=True, squeeze=False)
    for ax, height in zip(axes[0], heights):
        sub = summary[summary["bed_height_cm"] == height]
        velocities = sorted(sub["velocity"].unique())
        cmap = plt.get_cmap("viridis")
        for i, velocity in enumerate(velocities):
            rows = sub[np.isclose(sub["velocity"], velocity)].sort_values("radial_position")
            ax.errorbar(rows["radial_position"], rows[value_col],
                        yerr=rows["eps_std_between_records"] if value_col == "eps_mean" else None,
                        fmt="-o", ms=4, lw=1.4, capsize=2,
                        color=cmap(i / max(1, len(velocities) - 1)), label=f"{velocity:g} m/s")
        ax.set_xlabel("dimensionless radial position $r/R$")
        ax.set_title(f"$H$ = {height:g} cm above distributor")
        ax.set_xlim(-0.05, 1.0)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, fontweight="bold")
    viz.save(fig, name)


def _figure_trends(summary, value_col, name) -> None:
    heights = sorted(summary["bed_height_cm"].dropna().unique())
    radials = sorted(summary["radial_position"].dropna().unique())
    fig, axes = plt.subplots(1, len(heights), figsize=(6.2 * len(heights), 4.2),
                             sharey=True, squeeze=False)
    for ax, height in zip(axes[0], heights):
        sub = summary[summary["bed_height_cm"] == height]
        cmap = plt.get_cmap("cividis")
        for i, radial in enumerate(radials):
            rows = sub[np.isclose(sub["radial_position"], radial)].sort_values("velocity")
            ax.plot(rows["velocity"], rows[value_col], "-o", ms=4, lw=1.4,
                    color=cmap(i / max(1, len(radials) - 1)), label=f"$r/R$ = {radial:g}")
        ax.set_xlabel("$U_g$ (m/s)")
        ax.set_title(f"$H$ = {height:g} cm")
    axes[0][0].set_ylabel("probe voltage (mV)")
    axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Probe voltage vs gas velocity at each radial position", fontweight="bold")
    viz.save(fig, name)


def _figure_contour(summary, value_col, name) -> None:
    heights = sorted(summary["bed_height_cm"].dropna().unique())
    velocities = sorted(summary["velocity"].dropna().unique())
    radials = sorted(summary["radial_position"].dropna().unique())
    fig, axes = plt.subplots(1, len(heights), figsize=(6.0 * len(heights), 4.3), squeeze=False)
    for ax, height in zip(axes[0], heights):
        sub = summary[summary["bed_height_cm"] == height]
        grid = np.full((len(radials), len(velocities)), np.nan)
        for i, r in enumerate(radials):
            for j, u in enumerate(velocities):
                sel = sub[np.isclose(sub["radial_position"], r) & np.isclose(sub["velocity"], u)]
                if not sel.empty:
                    grid[i, j] = float(sel[value_col].iloc[0])
        cs = ax.contourf(velocities, radials, grid, levels=14, cmap="viridis")
        ax.scatter(np.repeat(velocities, len(radials)), np.tile(radials, len(velocities)),
                   s=4, c="white", alpha=0.6)
        ax.set_xlabel("$U_g$ (m/s)")
        ax.set_ylabel("$r/R$")
        ax.set_title(f"$H$ = {height:g} cm")
        ax.grid(False)
        fig.colorbar(cs, ax=ax, label="probe voltage (mV)")
    fig.suptitle("Measured field: time-averaged optical-probe voltage "
                 "(white dots = measurement positions)", fontweight="bold")
    viz.save(fig, name)


def _example_timeseries() -> None:
    candidates = list(data_ofp.iter_pvd_files())
    if not candidates:
        return
    pick = next((p for p in candidates if "/0.5/" in str(p) or "\\0.5\\" in str(p)), candidates[0])
    rec = data_ofp.read_pvd(pick)
    eps = rec.solids_fraction(channel=1)
    t = np.arange(eps.size) / rec.fs_hz
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)
    axes[0].plot(t, rec.counts[:, 1], lw=0.4, color=viz.PALETTE[0])
    axes[0].set_ylabel("probe voltage (mV)")
    axes[0].set_title(f"raw probe record {Path(pick).name} | $f_s$ = {rec.fs_hz / 1000:.0f} kHz, "
                      f"{rec.duration_s:.2f} s, {rec.n_samples:,} samples")
    axes[1].plot(t, eps, lw=0.4, color=viz.PALETTE[1])
    axes[1].set_ylabel("$\\varepsilon_s$ (-)\n(unvalidated)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"calibrated with the recovered legacy constants "
                      f"(mean {eps.mean():.3f}, std {eps.std(ddof=1):.3f}) -- see docs/data_quality.md")
    viz.save(fig, "ofp_timeseries_example")


if __name__ == "__main__":
    raise SystemExit(main())
