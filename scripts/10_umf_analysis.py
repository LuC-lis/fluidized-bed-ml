#!/usr/bin/env python
"""
10 -- The U_mf question, answered honestly.

The two-line intersection method for `U_mf`, applied to this campaign. The script
does the pressure-drop bookkeeping and reports what the velocity range permits:

* the **total** bed pressure drop is flat to ~1 % across 0.1-0.9 m/s, which is the
  signature of a bed that is fluidised at every measured velocity;
* consequently the fixed-bed branch is never sampled, the two-line intersection
  is **not identifiable**, and the ±0.05 m/s claim cannot be reproduced from this
  campaign -- with the profile-likelihood curve as evidence;
* the same numbers cross-check the unknown pressure unit: if the readings are kPa,
  the implied bed-averaged bulk density is ~660 kg/m3, in the expected range for
  an FCC bed.

Outputs
-------
results/tables/umf_total_pressure.csv
results/tables/umf_plateau_statistics.json
results/tables/umf_identifiability_profile.csv
results/tables/umf_unit_crosscheck.json
results/figures/umf_total_pressure.png
results/figures/umf_cumulative_profile.png
results/figures/umf_identifiability.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_regime as reg
from fbfml import analysis_umf as umf
from fbfml import data_pressure, paths, viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    table = reg.channel_statistics()
    total = umf.total_and_cumulative(table)
    viz.write_table(total, "umf_total_pressure")

    stats = umf.plateau_statistics(total)
    (paths.TABLES_DIR / "umf_plateau_statistics.json").write_text(json.dumps(stats, indent=2))
    print("=== is the total bed pressure drop flat? ===")
    print(f"  mean                    : {stats['mean']:.3f}")
    print(f"  relative spread (std)   : {stats['relative_spread'] * 100:.2f} %")
    print(f"  max deviation from mean : {stats['max_relative_deviation'] * 100:.2f} %")
    print(f"  slope                   : {stats['slope_per_ms']:+.3f} per (m/s)  "
          f"95% CI [{stats['slope_ci_low']:+.3f}, {stats['slope_ci_high']:+.3f}]")
    print(f"  change over 0.1->0.9 m/s: {stats['relative_change_across_range'] * 100:+.2f} %  "
          f"95% CI [{stats['relative_change_ci_low'] * 100:+.2f}, "
          f"{stats['relative_change_ci_high'] * 100:+.2f}] %")
    print(f"  a packed bed would change by a factor of "
          f"{stats['packed_bed_expectation_ratio']:.0f} "
          f"({(stats['packed_bed_expectation_ratio'] - 1) * 100:.0f} %)")
    print("  -> the total is pinned to the bed weight while the individual taps")
    print("     swing by up to 47x: the bed is fully fluidised at every measured")
    print("     velocity, so no fixed-bed branch was sampled.")

    profile = umf.umf_identifiability(total)
    viz.write_table(profile, "umf_identifiability_profile")
    identifiable = profile[profile["identifiable"]]
    print("\n=== U_mf identifiability (profile likelihood) ===")
    if identifiable.empty:
        print("  no admissible U_mf: the parameter is not identifiable at all")
    else:
        print(f"  95% profile-likelihood interval: {identifiable['umf'].min():.2f} - "
              f"{identifiable['umf'].max():.2f} m/s   "
              f"({len(identifiable)}/{len(profile)} grid points)")
        print("  a interval this wide cannot support a +/-0.05 m/s claim")

    crosscheck = umf.implied_bulk_density(stats["mean"])
    (paths.TABLES_DIR / "umf_unit_crosscheck.json").write_text(json.dumps(crosscheck, indent=2))
    print("\n=== pressure-unit cross-check (docs/data_quality.md section 9) ===")
    print(f"  if the readings are {crosscheck['assumed_unit']}:")
    print(f"    DeltaP_total = {crosscheck['pressure_drop_pa']:.0f} Pa over H = 1.0 m")
    print(f"    implied bed-averaged bulk density = "
          f"{crosscheck['implied_bulk_density']:.0f} kg/m3")
    print(f"    expected range for an FCC bed     = {crosscheck['reference_range']} kg/m3")
    print(f"    within the expected range         = {crosscheck['within_expected_range']}")

    _figure_total(total, stats)
    _figure_cumulative(total)
    _figure_identifiability(profile, identifiable)

    print("\nDone. See results/figures.")
    return 0


def _figure_total(total: pd.DataFrame, stats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    for channel, color in zip(data_pressure.CHANNELS_LOW_TO_HIGH, viz.TAP_COLORS):
        piv = None  # placeholder to keep the loop readable
    # per-tap means (the volatile part)
    raw = reg.channel_statistics()
    for channel, color in zip(data_pressure.CHANNELS_LOW_TO_HIGH, viz.TAP_COLORS):
        sub = raw[raw["channel"] == channel].sort_values("velocity")
        ax.plot(sub["velocity"], sub["mean"], "-o", ms=3, lw=1.3, color=color,
                label=data_pressure.channel_label(channel))
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("tap mean (a.u.)")
    ax.set_title("Individual taps move a lot ...")
    ax.legend(fontsize=7)

    ax = axes[1]
    y = total["total"].to_numpy()
    ax.plot(total["velocity"], y, "-o", color=viz.PALETTE[1], lw=2, ms=6)
    ax.axhline(y.mean(), ls="--", color="k", lw=1)
    ax.fill_between(total["velocity"], y.min(), y.max(), color=viz.PALETTE[1], alpha=0.08)
    ax.set_ylim(0, y.max() * 1.15)
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("total bed pressure drop (a.u.)")
    ax.set_title(f"... but their sum moves by only "
                 f"{stats['relative_change_across_range'] * 100:.1f} % over the whole range\n"
                 f"(a packed bed would change by "
                 f"{stats['packed_bed_expectation_ratio']:.0f}x)")
    for ax_, letter in zip(axes, "ab"):
        viz.annotate_panel(ax_, letter)
    viz.save(fig, "umf_total_pressure")


def _figure_cumulative(total: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    columns = [c for c in total.columns if c.startswith("cum_to_")]
    heights = [int(c.replace("cum_to_", "").replace("cm", "")) for c in columns]
    cmap = plt.get_cmap("viridis")
    speeds = total["velocity"].to_numpy()
    for i, velocity in enumerate(speeds):
        row = total[total["velocity"] == velocity]
        values = [0.0] + [float(row[c].iloc[0]) for c in columns]
        ax.plot(values, [0] + heights, "-o", ms=3, lw=1.3,
                color=cmap(i / max(1, len(speeds) - 1)), label=f"{velocity:g} m/s")
    ax.axvline(float(total["total"].mean()), ls="--", color="k", lw=1,
               label="bed weight per unit area")
    ax.set_xlabel("cumulative pressure drop from the distributor (a.u.)")
    ax.set_ylabel("height above distributor (cm)")
    ax.set_title("Where the bed holds its solids: low taps unload, high taps fill\n"
                 "(total unchanged -- internal redistribution, not loss)")
    ax.legend(fontsize=7, ncol=2)
    viz.save(fig, "umf_cumulative_profile")


def _figure_identifiability(profile: pd.DataFrame, identifiable: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(profile["umf"], profile["excess_sse"], "-", color=viz.PALETTE[0], lw=2)
    ax.axhline(3.84, ls="--", color="k", lw=1)
    ax.annotate("95 % profile-likelihood threshold ($\\chi^2_1$ = 3.84)",
                xy=(profile["umf"].max(), 3.84), xytext=(-6, 6),
                textcoords="offset points", ha="right", fontsize=8)
    if not identifiable.empty:
        ax.axvspan(identifiable["umf"].min(), identifiable["umf"].max(),
                   color=viz.PALETTE[2], alpha=0.18)
        ax.annotate(f"admissible $U_{{mf}}$:\n{identifiable['umf'].min():.2f}-"
                    f"{identifiable['umf'].max():.2f} m/s",
                    xy=(identifiable["umf"].mean(), 3.84), xytext=(0, 22),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0.1, color=viz.PALETTE[1], lw=1.6, ls=":")
    ax.annotate("lowest measured\nvelocity", xy=(0.1, profile["excess_sse"].max() * 0.62),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=viz.PALETTE[1])
    ax.set_xlabel("assumed $U_{mf}$ (m/s)")
    ax.set_ylabel("excess residual sum of squares")
    ax.set_title("Why ±0.05 m/s is not reproducible from this campaign:\n"
                 "the fixed-bed branch was never sampled")
    viz.save(fig, "umf_identifiability")


if __name__ == "__main__":
    raise SystemExit(main())
