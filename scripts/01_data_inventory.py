#!/usr/bin/env python
"""
01 -- Data inventory and completeness audit.

Answers, reproducibly and in one command: what raw data exists, how big it is,
which conditions are missing repeats, and which recording files are damaged.

Outputs
-------
results/tables/inventory_summary.csv
results/tables/inventory_pressure.csv
results/tables/inventory_vibration_audit.csv
results/tables/inventory_ofp_coverage.csv
results/tables/inventory_ofp_gaps.csv
results/figures/inventory_ofp_coverage.png
results/figures/inventory_bytes.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401  (side effect: sys.path)

from fbfml import analysis_inventory as inv
from fbfml import paths, viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-header-scan", action="store_true",
                        help="skip reading the .pvd headers (the slow part)")
    args = parser.parse_args()

    print(paths.describe())
    viz.set_style()
    paths.ensure_results_dirs()

    # --- asset register -----------------------------------------------------
    summary = inv.full_inventory()
    viz.write_table(summary, "inventory_summary")
    print("\n=== asset register ===")
    print(summary.to_string(index=False))

    # --- pressure -----------------------------------------------------------
    pressure = inv.pressure_summary()
    viz.write_table(pressure, "inventory_pressure")
    print("\n=== pressure logs ===")
    print(pressure.to_string(index=False))

    # --- vibration health ---------------------------------------------------
    audit = inv.vibration_audit()
    viz.write_table(audit, "inventory_vibration_audit")
    bad = audit[audit["status"] != "ok"]
    print(f"\n=== vibration audit: {len(audit)} files, {len(bad)} unusable ===")
    if not bad.empty:
        print(bad.to_string(index=False))

    # --- probe campaign coverage -------------------------------------------
    coverage = inv.ofp_coverage()
    viz.write_table(coverage, "inventory_ofp_coverage")
    gaps = inv.coverage_gaps(coverage)
    viz.write_table(gaps, "inventory_ofp_gaps")
    print(f"\n=== probe coverage: {len(coverage)} conditions, {len(gaps)} with <5 repeats ===")

    # --- figures ------------------------------------------------------------
    _figure_coverage(coverage)
    _figure_bytes(summary)

    print("\nDone. See results/tables and results/figures.")
    return 0


def _figure_coverage(coverage) -> None:
    heights = sorted(coverage["bed_height_cm"].unique())
    radials = sorted(coverage["radial_position"].unique())
    velocities = sorted(coverage["velocity"].unique())
    fig, axes = plt.subplots(1, len(heights), figsize=(6.4 * len(heights), 4.4), squeeze=False)
    for ax, height in zip(axes[0], heights):
        sub = coverage[coverage["bed_height_cm"] == height]
        grid = np.zeros((len(radials), len(velocities)))
        for i, r in enumerate(radials):
            for j, u in enumerate(velocities):
                sel = sub[np.isclose(sub["radial_position"], r) & np.isclose(sub["velocity"], u)]
                grid[i, j] = 0 if sel.empty else float(sel["n_records"].iloc[0])
        im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=max(5, grid.max()), aspect="auto")
        ax.set_xticks(range(len(velocities)), [f"{v:g}" for v in velocities], rotation=0)
        ax.set_yticks(range(len(radials)), [f"{r:g}" for r in radials])
        ax.set_xlabel("gas velocity $U_g$ (m/s)")
        ax.set_ylabel("radial position $r/R$")
        ax.set_title(f"bed height $H$ = {height} cm")
        ax.grid(False)
        for i in range(len(radials)):
            for j in range(len(velocities)):
                value = int(grid[i, j])
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                        color="white" if value > 0.6 * max(5, grid.max()) else "black")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="usable repeats")
    fig.suptitle("Optical-fibre-probe campaign completeness", fontweight="bold")
    viz.save(fig, "inventory_ofp_coverage")


def _figure_bytes(summary) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = summary.sort_values("size_mb")
    bars = ax.barh(order["dataset"], order["size_mb"], color=viz.PALETTE[: len(order)])
    for rect, value in zip(bars, order["size_mb"]):
        ax.annotate(f"{value:,.0f} MB", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("size on disk (MB)")
    ax.set_title("Raw data volume per dataset")
    ax.set_xlim(0, order["size_mb"].max() * 1.18)
    viz.save(fig, "inventory_bytes")


if __name__ == "__main__":
    raise SystemExit(main())
