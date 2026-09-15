#!/usr/bin/env python
"""
03 -- Regression test of the wavelet feature extractor against the group's tables.

Verification of the wavelet feature definition.

``src/fbfml/features_wavelet.py`` documents the definition the models were trained
on (db4, 12 levels, ``mode="symmetric"``, DC removed).  This script recomputes
every band energy from the raw pressure logs and compares it against the stored
tables ``Pressure_Energy/energy_0.X.xlsx``.

An exact match is what makes the rest of the pipeline checkable: it shows that the
model inputs are reproducible from the raw signals alone, and it states the
definition in executable form.

Outputs
-------
results/tables/wavelet_validation.csv
results/figures/wavelet_validation.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import data_pressure, features_wavelet, paths, viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="maximum relative error accepted (default 1e-6)")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    records = data_pressure.load_all_pressure()
    rows = []
    for velocity, rec in sorted(records.items()):
        vendor_path = paths.pressure_dir() / f"energy_{velocity:g}.xlsx"
        if not vendor_path.exists():
            print(f"  ! no stored table for {velocity} m/s, skipping comparison")
            continue
        vendor = data_pressure.load_vendor_energy_table(vendor_path)
        for i, channel in enumerate(data_pressure.CHANNEL_COLUMNS):
            signal = rec.signals[channel].to_numpy()
            mine = features_wavelet.band_energies(signal)
            theirs = vendor.iloc[i, :12].to_numpy(dtype=float)
            abs_err = np.max(np.abs(mine - theirs))
            denom = np.maximum(np.abs(theirs), 1e-30)
            rel_err = float(np.max(np.abs(mine - theirs) / denom))
            rows.append(
                dict(
                    velocity=velocity,
                    channel=data_pressure.channel_key(channel),
                    n_samples=signal.size,
                    max_abs_error=abs_err,
                    max_rel_error=rel_err,
                    total_energy_mine=float(mine.sum()),
                    total_energy_vendor=float(vendor.iloc[i, 12]),
                    matches=bool(rel_err <= args.tolerance),
                )
            )

    table = pd.DataFrame(rows)
    viz.write_table(table, "wavelet_validation")
    print("\n=== wavelet feature validation (recomputed vs stored tables) ===")
    print(f"  comparisons      : {len(table)}")
    print(f"  exact matches    : {int(table['matches'].sum())} / {len(table)}")
    print(f"  worst rel. error : {table['max_rel_error'].max():.3e}")
    print(f"  worst abs. error : {table['max_abs_error'].max():.3e}")
    print(f"  definition       : {features_wavelet.WaveletConfig().as_dict()}")

    if not bool(table["matches"].all()):
        print("\n!! MISMATCH -- the feature definition has drifted from the stored tables")
        print(table[~table["matches"]].to_string(index=False))
        return 2

    _figure(table)
    print("\nDone: the feature pipeline reproduces the group's stored features exactly.")
    return 0


def _figure(table) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    pivot = table.pivot_table(index="channel", columns="velocity", values="max_rel_error")
    order = [data_pressure.channel_key(c) for c in data_pressure.CHANNELS_LOW_TO_HIGH]
    pivot = pivot.reindex(order)
    im = ax.imshow(np.log10(pivot.to_numpy() + 1e-30), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("pressure tap")
    ax.set_title("Relative error of the recomputed db4 12-level band energies\n"
                 "(all cells at machine precision $\\Rightarrow$ exact reproduction)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="$\\log_{10}$(max relative error)")
    viz.save(fig, "wavelet_validation")


if __name__ == "__main__":
    raise SystemExit(main())
