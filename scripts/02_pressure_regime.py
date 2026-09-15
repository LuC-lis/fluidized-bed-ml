#!/usr/bin/env python
"""
02 -- Pressure analysis and flow-regime transition.

Reproduces the central experimental claim of the project: the differential
pressure measured at six heights shows a clear bubbling -> turbulent transition
in the 0.5-0.7 m/s band, visible both in the fluctuation intensity ratio and in
the steep rise of the topmost tap.

Outputs
-------
results/tables/pressure_channel_statistics.csv
results/tables/pressure_fluctuation_summary.csv
results/tables/pressure_axial_contrast.csv
results/tables/pressure_wavelet_energy.csv
results/tables/pressure_regime_transition.json
results/figures/pressure_mean_vs_velocity.png
results/figures/pressure_std_vs_velocity.png
results/figures/pressure_axial_profile.png
results/figures/pressure_spectra.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_regime as reg
from fbfml import data_pressure, features_wavelet, paths, viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-spectra", action="store_true", help="skip the PSD figure")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    records = data_pressure.load_all_pressure()
    if not records:
        print(f"no pressure files found in {paths.pressure_dir()}")
        return 1
    print(f"loaded {len(records)} pressure records from {paths.pressure_dir()}")

    table = reg.channel_statistics(records)
    viz.write_table(table, "pressure_channel_statistics")
    summary = reg.fluctuation_summary(table)
    viz.write_table(summary, "pressure_fluctuation_summary")
    contrast = reg.axial_contrast(table)
    viz.write_table(contrast, "pressure_axial_contrast")

    transition = reg.detect_transition_velocity(table)
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in transition.items()}
    (paths.TABLES_DIR / "pressure_regime_transition.json").write_text(json.dumps(payload, indent=2))
    print("\n=== transition-velocity estimates (m/s) ===")
    for key in ("u_top_tap_break", "u_axial_contrast", "u_low_tap_decline"):
        print(f"  {key:22s} = {transition[key]:.2f}")
    print(f"  {'recommended_band':22s} = {transition['recommended_band'][0]:.2f}"
          f" - {transition['recommended_band'][1]:.2f}")
    print(f"  fluctuation_peak_interior = {transition['fluctuation_peak_interior']}")

    # --- wavelet feature table (the definition the models consumed) ---------
    rows = []
    for velocity, rec in sorted(records.items()):
        for channel in rec.signals.columns:
            row = features_wavelet.feature_row(rec.signals[channel].to_numpy())
            row.update(dict(velocity=velocity, channel=data_pressure.channel_key(channel)))
            rows.append(row)
    energy = pd.DataFrame(rows)
    viz.write_table(energy, "pressure_wavelet_energy")
    print(f"\nwrote wavelet feature table with shape {energy.shape}")

    # --- figures ------------------------------------------------------------
    _figure_mean(table, transition)
    _figure_std(table, transition)
    _figure_axial(table)
    if not args.no_spectra:
        _figure_spectra(records)

    print("\nDone. See results/figures.")
    return 0


def _series(table, channel):
    sub = table[table["channel"] == channel].sort_values("velocity")
    return sub["velocity"].to_numpy(), sub


def _figure_mean(table, transition) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for color, channel in zip(viz.TAP_COLORS, data_pressure.CHANNELS_LOW_TO_HIGH):
        velocity, sub = _series(table, channel)
        ax.plot(velocity, sub["mean"], "-o", ms=4, lw=1.6, color=color,
                label=data_pressure.channel_label(channel))
    band = transition["recommended_band"]
    ax.axvspan(*band, color="grey", alpha=0.18, zorder=0)
    ax.annotate(f"transition band\n{band[0]:.1f}-{band[1]:.1f} m/s",
                xy=(np.mean(band), ax.get_ylim()[1]),
                xytext=(0, -4), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="#333333")
    ax.set_xlabel("superficial gas velocity $U_g$ (m/s)")
    ax.set_ylabel("time-averaged differential pressure (a.u.)")
    ax.set_title("Mean differential pressure at six bed heights")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    viz.save(fig, "pressure_mean_vs_velocity")


def _figure_std(table, transition) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for color, channel in zip(viz.TAP_COLORS, data_pressure.CHANNELS_LOW_TO_HIGH):
        velocity, sub = _series(table, channel)
        axes[0].plot(velocity, sub["std"], "-o", ms=4, lw=1.6, color=color,
                     label=data_pressure.channel_label(channel))
    axes[0].axvspan(*transition["recommended_band"], color="grey", alpha=0.18, zorder=0)
    axes[0].set_xlabel("$U_g$ (m/s)")
    axes[0].set_ylabel("std of differential pressure (a.u.)")
    axes[0].set_title("Fluctuation intensity vs height")
    axes[0].legend(fontsize=8)

    summary = table.groupby("velocity")["std"].sum().reset_index()
    axes[1].plot(summary["velocity"], summary["std"], "-o", color=viz.PALETTE[1], lw=2)
    for value, style, label in (
        (transition["u_top_tap_break"], ":", "top-tap break"),
        (transition["u_axial_contrast"], "--", "axial-contrast break"),
        (transition["u_low_tap_decline"], "-.", "low-tap peak"),
    ):
        axes[1].axvline(value, ls=style, color="k", lw=1.1)
        axes[1].annotate(f"{label}\n{value:.2f} m/s", xy=(value, summary["std"].min()),
                         xytext=(2, 6), textcoords="offset points", fontsize=8, rotation=0)
    axes[1].set_xlabel("$U_g$ (m/s)")
    axes[1].set_ylabel("sum over six taps (a.u.)")
    axes[1].set_title("Total fluctuation intensity (monotone: no interior peak)")
    for ax, letter in zip(axes, "ab"):
        viz.annotate_panel(ax, letter)
    viz.save(fig, "pressure_std_vs_velocity")


def _figure_axial(table) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    heights = [np.mean(data_pressure.CHANNEL_HEIGHTS_CM[c])
               for c in data_pressure.CHANNELS_LOW_TO_HIGH]
    speeds = sorted(table["velocity"].unique())
    cmap = plt.get_cmap("viridis")
    for i, velocity in enumerate(speeds):
        sub = table[table["velocity"] == velocity].set_index("channel").loc[
            data_pressure.CHANNELS_LOW_TO_HIGH
        ]
        ax.plot(sub["mean"], heights, "-o", ms=4,
                color=cmap(i / max(1, len(speeds) - 1)), label=f"{velocity:g} m/s")
    ax.set_ylabel("height above distributor (cm)")
    ax.set_xlabel("time-averaged differential pressure (a.u.)")
    ax.set_title("Axial pressure profile: the bed inventories\nthe upper section as velocity rises")
    ax.legend(ncol=2, fontsize=8)
    viz.save(fig, "pressure_axial_profile")


def _figure_spectra(records) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, velocity in zip(axes, [0.1, 0.5, 0.9]):
        rec = records[velocity]
        fs = rec.sampling_hz
        for color, channel in zip(viz.TAP_COLORS, data_pressure.CHANNELS_LOW_TO_HIGH):
            x = rec.signals[channel].to_numpy()
            x = x - x.mean()
            freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
            psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            psd = psd / psd.max()
            ax.semilogy(freqs[1:], psd[1:], lw=1.1, color=color,
                        label=data_pressure.channel_label(channel))
        ax.set_xlim(0, 40)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("normalised PSD")
        ax.set_title(f"$U_g$ = {velocity:g} m/s")
    axes[0].legend(fontsize=7)
    viz.save(fig, "pressure_spectra")


if __name__ == "__main__":
    raise SystemExit(main())
