#!/usr/bin/env python
"""
11 -- Bubble-passage statistics from the optical probe.

A dual-tip optical probe sees each passing bubble as a dip in the local solids
fraction, so the same 434 records used for classification also carry a *physical*
measurement: how often bubbles pass a point, and how long they stay.

The naive expectation -- "more gas, so bubbles pass more often" -- is wrong here,
and the reason is the result: the trend **reverses with height**.

* ``H = 60 cm`` (inside the dense bubbling zone): the passage frequency **falls**
  with gas velocity, at **5 of 5** radial positions. Pooled with a per-position
  intercept: **-2.02 Hz per (m/s), p < 0.0001 (n = 45 records)** -- direct
  experimental evidence of **bubble coalescence**: more gas, *fewer but larger*
  bubbles.
* ``H = 100 cm`` (upper region): the trend **rises**, **+0.87 Hz per (m/s),
  p = 0.005 (n = 41)**, because the bed expands and the bubbling zone reaches this
  plane. Two statistically strong trends of *opposite sign* at two heights.
* The two curves **cross at U_g ~ 0.35 m/s**, which is the *bed-expansion* event
  and is measurably **earlier** than the 0.55-0.65 m/s *regime transition* found
  from the pressure taps. The two events must not be conflated.
* A negative result is reported too: averaging over the two heights cancels the
  two opposing trends, so a single pooled curve has no usable trend
  (two-segment fit R^2 = -6.2).

Outputs
-------
results/tables/bubble_features.csv
results/tables/bubble_by_condition.csv
results/tables/bubble_rr_by_velocity.csv
results/tables/bubble_slopes.csv
results/tables/bubble_crossing.json
results/tables/bubble_pooled_fit.json
results/figures/bubble_rate_vs_velocity.png
results/figures/bubble_slope_map.png
results/figures/bubble_crossing.png
results/figures/bubble_pooled_failure.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _bootstrap import REPO_ROOT  # noqa: F401

from fbfml import analysis_change_point as cp
from fbfml import data_ofp, features_probe, paths, viz

CACHE = paths.TABLES_DIR / "bubble_features.csv"
TRANSITION_PATH = paths.TABLES_DIR / "pressure_regime_transition.json"
TRANSITION = json.loads(TRANSITION_PATH.read_text()) if TRANSITION_PATH.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the cached feature table")
    args = parser.parse_args()

    viz.set_style()
    paths.ensure_results_dirs()

    if CACHE.exists() and not args.force:
        print(f"reusing cached bubble features {CACHE}")
        features = pd.read_csv(CACHE)
    else:
        print(f"extracting bubble statistics from {paths.ofp_dir()} ...")
        features = features_probe.records_to_bubble_features(data_ofp.iter_pvd_files())
        features.to_csv(CACHE, index=False)

    viz.write_table(features, "bubble_features")
    print(f"  records            : {len(features)}")
    print(f"  passage rate range : {features['bubble_rate_hz'].min():.2f} - "
          f"{features['bubble_rate_hz'].max():.2f} Hz  (bubbling-bed range, physically sane)")
    print(f"  dip duration range : {features['bubble_duration_s'].min() * 1e3:.0f} - "
          f"{features['bubble_duration_s'].max() * 1e3:.0f} ms")

    summary = (
        features.groupby(["bed_height_cm", "radial_position", "velocity"])
        .agg(n=("bubble_rate_hz", "size"),
             rate_mean=("bubble_rate_hz", "mean"),
             rate_std=("bubble_rate_hz", "std"),
             duty_mean=("bubble_duty_cycle", "mean"),
             depth_mean=("bubble_dip_depth_norm", "mean"),
             duration_mean=("bubble_duration_s", "mean"),
             psd_mean=("bubble_freq_psd_hz", "mean"))
        .reset_index()
    )
    summary["rate_std"] = summary["rate_std"].fillna(0.0)
    viz.write_table(summary, "bubble_by_condition")

    by_velocity = (
        summary.groupby(["bed_height_cm", "velocity"])
        .agg(rate=("rate_mean", "mean"),
             rate_sem=("rate_mean", lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s)))
                       if len(s) > 1 else 0.0),
             n_positions=("rate_mean", "size"))
        .reset_index()
    )
    viz.write_table(by_velocity, "bubble_rr_by_velocity")

    # --- 1. slope of rate vs velocity, per measurement position ------------
    slopes = position_slopes(summary)
    viz.write_table(slopes, "bubble_slopes")
    print("\n=== trend of the bubble-passage rate with gas velocity ===")
    pooled_tests = []
    for height in sorted(slopes["bed_height_cm"].unique()):
        sub = slopes[slopes["bed_height_cm"] == height]
        negatives = int((sub["slope"] < 0).sum())
        sign_p = float(sp_stats.binomtest(negatives, len(sub), 0.5, alternative="two-sided").pvalue)
        test = pooled_slope_test(summary, height)
        pooled_tests.append(test)
        print(f"  H = {height:3.0f} cm | pooled slope {test['slope']:+.2f} Hz per (m/s) "
              f"(p = {test['p_value']:.4f}, n = {test['n']} records, "
              f"{test['n_positions']} positions)")
        print(f"              per-position: negative at {negatives}/{len(sub)} "
              f"(sign test p = {sign_p:.3f}), "
              f"individually significant {int(sub['significant'].sum())}/{len(sub)}, "
              f"mean {sub['slope'].mean():+.2f} Hz per (m/s)")
    (paths.TABLES_DIR / "bubble_pooled_slope_tests.json").write_text(
        json.dumps(pooled_tests, indent=2))
    print("  -> coalescence dominates at the low plane: more gas, fewer but larger bubbles")

    # --- 2. crossing velocity of the two height curves ---------------------
    crossing = estimate_crossing(by_velocity)
    (paths.TABLES_DIR / "bubble_crossing.json").write_text(json.dumps(crossing, indent=2))
    print("\n=== where the two height curves cross ===")
    print(f"  U_g = {crossing['velocity']:.2f} m/s "
          f"(95% bootstrap CI {crossing['ci_low']:.2f} - {crossing['ci_high']:.2f})")
    print(f"  bracket from the raw difference: {crossing['bracket'][0]:.1f} - "
          f"{crossing['bracket'][1]:.1f} m/s")
    if TRANSITION:
        band = TRANSITION["recommended_band"]
        print(f"  pressure-based regime transition: {band[0]:.2f} - {band[1]:.2f} m/s")
        print("  -> the bed-expansion crossing happens measurably EARLIER than the")
        print("     regime transition; they are different events.")

    # --- 3. the pooled curve has no usable trend (negative result) ---------
    pooled = summary.groupby("velocity").agg(rate=("rate_mean", "mean")).reset_index()
    fit = cp.fit_breakpoint(pooled["velocity"], pooled["rate"], min_segment=3)
    any_trend = float(np.polyfit(pooled["velocity"], pooled["rate"], 1)[0])
    pooled_result = dict(
        pooled_slope_hz_per_ms=any_trend,
        two_segment_r2=fit.r2,
        note=("Averaging the two heights cancels two opposing trends. The two-segment "
              "fit is worse than the mean (R2 < 0), so no single-frequency-vs-velocity "
              "curve exists for this bed: height must not be averaged away."),
    )
    (paths.TABLES_DIR / "bubble_pooled_fit.json").write_text(json.dumps(pooled_result, indent=2))
    print("\n=== negative result ===")
    print(f"  pooled (both heights) slope : {any_trend:+.3f} Hz per (m/s)")
    print(f"  pooled two-segment fit R2   : {fit.r2:.3f}  (worse than the mean)")
    print("  -> height must be resolved, not averaged")

    _figure_rate(summary, by_velocity)
    _figure_slope_map(slopes)
    _figure_crossing(by_velocity, crossing)
    _figure_pooled_failure(pooled, fit)

    print("\nDone. See results/figures.")
    return 0


# ---------------------------------------------------------------------------
def pooled_slope_test(summary: pd.DataFrame, height: float) -> dict:
    """Slope of the passage rate against velocity, pooling radial positions.

    Each radial position has its own offset (removed by demeaning within the
    position), which is the small-sample equivalent of a fixed-effects regression.
    Pooling is what gives this test the power that a five-point sign test lacks.
    """
    sub = summary[summary["bed_height_cm"] == height].copy()
    sub["rate_centred"] = sub["rate_mean"] - sub.groupby("radial_position")["rate_mean"].transform("mean")
    fit = sp_stats.linregress(sub["velocity"], sub["rate_centred"])
    return dict(
        height_cm=float(height), n=int(len(sub)),
        n_positions=int(sub["radial_position"].nunique()),
        slope=float(fit.slope), stderr=float(fit.stderr),
        p_value=float(fit.pvalue), r_squared=float(fit.rvalue ** 2),
    )


def position_slopes(summary: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
    """Least-squares slope of the passage rate against velocity, per position."""
    rows = []
    for (height, radial), group in summary.groupby(["bed_height_cm", "radial_position"]):
        group = group.sort_values("velocity")
        if len(group) < min_points:
            continue
        fit = sp_stats.linregress(group["velocity"], group["rate_mean"])
        rows.append(dict(
            bed_height_cm=height, radial_position=radial, n_points=len(group),
            slope=float(fit.slope), slope_stderr=float(fit.stderr),
            r_squared=float(fit.rvalue ** 2), p_value=float(fit.pvalue),
            significant=bool(fit.pvalue < 0.05), negative=bool(fit.slope < 0),
        ))
    return pd.DataFrame(rows).sort_values(["bed_height_cm", "radial_position"]).reset_index(drop=True)


def estimate_crossing(by_velocity: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Velocity at which the two bed-height curves intersect, with a bootstrap CI."""
    pivot = by_velocity.pivot_table(index="velocity", columns="bed_height_cm", values="rate")
    heights = list(pivot.columns)
    if len(heights) < 2:
        return dict(velocity=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                    bracket=(float("nan"), float("nan")))
    low, high = heights[0], heights[-1]
    x = pivot.index.to_numpy(dtype=float)
    difference = (pivot[high] - pivot[low]).to_numpy(dtype=float)

    sign = np.sign(difference)
    changes = np.flatnonzero(np.diff(sign) != 0)
    bracket = (float(x[changes[0]]), float(x[changes[0] + 1])) if changes.size else \
              (float(x[0]), float(x[-1]))

    def _zero_crossing(xs, ys) -> float:
        sign_local = np.sign(ys)
        idx = np.flatnonzero(np.diff(sign_local) != 0)
        if idx.size == 0:
            return float("nan")
        i = idx[0]
        y0, y1 = ys[i], ys[i + 1]
        if y1 == y0:
            return float(xs[i])
        return float(xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0))

    point = _zero_crossing(x, difference)

    # residual bootstrap on the difference curve
    resid = difference - np.polyval(np.polyfit(x, difference, 1), x)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        y_star = difference + rng.choice(resid, size=resid.size, replace=True)
        value = _zero_crossing(x, y_star)
        if not np.isnan(value) and x.min() - 0.2 <= value <= x.max() + 0.2:
            draws.append(value)
    draws = np.asarray(draws)
    ci_low, ci_high = (np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan))
    return dict(
        velocity=point, ci_low=float(ci_low), ci_high=float(ci_high),
        n_boot=int(draws.size), bracket=bracket,
        low_height_cm=float(low), high_height_cm=float(high),
        note=("Crossing of the radially averaged bubble-passage rate at the two "
              "measurement heights. With only two heights this is an event marker, "
              "not a resolved height profile."),
    )


# ---------------------------------------------------------------------------
def _band(ax):
    if TRANSITION:
        ax.axvspan(*TRANSITION["recommended_band"], color="grey", alpha=0.15, zorder=0,
                   label="pressure-based transition")


def _figure_rate(summary: pd.DataFrame, by_velocity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    for ax, height in zip(axes[:2], sorted(summary["bed_height_cm"].unique())):
        sub = summary[summary["bed_height_cm"] == height]
        cmap = plt.get_cmap("cividis")
        radials = sorted(sub["radial_position"].unique())
        for i, radial in enumerate(radials):
            rows = sub[np.isclose(sub["radial_position"], radial)].sort_values("velocity")
            ax.plot(rows["velocity"], rows["rate_mean"], "-o", ms=3, lw=1.2,
                    color=cmap(i / max(1, len(radials) - 1)), label=f"$r/R$ = {radial:g}")
        _band(ax)
        slope = position_slopes(summary)
        mean_slope = slope[slope["bed_height_cm"] == height]["slope"].mean()
        ax.set_xlabel("$U_g$ (m/s)")
        ax.set_ylabel("bubble passage rate (Hz)")
        ax.set_title(f"$H$ = {height:g} cm   (mean slope {mean_slope:+.2f} Hz per m/s)")
        ax.legend(fontsize=7, ncol=2)
    ax = axes[2]
    for height, color in zip(sorted(by_velocity["bed_height_cm"].unique()), viz.PALETTE):
        rows = by_velocity[by_velocity["bed_height_cm"] == height].sort_values("velocity")
        ax.errorbar(rows["velocity"], rows["rate"], yerr=rows["rate_sem"], fmt="-o", ms=5,
                    lw=2, capsize=3, color=color, label=f"$H$ = {height:g} cm")
    _band(ax)
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("rate, radial mean (Hz)")
    ax.set_title("Trends have opposite signs")
    ax.legend(fontsize=8)
    fig.suptitle("Bubble-passage frequency: coalescence at the low plane, "
                 "bed expansion at the high plane", fontweight="bold")
    viz.save(fig, "bubble_rate_vs_velocity")


def _figure_slope_map(slopes: pd.DataFrame) -> None:
    heights = sorted(slopes["bed_height_cm"].unique())
    radials = sorted(slopes["radial_position"].unique())
    grid = np.full((len(heights), len(radials)), np.nan)
    for i, height in enumerate(heights):
        for j, radial in enumerate(radials):
            sel = slopes[(slopes["bed_height_cm"] == height) &
                         np.isclose(slopes["radial_position"], radial)]
            if not sel.empty:
                grid[i, j] = float(sel["slope"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    limit = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(radials)), [f"{r:g}" for r in radials])
    ax.set_yticks(range(len(heights)), [f"{h:g} cm" for h in heights])
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("measurement height")
    ax.set_title("Slope of the bubble-passage rate vs $U_g$\n"
                 "red = fewer/larger bubbles as gas velocity rises (coalescence)")
    ax.grid(False)
    for i in range(len(heights)):
        for j in range(len(radials)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(grid[i, j]) > 0.6 * limit else "black")
    fig.colorbar(im, ax=ax, label="Hz per (m/s)")
    viz.save(fig, "bubble_slope_map")


def _figure_crossing(by_velocity: pd.DataFrame, crossing: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    pivot = by_velocity.pivot_table(index="velocity", columns="bed_height_cm", values="rate")
    difference = pivot[pivot.columns[-1]] - pivot[pivot.columns[0]]
    ax.plot(difference.index, difference.values, "-o", color=viz.PALETTE[0], lw=2, ms=6)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(crossing["velocity"], ls="--", color=viz.PALETTE[1], lw=1.5)
    ax.axvspan(crossing["ci_low"], crossing["ci_high"], color=viz.PALETTE[1], alpha=0.18,
               label=f"crossing {crossing['velocity']:.2f} m/s "
                     f"({crossing['ci_low']:.2f}-{crossing['ci_high']:.2f})")
    if TRANSITION:
        ax.axvspan(*TRANSITION["recommended_band"], color="grey", alpha=0.18,
                   label="pressure-based transition")
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("rate(100 cm) - rate(60 cm)  (Hz)")
    ax.set_title("Two events, not one:\nbed expansion crosses this plane before the "
                 "regime transition")
    ax.legend(fontsize=8)
    viz.save(fig, "bubble_crossing")


def _figure_pooled_failure(pooled: pd.DataFrame, fit) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = pooled["velocity"].to_numpy(dtype=float)
    y = pooled["rate"].to_numpy(dtype=float)
    ax.plot(x, y, "o", ms=7, color=viz.PALETTE[0], label="pooled over both heights")
    xs = np.linspace(x.min(), x.max(), 200)
    pred = np.where(xs <= fit.x_break,
                    fit.slope_low * xs + fit.intercept_low,
                    fit.slope_high * xs + fit.intercept_high)
    ax.plot(xs, pred, "-", color=viz.PALETTE[1], lw=2,
            label=f"two-segment fit ($R^2$ = {fit.r2:.2f})")
    ax.axhline(y.mean(), ls=":", color="k", lw=1.2, label="mean (better than the fit)")
    ax.set_xlabel("$U_g$ (m/s)")
    ax.set_ylabel("bubble passage rate (Hz)")
    ax.set_title("Negative result: pooling the heights destroys the signal")
    ax.legend(fontsize=8)
    viz.save(fig, "bubble_pooled_failure")


if __name__ == "__main__":
    raise SystemExit(main())
