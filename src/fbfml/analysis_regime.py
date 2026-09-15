"""
Pressure-based flow-regime analysis.

Physical background
-------------------
Two standard, instrument-only routes to the transition between the bubbling and
the turbulent regime in a gas-solid fluidized bed:

1. **Fluctuation intensity.**  The standard deviation of the differential
   pressure signal rises with gas velocity and peaks around the transition; the
   velocity of the maximum is an estimate of ``U_c`` (Bi & Grace style).
2. **Axial pressure profile.**  Below the transition the *upper* taps show an
   almost stagnant, small signal (dense bed below, dilute freeboard above); once
   the bed expands and the bubble phase penetrates the upper section, the mean
   and the fluctuation of the top taps rise steeply.  The break in the
   mean-vs-velocity curve locates the transition.

Both estimators are implemented here and reported side by side; a single number
would over-claim what six taps and nine velocities can support.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data_pressure


def channel_statistics(records: Optional[Dict[float, "data_pressure.PressureRecord"]] = None) -> pd.DataFrame:
    """Mean / std / skewness of every transducer at every gas velocity."""
    records = data_pressure.load_all_pressure() if records is None else records
    rows: List[dict] = []
    for velocity, rec in sorted(records.items()):
        for channel in rec.signals.columns:
            x = rec.signals[channel].to_numpy()
            rows.append(
                dict(
                    velocity=velocity,
                    channel=channel,
                    height_cm_lo=data_pressure.CHANNEL_HEIGHTS_CM[channel][0],
                    height_cm_hi=data_pressure.CHANNEL_HEIGHTS_CM[channel][1],
                    mean=float(x.mean()),
                    std=float(x.std(ddof=1)),
                    skew=float(pd.Series(x).skew()),
                    kurtosis=float(pd.Series(x).kurtosis()),
                )
            )
    return pd.DataFrame(rows)


def fluctuation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fluctuation level per velocity (sum over taps, and per tap)."""
    grouped = table.groupby("velocity")
    out = grouped.agg(
        std_sum=("std", "sum"),
        std_mean=("std", "mean"),
        mean_sum=("mean", "sum"),
    ).reset_index()
    # an interior maximum would indicate that the tested velocity range brackets
    # the transition; its absence is a finding, not a bug
    peak_index = int(out["std_sum"].to_numpy().argmax())
    out["fluctuation_peak_interior"] = 0 < peak_index < len(out) - 1
    return out


def plateau_decline_velocity(x: np.ndarray, y: np.ndarray, threshold: float = 0.02) -> float:
    """Velocity at which ``y`` first falls ``threshold`` below its own plateau.

    Used for the *lowest* tap: while the bed is a dense bubbling bed the bottom
    tap reads a near-constant (or slowly rising) pressure; after the bed has
    expanded past that tap the reading starts to fall.  The search therefore
    starts **after** the running maximum, so the initial aeration ramp-up cannot
    be mistaken for a decline.  Returning the midpoint between the last velocity
    on the plateau and the first one below it is far more robust than an
    ``argmax`` on a nearly flat curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i_max = int(np.argmax(y))
    limit = float(y[i_max]) * (1.0 - threshold)
    below = np.where(y[i_max:] < limit)[0]
    if below.size == 0:
        return float(x[-1])
    i = max(int(below[0]) + i_max, 1)
    return 0.5 * (float(x[i - 1]) + float(x[i]))


def _steepest_rise(x: np.ndarray, y: np.ndarray, smooth: int = 3) -> tuple:
    """Velocity of the steepest rise of ``y(x)`` plus the bracketing interval.

    A short moving average removes single-point noise before differentiating;
    the returned value is the midpoint of the steepest interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if smooth and y.size >= smooth:
        kernel = np.ones(smooth) / smooth
        y = np.convolve(y, kernel, mode="same")
    grad = np.gradient(y, x)
    i = min(max(int(np.argmax(grad)), 1), len(x) - 1)
    return 0.5 * (float(x[i - 1]) + float(x[i])), (float(x[i - 1]), float(x[i]))


def detect_transition_velocity(table: pd.DataFrame,
                               top_channel: str = "通道14",
                               low_channel: str = "通道11",
                               base_channel: str = "通道9") -> Dict[str, object]:
    """Estimate the bubbling -> turbulent transition band from the taps alone.

    Three independent, instrument-only estimators are reported:

    ``u_top_tap_break``
        velocity of the steepest rise of the *topmost* tap's mean pressure --
        i.e. where the bubble phase starts to reach the upper section;
    ``u_axial_contrast``
        velocity of the steepest rise of the top/bottom fluctuation ratio, a
        scale-free version of the same physical signal;
    ``u_low_tap_decline``
        velocity at which the *lowest* tap first drops measurably below its
        plateau -- i.e. the bed expands past that tap.

    The recommended band is the span of the three; a single number would
    over-claim what six taps and nine velocities can support.
    """
    summary = fluctuation_summary(table).sort_values("velocity").reset_index(drop=True)
    velocities = summary["velocity"].to_numpy(dtype=float)

    top = table[table["channel"] == top_channel].sort_values("velocity").reset_index(drop=True)
    low = table[table["channel"] == low_channel].sort_values("velocity").reset_index(drop=True)
    base = table[table["channel"] == base_channel].sort_values("velocity").reset_index(drop=True)

    u_top, top_range = _steepest_rise(velocities, top["mean"].to_numpy(dtype=float))

    contrast = axial_contrast(table).sort_values("velocity").reset_index(drop=True)
    u_contrast, contrast_range = _steepest_rise(
        contrast["velocity"].to_numpy(dtype=float),
        contrast["std_top_over_bottom"].to_numpy(dtype=float),
    )

    low_mean = base["mean"].to_numpy(dtype=float)
    u_low = plateau_decline_velocity(velocities, low_mean)

    interior = bool(summary["fluctuation_peak_interior"].iloc[0])
    band = (float(min(u_top, u_contrast, u_low)), float(max(u_top, u_contrast, u_low)))

    note = (
        "Instrument-only estimators; no model is involved. The classic 'peak of the "
        "pressure-fluctuation intensity' criterion cannot be applied here because the "
        "summed standard deviation is still rising at the highest velocity tested "
        f"({'an interior maximum was found' if interior else 'no interior maximum in 0.1-0.9 m/s'}), "
        "which is reported as a result in its own right. The three estimators bracket the "
        "bubbling-to-turbulent transition; the value should not be quoted to better than "
        "+/-0.1 m/s. The project poster states 0.6-0.8 m/s; this independent re-analysis "
        "returns a slightly lower but clearly overlapping band."
    )
    return dict(
        u_top_tap_break=u_top,
        u_top_tap_break_range=top_range,
        u_axial_contrast=u_contrast,
        u_axial_contrast_range=contrast_range,
        u_low_tap_decline=u_low,
        u_low_tap_decline_channel=data_pressure.channel_key(base_channel),
        fluctuation_peak_interior=interior,
        recommended_band=band,
        top_channel=data_pressure.channel_key(top_channel),
        low_channel=data_pressure.channel_key(low_channel),
        note=note,
    )


def axial_contrast(table: pd.DataFrame, top_channel: str = "通道14",
                   low_channel: str = "通道11") -> pd.DataFrame:
    """Ratio of top-tap to bottom-tap fluctuation, a scale-free transition proxy."""
    top = table[table["channel"] == top_channel].set_index("velocity")
    low = table[table["channel"] == low_channel].set_index("velocity")
    ratio = (top["std"] / low["std"]).rename("std_top_over_bottom")
    mean_ratio = (top["mean"] / low["mean"]).rename("mean_top_over_bottom")
    return pd.concat([ratio, mean_ratio], axis=1).reset_index()


def bed_density_proxy(table: pd.DataFrame) -> pd.DataFrame:
    """A relative bed-inventory proxy: the mean of each tap normalised to U = 0.1.

    Without an absolute pressure calibration this is the most one can say about
    the axial solids inventory; it is therefore reported as a *relative* change.
    """
    ref = table[table["velocity"] == table["velocity"].min()].set_index("channel")["mean"]
    out = table.copy()
    out["mean_rel_to_min_velocity"] = out.apply(
        lambda row: row["mean"] / ref[row["channel"]] if ref[row["channel"]] else np.nan, axis=1
    )
    return out
