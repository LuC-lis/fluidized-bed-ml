"""
Pressure-drop bookkeeping and the `U_mf` question.

The physics
-----------
Two regimes, one measurement:

* **below** minimum fluidisation the bed is a packed bed, so the total pressure
  drop rises with gas velocity (Ergun);
* **above** it the bed is fully fluidised and the total pressure drop is fixed by
  the bed weight per unit area, ``ΔP_total = W/A = ρ_bulk · g · H_bed``, i.e.
  *independent of gas velocity*.

The classic `U_mf` method intersects the fixed-bed line with the fluidised
plateau.  If the campaign never sampled the fixed-bed branch, the intersection
is not identifiable -- and saying so, with the numbers that show it, is a result
in its own right.

The same bookkeeping also cross-checks an open question from
``docs/data_quality.md`` §9: the unknown unit of the pressure readings.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import data_pressure


def total_and_cumulative(table: pd.DataFrame) -> pd.DataFrame:
    """Total bed pressure drop and the cumulative drop up to each tap height.

    ``table`` is the output of :func:`fbfml.analysis_regime.channel_statistics`
    (one row per velocity x channel).
    """
    channels = data_pressure.CHANNELS_LOW_TO_HIGH
    heights = [data_pressure.CHANNEL_HEIGHTS_CM[c][1] for c in channels]
    pivot = (
        table.pivot_table(index="velocity", columns="channel", values="mean")
        .reindex(columns=channels)
        .sort_index()
    )
    out = pd.DataFrame({"velocity": pivot.index.astype(float)})
    cumulative = pivot.cumsum(axis=1)
    for channel, height in zip(channels, heights):
        out[f"cum_to_{height}cm"] = cumulative[channel].to_numpy()
    out["total"] = pivot.sum(axis=1).to_numpy()
    return out.reset_index(drop=True)


def plateau_statistics(total: pd.DataFrame, column: str = "total") -> Dict[str, float]:
    """Is the total pressure drop flat?  Slope, relative spread, bootstrap CI."""
    x = total["velocity"].to_numpy(dtype=float)
    y = total[column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)

    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(2000):
        y_star = slope * x + intercept + rng.choice(residual, size=residual.size, replace=True)
        slopes.append(float(np.polyfit(x, y_star, 1)[0]))
    slopes = np.asarray(slopes)
    ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])
    x_span = float(x.max() - x.min())
    return dict(
        mean=float(y.mean()),
        std=float(y.std(ddof=1)),
        relative_spread=float(y.std(ddof=1) / y.mean()),
        slope_per_ms=float(slope),
        slope_ci_low=float(ci_low),
        slope_ci_high=float(ci_high),
        slope_consistent_with_zero=bool(ci_low < 0 < ci_high),
        max_relative_deviation=float(np.max(np.abs(y - y.mean())) / y.mean()),
        # the quantity that actually settles the physics: how much does the total
        # change over the whole velocity range, against what a packed bed would do
        relative_change_across_range=float(slope * x_span / y.mean()),
        relative_change_ci_low=float(ci_low * x_span / y.mean()),
        relative_change_ci_high=float(ci_high * x_span / y.mean()),
        packed_bed_expectation_ratio=float(x.max() / x.min()),
    )


def umf_identifiability(
    total: pd.DataFrame,
    umf_grid: Optional[Sequence[float]] = None,
    column: str = "total",
) -> pd.DataFrame:
    """Profile the residual sum of squares against an assumed ``U_mf``.

    Model: ``ΔP = a + b·U`` for ``U < U_mf``, and ``ΔP = ΔP_plateau`` for
    ``U >= U_mf``.  Scanning ``U_mf`` over the whole plausible range and recording
    the best achievable SSE shows whether the data pin the parameter down at all.
    A flat profile means "not identifiable".
    """
    x = total["velocity"].to_numpy(dtype=float)
    y = total[column].to_numpy(dtype=float)
    if umf_grid is None:
        umf_grid = np.round(np.arange(0.01, 0.61, 0.01), 3)

    rows = []
    for umf in umf_grid:
        low = x < umf
        high = ~low
        if low.sum() >= 2:
            slope, intercept = np.polyfit(x[low], y[low], 1)
            pred_low = slope * x[low] + intercept
            sse_low = float(np.sum((y[low] - pred_low) ** 2))
        else:
            # no fixed-bed data at all: the only admissible "fixed-bed" model is a
            # delta, so a model that skips straight to the plateau costs nothing
            sse_low = 0.0
        plateau = float(y[high].mean()) if high.sum() else float(y.mean())
        sse_high = float(np.sum((y[high] - plateau) ** 2))
        rows.append(dict(umf=float(umf), n_below=int(low.sum()), n_above=int(high.sum()),
                         sse=sse_low + sse_high, plateau=plateau))
    profile = pd.DataFrame(rows)
    best = float(profile["sse"].min())
    profile["excess_sse"] = profile["sse"] - best
    # 95 % profile-likelihood interval, chi-square(1) threshold
    profile["identifiable"] = profile["excess_sse"] <= 3.84
    return profile


def implied_bulk_density(
    total_value: float,
    bed_height_m: float = 1.0,
    g: float = 9.81,
    unit_scale_to_pa: float = 1000.0,
    bulk_density_reference: tuple = (550.0, 850.0),
) -> Dict[str, float]:
    """Back out the bed-averaged bulk density *if* the readings are in kPa.

    ``ΔP_total = ρ_bulk · g · H``.  For an FCC catalyst bed, bulk densities of
    roughly 550-850 kg/m3 are expected (particle density ~1500-1800 kg/m3 with a
    bubbling-bed voidage of 0.5-0.65), so agreement is evidence -- not proof --
    that the pressure unit is kPa.
    """
    pa = float(total_value) * unit_scale_to_pa
    rho = pa / (g * bed_height_m)
    lo, hi = bulk_density_reference
    return dict(
        total_reading=float(total_value),
        assumed_unit="kPa",
        pressure_drop_pa=pa,
        implied_bulk_density=rho,
        reference_range=bulk_density_reference,
        within_expected_range=bool(lo <= rho <= hi),
    )
