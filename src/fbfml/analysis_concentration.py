"""
Optical-fibre-probe analysis: the local solids-fraction field.

For every ``.pvd`` record the module returns the time-averaged local solids
fraction ``eps_s`` (from the two-parameter exponential calibration described in
:mod:`fbfml.data_ofp`) together with its fluctuation.  Aggregating over the
repeats of a condition yields the radial profile ``eps_s(r/R)`` and, stacking
the two bed heights, an axial picture of the bed.

This is the only part of the project with a genuinely *dense* design: two bed
heights x six radial positions x nine gas velocities x (up to five) repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_ofp, paths

Calibration = Tuple[float, float]


@dataclass
class ConditionResult:
    bed_height_cm: int
    radial_position: float
    velocity: float
    n_records: int
    eps_mean: float
    eps_std_between_records: float
    eps_std_within_record: float
    counts_mean: float


def _progress(iterable, enabled: bool, total: Optional[int] = None):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="OFP records", unit="file")
    except ImportError:  # pragma: no cover
        return iterable


def process_records(
    directory: Optional[Path] = None,
    calibration: Calibration = data_ofp.DEFAULT_CALIBRATION,
    channel: int = 1,
    progress: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read every probe record and return one row per record.

    Columns: ``bed_height_cm, radial_position, velocity, file, n_samples,
    duration_s, counts_mean, eps_mean, eps_std, eps_p95``.
    """
    files = data_ofp.iter_pvd_files(directory)
    if limit is not None:
        files = files[: int(limit)]
    rows: List[dict] = []
    for path in _progress(files, progress, total=len(files)):
        try:
            rec = data_ofp.read_pvd(path)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(dict(file=str(path), error=f"{type(exc).__name__}: {exc}"))
            continue
        eps = rec.solids_fraction(calibration, channel=channel)
        counts = rec.counts[:, channel]
        rows.append(
            dict(
                bed_height_cm=rec.bed_height_cm,
                radial_position=rec.radial_position,
                velocity=rec.velocity,
                file=path.name,
                n_samples=rec.n_samples,
                duration_s=round(rec.duration_s, 3),
                counts_mean=float(counts.mean()),
                counts_min=float(counts.min()),
                counts_max=float(counts.max()),
                frac_below_zero=float(np.mean(counts < 0.0)),
                frac_rails=float(np.mean(np.abs(counts) >= 4990.0)),
                eps_mean=float(eps.mean()),
                eps_std=float(eps.std(ddof=1)),
                eps_p95=float(np.percentile(eps, 95)),
            )
        )
    return pd.DataFrame(rows)


def aggregate_by_condition(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-record table to one row per (height, r/R, Ug)."""
    if records.empty:
        return records
    usable = records.dropna(subset=["bed_height_cm", "radial_position", "velocity", "eps_mean"])
    grouped = usable.groupby(["bed_height_cm", "radial_position", "velocity"])
    out = grouped.agg(
        n_records=("eps_mean", "size"),
        eps_mean=("eps_mean", "mean"),
        eps_std_between_records=("eps_mean", "std"),
        eps_std_within_record=("eps_std", "mean"),
        counts_mean=("counts_mean", "mean"),
    ).reset_index()
    out["eps_std_between_records"] = out["eps_std_between_records"].fillna(0.0)
    return out.sort_values(["bed_height_cm", "radial_position", "velocity"]).reset_index(drop=True)


def radial_profiles(summary: pd.DataFrame, bed_height_cm: Optional[int] = None) -> pd.DataFrame:
    """Convenience view: ``eps_mean`` vs ``r/R`` for each velocity."""
    df = summary if bed_height_cm is None else summary[summary["bed_height_cm"] == bed_height_cm]
    return df.pivot_table(
        index="radial_position", columns="velocity", values="eps_mean"
    ).sort_index()


def axial_profile(summary: pd.DataFrame, velocity: Optional[float] = None) -> pd.DataFrame:
    """``eps_mean`` at each bed height, for a given velocity."""
    df = summary if velocity is None else summary[np.isclose(summary["velocity"], velocity)]
    return df.pivot_table(
        index="radial_position", columns="bed_height_cm", values="eps_mean"
    ).sort_index()


def core_annulus_ratio(summary: pd.DataFrame, value_col: str = "eps_mean") -> pd.DataFrame:
    """Core-to-annulus contrast (``r/R = 0`` over ``r/R = 0.95``) per condition.

    A convenient scalar summary of the radial non-uniformity that any scale-up
    model has to reproduce.  Works on the calibration-free probe voltage
    (``value_col="counts_mean"``) as well as on the calibrated solids fraction.
    """
    core = summary[np.isclose(summary["radial_position"], 0.0)]
    wall = summary[np.isclose(summary["radial_position"], 0.95)]
    merged = core.merge(
        wall,
        on=["bed_height_cm", "velocity"],
        suffixes=("_core", "_wall"),
    )
    if merged.empty:
        return merged
    merged["core_over_wall"] = merged[f"{value_col}_core"] / merged[f"{value_col}_wall"]
    return merged[
        ["bed_height_cm", "velocity", f"{value_col}_core", f"{value_col}_wall",
         "core_over_wall", "n_records_core", "n_records_wall"]
    ].sort_values(["bed_height_cm", "velocity"]).reset_index(drop=True)


def calibration_plausibility(summary: pd.DataFrame) -> dict:
    """Diagnose whether the recovered calibration can be trusted on this campaign.

    Two checks:

    * **range** -- the calibrated solids fraction of a bubbling/turbulent bed
      should span a wide interval (roughly 0.02-0.6).  A field squeezed into a
      narrow high band means the constants do not belong to this probe;
    * **radial gradient** -- the time-averaged solids fraction must be *lower*
      on the axis than near the wall (core-annulus structure).

    Returns a dict with the numbers and a boolean verdict.  This exists so that
    an unusable absolute calibration is caught by the code instead of being
    published.
    """
    eps = summary["eps_mean"]
    span = float(eps.max() - eps.min())
    contrast = core_annulus_ratio(summary, "eps_mean")
    gradient_ok = bool(len(contrast) > 0 and (contrast["core_over_wall"] < 1.0).all())
    # physical plausibility: a real campaign spans a wide range of solids fraction
    range_ok = bool(span >= 0.25)
    return dict(
        eps_min=float(eps.min()),
        eps_max=float(eps.max()),
        eps_span=span,
        range_plausible=range_ok,
        core_annulus_gradient_correct=gradient_ok,
        n_conditions=len(summary),
        calibration_usable=bool(range_ok and gradient_ok),
    )


def calibration_sensitivity(records: pd.DataFrame) -> pd.DataFrame:
    """How much do the two recovered calibration constants disagree?

    Returned as the relative difference of the mean solids fraction, per
    condition; it bounds how much of any absolute conclusion is trustworthy.
    """
    alt = process_records(calibration=data_ofp.ALTERNATIVE_CALIBRATION, progress=False)
    a = aggregate_by_condition(records)[["bed_height_cm", "radial_position", "velocity", "eps_mean"]]
    b = aggregate_by_condition(alt)[["bed_height_cm", "radial_position", "velocity", "eps_mean"]]
    merged = a.merge(b, on=["bed_height_cm", "radial_position", "velocity"], suffixes=("_default", "_alt"))
    merged["rel_diff"] = (
        (merged["eps_mean_default"] - merged["eps_mean_alt"]).abs() / merged["eps_mean_default"]
    )
    return merged


# ---------------------------------------------------------------------------
# quantitative statements about the measured field
# ---------------------------------------------------------------------------
def radial_gradient_test(summary: pd.DataFrame, value_col: str = "counts_mean",
                         min_points: int = 4) -> pd.DataFrame:
    """Least-squares fit of the probe signal against ``r/R`` for each condition.

    The core-annulus structure predicts a **positive** slope (the local solids
    fraction, hence the probe signal, grows from the axis towards the wall).
    Reporting the slope with its R2 and a significance flag turns a
    hand-waving "the profiles look reasonable" into a testable statement.
    """
    from scipy import stats as sp_stats

    rows: List[dict] = []
    for (height, velocity), sub in summary.groupby(["bed_height_cm", "velocity"]):
        sub = sub.dropna(subset=[value_col]).sort_values("radial_position")
        if len(sub) < min_points:
            continue
        x = sub["radial_position"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        fit = sp_stats.linregress(x, y)
        rows.append(
            dict(
                bed_height_cm=height,
                velocity=velocity,
                n_points=len(sub),
                slope_per_unit_rR=float(fit.slope),
                stderr=float(fit.stderr),
                r_squared=float(fit.rvalue ** 2),
                p_value=float(fit.pvalue),
                significant=bool(fit.pvalue < 0.05),
                slope_positive=bool(fit.slope > 0),
            )
        )
    return pd.DataFrame(rows).sort_values(["bed_height_cm", "velocity"]).reset_index(drop=True)


def noise_to_signal(summary: pd.DataFrame, value_col: str = "counts_mean") -> dict:
    """Compare run-to-run repeatability with the physical spatial variation.

    If the standard deviation between repeats of the *same* condition is of the
    same order as the variation *across* the field, then only broad trends can be
    read from the campaign and single-point comparisons are meaningless.  This
    number is the honest error budget of the probe dataset.
    """
    if "eps_std_between_records" not in summary.columns:
        raise ValueError("summary must contain eps_std_between_records (repeat spread)")
    # repeat spread expressed on the same scale as the signal
    repeat_spread = float(summary["eps_std_between_records"].mean())
    # convert a typical calibrated spread back to voltage units through the
    # calibration slope d(eps)/dV evaluated at the mean level
    a, b = data_ofp.DEFAULT_CALIBRATION
    v_mean = float(summary[value_col].mean())
    d_eps_dv = a * b * np.exp(b * v_mean / 1000.0) / 1000.0
    repeat_spread_mv = repeat_spread / d_eps_dv if d_eps_dv > 0 else float("nan")
    spatial_range_mv = float(summary[value_col].max() - summary[value_col].min())
    return dict(
        mean_repeat_spread_mv=repeat_spread_mv,
        spatial_range_mv=spatial_range_mv,
        noise_to_signal=repeat_spread_mv / spatial_range_mv if spatial_range_mv else float("nan"),
        interpretation=(
            "ratio << 1: the spatial structure dominates the run-to-run drift; "
            "ratio ~ 1: single-point comparisons are not meaningful and only "
            "trends averaged over many conditions should be reported."
        ),
    )
