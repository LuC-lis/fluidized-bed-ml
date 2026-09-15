"""
Vibration-side analysis.

Two things are computed here:

1. **Feature extraction** -- the db4 12-level wavelet energies plus time-domain
   descriptors, per sensor per axis per condition.  This is the feature family
   the models consumed.
2. **Condition ordering** -- only two of the ten condition folders carry a gas
   velocity label.  The remaining eight are ordered by a monotone physical
   statistic (growth of wall-vibration intensity with gas velocity), and every
   table produced here keeps ``velocity_label`` (given) and ``condition_rank``
   (inferred) so the two can never be confused.

Memory note
-----------
The raw recordings are ~1.7 M samples x 3 axes x 6 sensors per condition
(~250 MB per condition).  The functions below therefore **stream** one condition
at a time and keep only the derived features; nothing loads the whole campaign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import data_vibration, features_stats, features_wavelet, paths


def _progress(iterable: Iterable, enabled: bool, total: Optional[int] = None):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="vibration conditions", unit="folder")
    except ImportError:  # pragma: no cover
        return iterable


def sensor_features(
    records: Dict[int, "data_vibration.VibrationRecord"],
    config: features_wavelet.WaveletConfig = features_wavelet.WaveletConfig(),
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Wavelet + time-domain features for every axis of every sensor in memory."""
    rows: List[dict] = []
    for sensor_id, rec in sorted(records.items()):
        if not rec.is_valid:
            rows.append(dict(sensor=sensor_id, status="SKIPPED_EMPTY_TIME_AXIS"))
            continue
        fs = float(rec.data_rate_hz or 1.0)
        acc = rec.acceleration_g if max_samples is None else rec.acceleration_g[:max_samples]
        for axis_index, axis_name in enumerate(data_vibration.AXES):
            signal = acc[:, axis_index]
            row = features_wavelet.feature_row(signal, config)
            row.update({f"stat_{k}": v for k, v in features_stats.basic_stats(signal).items()})
            row["dominant_freq_hz"] = features_stats.dominant_frequency_hz(signal, fs)
            row.update(dict(sensor=sensor_id, axis=axis_name, status="ok",
                            n_samples=int(signal.size), fs_hz=fs))
            rows.append(row)
    return pd.DataFrame(rows)


def condition_feature_table(
    progress: bool = True,
    max_samples: Optional[int] = None,
    config: features_wavelet.WaveletConfig = features_wavelet.WaveletConfig(),
) -> pd.DataFrame:
    """One row per (condition, sensor, axis), streamed condition by condition.

    The condition -> gas-velocity mapping comes from the documented timestamp
    rule in :func:`fbfml.data_vibration.velocity_assignments`; the vibration
    intensity is carried along as a physical cross-check, not as the ordering
    key.
    """
    conditions = data_vibration.list_conditions()
    frames: List[pd.DataFrame] = []
    for condition in _progress(conditions, progress, total=len(conditions)):
        records = data_vibration.load_condition(condition)
        df = sensor_features(records, config=config, max_samples=max_samples)
        del records
        if df.empty:
            continue
        df.insert(0, "condition", condition.name)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    assignments = data_vibration.velocity_assignments(conditions)
    mapping = assignments.set_index("condition")
    out["velocity_assigned"] = out["condition"].map(mapping["velocity_assigned"])
    out["velocity_label"] = out["condition"].map(mapping["velocity_label"])
    out["velocity_inferred"] = out["condition"].map(mapping["velocity_inferred"])
    out["condition_rank"] = out["condition"].map(mapping["rank"])
    out["condition_complete"] = out["condition"].map(mapping["complete"])

    # physical cross-check only (never the ordering key): the wall-vibration
    # intensity must grow with gas velocity
    ok = out[out["status"] == "ok"]
    out["intensity"] = out["condition"].map(ok.groupby("condition")["stat_std"].mean())
    return out


def band_trends(features: pd.DataFrame, sensor: int = 1, axis: str = "Z") -> pd.DataFrame:
    """Band energy ratios vs the assigned gas velocity, for one channel."""
    df = features[(features["sensor"] == sensor) & (features["axis"] == axis)].copy()
    df = df.sort_values("condition_rank")
    keep = [f"Level{i}_ratio" for i in range(1, 13)]
    return df[["condition", "condition_rank", "velocity_label", "velocity_assigned",
               "intensity", *keep]]


def intensity_vs_velocity(features: pd.DataFrame) -> pd.DataFrame:
    """Condition intensity vs the assigned velocity (a physical cross-check)."""
    return (
        features.dropna(subset=["condition_rank"])
        .groupby(["condition", "condition_rank", "velocity_assigned", "velocity_label"], dropna=False)
        .agg(intensity=("intensity", "first"), n=("stat_std", "size"))
        .reset_index()
        .sort_values("condition_rank")
    )


def per_sensor_velocity_check(features: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of vibration intensity with velocity, *per sensor*.

    Done per sensor on purpose: aggregating across sensors first would let a
    sensor that is missing from one condition bias the whole comparison (which is
    exactly what happens in this campaign).
    """
    from scipy import stats as sp_stats

    ok = features[(features["status"] == "ok")].dropna(subset=["condition_rank"])
    rows = []
    for (sensor, axis), sub in ok.groupby(["sensor", "axis"]):
        per_condition = sub.groupby("condition_rank")["stat_std"].mean()
        if per_condition.size < 4:
            continue
        rho, p_value = sp_stats.spearmanr(per_condition.index, per_condition.values)
        rows.append(dict(sensor=sensor, axis=axis, n_conditions=per_condition.size,
                         spearman_rho=float(rho), p_value=float(p_value)))
    return pd.DataFrame(rows).sort_values(["sensor", "axis"]).reset_index(drop=True)


def per_sensor_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Sanity table: sample counts and intensity per sensor."""
    ok = features[features["status"] == "ok"]
    return (
        ok.groupby("sensor")
        .agg(n_rows=("axis", "size"),
             mean_intensity=("intensity", "mean"),
             mean_dominant_freq=("dominant_freq_hz", "mean"))
        .reset_index()
    )


def band_ratio_table(features: pd.DataFrame) -> pd.DataFrame:
    """Median band-energy ratio per level across all sensors, per condition.

    Aggregating over sensors suppresses sensor-to-sensor mounting differences
    and exposes the condition-level trend.
    """
    ok = features[features["status"] == "ok"].copy()
    ratio_cols = [f"Level{i}_ratio" for i in range(1, 13)]
    out = ok.groupby(
        ["condition", "condition_rank", "velocity_assigned"], dropna=False
    )[ratio_cols].median()
    return out.reset_index().sort_values("condition_rank")


def monotonicity_score(features: pd.DataFrame, value_col: str) -> float:
    """Spearman rank correlation of ``value_col`` with the assigned velocity.

    Values near +/-1 mean the feature tracks the gas-velocity sweep
    monotonically; values near 0 mean it carries no velocity information.
    """
    from scipy import stats as sp_stats

    df = features.dropna(subset=["condition_rank"])
    if df.empty:
        return float("nan")
    per_condition = df.groupby("condition_rank")[value_col].mean()
    if per_condition.size < 3:
        return float("nan")
    rho, _ = sp_stats.spearmanr(per_condition.index, per_condition.values)
    return float(rho)
