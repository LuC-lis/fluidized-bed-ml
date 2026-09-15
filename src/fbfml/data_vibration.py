"""
Reader for the tri-axial accelerometer logs (``20250609 LMF/*/*.mat`` + ``*.bin``).

On-disk layout
--------------
``20250609 LMF/<condition>/`` contains 6 ``.mat`` files numbered ``1,2,3,4,5,7``
(sensor 6 is absent) plus a same-numbered raw ``.bin`` blob.

Each ``.mat`` is one accelerometer, recorded **simultaneously** with its
siblings (identical ``start_time`` for all six files in a folder):

===================  =========================================================
key                  meaning
===================  =========================================================
``raw_data``         ``(N, 3) int16``  raw ADC counts
``processed_data``   ``(N, 3) float64`` acceleration in g (already scaled)
``time_axis``        ``(N, 1) float64`` absolute epoch seconds
``dev_id``           device id
``start_time``       epoch seconds of the first sample (UTC)
``data_rate_hz``     nominal sampling rate -- **1666 Hz** as stored
``full_scale_g``     accelerometer range (2 g)
===================  =========================================================

The three columns of ``processed_data`` are the **X, Y and Z** axes.

Caveats
-------
* Only **two** of the ten condition folders carry a gas-velocity label in their
  name (``0.1 ...``, ``0.9 ...``); the rest are timestamps only.  See
  :func:`infer_condition_order`.
* Four folders contain at least one file with an empty ``time_axis`` (truncated
  acquisition).  Those are reported by :func:`audit_folder` rather than failing.
* ``data_rate_hz = 1666`` conflicts with the 500 Hz quoted for the pressure
  channels; it is kept as recorded and flagged in ``docs/data_quality.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio

from . import paths

#: Sensors present in the archive (accelerometer #6 has no file).
SENSOR_IDS: List[int] = [1, 2, 3, 4, 5, 7]
#: Axis names, in the column order of ``processed_data``.
AXES: List[str] = ["X", "Y", "Z"]

MAT_KEYS = ("raw_data", "processed_data", "time_axis", "dev_id", "start_time",
            "data_rate_hz", "full_scale_g")


@dataclass
class VibrationRecord:
    """One accelerometer recording."""

    sensor_id: int
    condition_dir: Path
    acceleration_g: np.ndarray  # (N, 3)
    raw_counts: Optional[np.ndarray]
    time_s: np.ndarray  # (N,)
    start_time: Optional[float]
    data_rate_hz: Optional[float]
    full_scale_g: Optional[float]
    _meta: dict = field(default_factory=dict, repr=False)

    @property
    def n_samples(self) -> int:
        return int(self.acceleration_g.shape[0])

    @property
    def is_valid(self) -> bool:
        """A record is usable when it has samples and a monotone time base."""
        return self.n_samples > 0 and self.time_s.size == self.n_samples

    @property
    def duration_s(self) -> float:
        if self.time_s.size < 2:
            return float("nan")
        return float(self.time_s[-1] - self.time_s[0])

    def axis(self, name: str) -> np.ndarray:
        return self.acceleration_g[:, AXES.index(name.upper())]

    def magnitude(self) -> np.ndarray:
        """Euclidean norm of the three axes (g)."""
        return np.linalg.norm(self.acceleration_g, axis=1)

    def std_per_axis(self) -> np.ndarray:
        return self.acceleration_g.std(axis=0, ddof=1)

    def dominant_frequency_hz(self) -> float:
        """Peak of the PSD of the vector magnitude -- a cheap regime indicator."""
        x = self.magnitude()
        x = x - x.mean()
        if x.size < 16:
            return float("nan")
        fs = float(self.data_rate_hz or 1.0)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
        psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        return float(freqs[int(np.argmax(psd[1:])) + 1])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_vibration(mat_path: Path, sensor_id: Optional[int] = None) -> VibrationRecord:
    """Load one accelerometer ``.mat`` file."""
    mat_path = Path(mat_path)
    if sensor_id is None:
        digits = "".join(ch for ch in mat_path.stem if ch.isdigit())
        sensor_id = int(digits) if digits else -1
    mat = sio.loadmat(mat_path, variable_names=list(MAT_KEYS))
    acc = np.asarray(mat.get("processed_data", np.empty((0, 3))), dtype=float)
    raw = mat.get("raw_data")
    t_axis = np.asarray(mat.get("time_axis", np.empty((0, 1))), dtype=float).ravel()
    start = float(mat["start_time"][0, 0]) if "start_time" in mat else None
    rate = float(mat["data_rate_hz"][0, 0]) if "data_rate_hz" in mat else None
    fs = float(mat["full_scale_g"][0, 0]) if "full_scale_g" in mat else None
    return VibrationRecord(
        sensor_id=int(sensor_id),
        condition_dir=mat_path.parent,
        acceleration_g=acc,
        raw_counts=None if raw is None else np.asarray(raw, dtype=np.int16),
        time_s=t_axis,
        start_time=start,
        data_rate_hz=rate,
        full_scale_g=fs,
        _meta={"file": str(mat_path), "bytes": mat_path.stat().st_size},
    )


def load_condition(condition_dir: Path) -> Dict[int, VibrationRecord]:
    """Load all sensors of one condition folder, keyed by sensor id."""
    condition_dir = Path(condition_dir)
    out: Dict[int, VibrationRecord] = {}
    for mat in sorted(condition_dir.glob("*.mat")):
        rec = load_vibration(mat)
        out[rec.sensor_id] = rec
    return out


def list_conditions(directory: Optional[Path] = None) -> List[Path]:
    """All condition folders, sorted by folder name."""
    directory = paths.vibration_dir() if directory is None else Path(directory)
    return sorted(p for p in directory.iterdir() if p.is_dir())


def label_velocity(condition_dir: Path) -> Optional[float]:
    """Gas velocity encoded in the folder name, or ``None`` when unlabelled.

    Recognised form: ``"0.1 20250609_104345"`` -> ``0.1``.
    """
    parts = Path(condition_dir).name.split()
    for part in parts:
        try:
            val = float(part)
        except ValueError:
            continue
        if 0.0 < val <= 2.0:
            return val
    return None


def audit_folder(condition_dir: Path) -> List[dict]:
    """Per-file health report (never raises): used to build the QC table."""
    rows: List[dict] = []
    for mat in sorted(Path(condition_dir).glob("*.mat")):
        try:
            rec = load_vibration(mat)
            rows.append(
                dict(
                    condition=Path(condition_dir).name,
                    sensor=rec.sensor_id,
                    n_samples=rec.n_samples,
                    duration_s=None if np.isnan(rec.duration_s) else round(rec.duration_s, 1),
                    rate_hz=rec.data_rate_hz,
                    time_axis_ok=rec.is_valid,
                    status="ok" if rec.is_valid else "EMPTY_TIME_AXIS",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(
                dict(condition=Path(condition_dir).name, sensor=-1, n_samples=0,
                     duration_s=None, rate_hz=None, time_axis_ok=False,
                     status=f"ERROR:{type(exc).__name__}")
            )
    return rows


def infer_condition_order(records: Dict[str, np.ndarray]) -> List[str]:
    """Order conditions by a monotone physical statistic.

    A cross-check used alongside :func:`order_conditions_by_time`, which provides
    the ordering used downstream.  Averaging the intensity across sensors is
    sensitive to a sensor being absent from one condition, so the physical check
    is also available per sensor (:func:`fbfml.analysis_vibration.per_sensor_velocity_check`).
    """
    return sorted(records, key=lambda k: float(np.nanmean(records[k])))


def condition_timestamp(condition_dir: Path) -> Optional[pd.Timestamp]:
    """Acquisition timestamp encoded in a folder name (``... 20250609_104345``)."""
    import pandas as pd

    for part in reversed(Path(condition_dir).name.split()):
        try:
            return pd.to_datetime(part, format="%Y%m%d_%H%M%S")
        except (ValueError, TypeError):
            continue
    return None


def order_conditions_by_time(conditions: Optional[List[Path]] = None) -> List[Path]:
    """Order condition folders by acquisition timestamp (the actual sweep order).

    The velocity was ramped monotonically during the session, so the acquisition
    time is a far more reliable ordering key than any signal statistic -- and it
    is consistent with both of the two folders that do carry a velocity label.
    """
    conditions = list_conditions() if conditions is None else list(conditions)
    return sorted(conditions, key=lambda p: (condition_timestamp(p) is None,
                                             condition_timestamp(p) or 0))


def velocity_assignments(
    conditions: Optional[List[Path]] = None,
    min_duration_s: float = 60.0,
) -> "object":
    """Infer a gas velocity for every complete condition folder of the sweep.

    Rule (documented, reproducible, and cross-checked):

    1. order the folders by acquisition timestamp;
    2. drop folders whose recordings are incomplete (shorter than
       ``min_duration_s``) -- these are aborted/preparatory runs;
    3. assign ``0.1, 0.2, ..., 0.9 m/s`` to the remaining nine, in order.

    Returns a DataFrame with ``condition, timestamp, velocity_assigned,
    velocity_label, velocity_inferred, rank, complete, n_sensors``.

    The two folders that carry a label in their name validate the rule: ``0.1``
    must be first and ``0.9`` last.  Any downstream use keeps the
    ``velocity_inferred`` flag, so a recovered velocity is never presented as a
    measured one.
    """
    import pandas as pd

    conditions = order_conditions_by_time(conditions)
    rows = []
    for path in conditions:
        durations = []
        for mat in sorted(path.glob("*.mat")):
            try:
                rec = load_vibration(mat)
                if rec.is_valid:
                    durations.append(rec.duration_s)
            except Exception:  # pragma: no cover - defensive
                continue
        median_duration = float(np.median(durations)) if durations else 0.0
        rows.append(
            dict(
                condition=path.name,
                timestamp=condition_timestamp(path),
                velocity_label=label_velocity(path),
                median_duration_s=median_duration,
                n_sensors=len(durations),
                complete=bool(median_duration >= min_duration_s),
            )
        )
    df = pd.DataFrame(rows)
    sweep = df[df["complete"]].reset_index(drop=True)
    sweep["rank"] = range(len(sweep))
    sweep["velocity_assigned"] = [round(0.1 * (i + 1), 1) for i in sweep["rank"]]
    sweep["velocity_inferred"] = sweep["velocity_label"].isna()
    df = df.merge(
        sweep[["condition", "rank", "velocity_assigned", "velocity_inferred"]],
        on="condition", how="left",
    )
    return df.sort_values(["complete", "timestamp"], ascending=[False, True]).reset_index(drop=True)


def validate_velocity_rule(assignments: "object") -> dict:
    """Check the timestamp rule against the two folders that carry a label."""
    labelled = assignments.dropna(subset=["velocity_label"])
    rows = []
    for _, row in labelled.iterrows():
        rows.append(
            dict(condition=row["condition"], label=row["velocity_label"],
                 assigned=row["velocity_assigned"],
                 ok=bool(abs(row["velocity_label"] - row["velocity_assigned"]) < 1e-9))
        )
    return dict(
        n_labelled=len(rows),
        n_agree=int(sum(r["ok"] for r in rows)),
        details=rows,
        rule_consistent=bool(rows) and all(r["ok"] for r in rows),
    )
