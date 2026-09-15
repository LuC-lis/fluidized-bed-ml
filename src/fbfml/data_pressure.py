"""
Reader for the differential-pressure logs (``Pressure_Energy/*.xlsx``).

File layout (one file per gas velocity)
---------------------------------------
* sheet ``Sheet1``
* first column ``时间`` = time in seconds, step 0.002 s  ->  500 Hz
* 60000 rows -> 120 s per record
* columns ``通道9 .. 通道14`` = 6 differential-pressure transducers mounted at
  increasing height (10-30 ... 110-130 cm above the distributor)

The physical unit of the readings is **not recorded anywhere in the project**
(volts or kPa are both plausible).  Everything downstream is therefore treated
as a *relative* pressure signal, and absolute calibration is flagged as an open
question in ``docs/data_quality.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import paths

TIME_COLUMN = "时间"
CHANNEL_COLUMNS: List[str] = [f"通道{k}" for k in range(9, 15)]
#: Nominal height (cm above distributor) of each transducer, low -> high.
CHANNEL_HEIGHTS_CM: Dict[str, tuple] = {
    "通道9": (10, 30),
    "通道10": (30, 50),
    "通道11": (50, 70),
    "通道12": (70, 90),
    "通道13": (90, 110),
    "通道14": (110, 130),
}
#: Velocities for which a pressure log exists (folder name == velocity in m/s).
KNOWN_VELOCITIES: List[float] = [round(0.1 * i, 1) for i in range(1, 10)]

#: ASCII-safe short labels for plots and JSON keys (the raw names are Chinese
#: and would render as missing glyphs on a machine without CJK fonts).
CHANNEL_LABELS: Dict[str, str] = {
    "通道9": "T1 (10-30 cm)",
    "通道10": "T2 (30-50 cm)",
    "通道11": "T3 (50-70 cm)",
    "通道12": "T4 (70-90 cm)",
    "通道13": "T5 (90-110 cm)",
    "通道14": "T6 (110-130 cm)",
}
#: Ordered low -> high, which is the order every figure must use.
CHANNELS_LOW_TO_HIGH: List[str] = sorted(CHANNEL_COLUMNS, key=lambda c: CHANNEL_HEIGHTS_CM[c][0])


def channel_label(channel: str) -> str:
    """ASCII label for a pressure channel (falls back to the raw name)."""
    return CHANNEL_LABELS.get(channel, channel)


def channel_key(channel: str) -> str:
    """ASCII identifier usable as a JSON key / column name: ``T1``..``T6``."""
    return channel_label(channel).split(" ")[0]


@dataclass
class PressureRecord:
    """One gas-velocity pressure record."""

    velocity: float
    time_s: np.ndarray  # (n,)
    signals: pd.DataFrame  # (n, 6) columns == CHANNEL_COLUMNS
    source: Path

    @property
    def sampling_hz(self) -> float:
        dt = np.median(np.diff(self.time_s))
        return float(1.0 / dt) if dt > 0 else float("nan")

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def std(self) -> pd.Series:
        """Standard deviation of each transducer (the 'error bar' in the deck)."""
        return self.signals.std(ddof=1)

    def mean(self) -> pd.Series:
        return self.signals.mean()


def velocity_from_filename(path: Path) -> Optional[float]:
    """``"0.3.xlsx"`` -> ``0.3``; returns ``None`` for the derived ``energy_*`` files."""
    stem = path.stem
    if stem.startswith("energy_"):
        return None
    try:
        return float(stem)
    except ValueError:
        return None


def load_pressure_record(path: Path, velocity: Optional[float] = None) -> PressureRecord:
    """Read one ``<velocity>.xlsx`` pressure log."""
    path = Path(path)
    if velocity is None:
        velocity = velocity_from_filename(path)
        if velocity is None:
            raise ValueError(f"cannot infer velocity from {path.name}; pass velocity=")
    df = pd.read_excel(path)
    missing = [c for c in (TIME_COLUMN, *CHANNEL_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")
    return PressureRecord(
        velocity=float(velocity),
        time_s=df[TIME_COLUMN].to_numpy(dtype=float),
        signals=df[CHANNEL_COLUMNS].astype(float),
        source=path,
    )


def load_all_pressure(directory: Optional[Path] = None) -> Dict[float, PressureRecord]:
    """Load every pressure log in a folder, keyed by gas velocity (m/s)."""
    directory = paths.pressure_dir() if directory is None else Path(directory)
    records: Dict[float, PressureRecord] = {}
    for path in sorted(directory.glob("*.xlsx")):
        velocity = velocity_from_filename(path)
        if velocity is None:
            continue
        records[velocity] = load_pressure_record(path, velocity)
    return records


def load_vendor_energy_table(path: Path) -> pd.DataFrame:
    """Load one of the group's stored ``energy_*.xlsx`` tables (for validation)."""
    return pd.read_excel(path)
