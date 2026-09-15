"""
Reader for the optical-fibre-probe records (``20250609 LC/**/*.pvd``).

What the instrument is
----------------------
The ``.pvd`` files come from a dual-channel optical fibre probe (PV6D family)
that measures the **local solids volume fraction** inside the bed.  The probe
reports *ADC counts* that are converted to a solids fraction with a two-parameter
exponential calibration.

File layout
-----------
A ``.pvd`` file is **plain text** (GBK/CP936 encoded header) followed by
whitespace-separated integer samples, two columns per row::

    文件名,20250606_110010
    采样时间,2025-06-06 11:00:10
    通道数, 2
    数据量, 524288
    采样频率(kHz), 50
    滤波频率(Hz),  2 kHz
    测点间距, 1.7            <- probe spacing, mm
    采样通道,1A,1B
    传感器编号,P1,P1
    传感器量程(V),5,5
    最大值,4815,4621
    最小值,650,623
    平均值,4334.983,4168.243
    标准差,420.632,402.945
    Corsele, 1221
     4458  4286
     4458  4286
     ...

So each record is ``524288 / 50000 Hz ~= 10.5 s`` of two-channel data.

Folder tree
-----------
``<bed height cm>/<r/R>/<gas velocity m/s>/<record>.pvd``, with
``r/R in {0, 0.316, 0.548, 0.707, 0.837, 0.95}`` and ``Ug in {0.1 ... 0.9}``.

Calibration
-----------
Two constant pairs of the form ``eps_s = a * exp(b * V)`` are recoverable from
the analysis scripts, with ``V`` in volts (the stored counts are in mV)::

    0.0164  * exp(0.7759  * V)   # used for h102 / r0.95
    0.01511 * exp(0.79292 * V)   # used for h40 / r0

They differ by a few per-cent.  Both are exposed here, the first is the default,
and :func:`fbfml.analysis_concentration.calibration_plausibility` decides
automatically whether either can be used on a given campaign; see
``docs/data_quality.md`` §4.  Trend and shape conclusions are insensitive to the
choice, absolute values are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import paths

#: Default calibration (counts -> solids fraction).
DEFAULT_CALIBRATION: Tuple[float, float] = (0.0164, 0.7759)
#: Alternative constants recovered from the legacy scripts.
ALTERNATIVE_CALIBRATION: Tuple[float, float] = (0.01511, 0.79292)

#: Seconds of data per record, derived from the header.
EXPECTED_SAMPLES = 524288
EXPECTED_FS_HZ = 50000.0
RADIAL_POSITIONS: List[float] = [0.0, 0.316, 0.548, 0.707, 0.837, 0.95]
BED_HEIGHTS_CM: List[int] = [60, 100]

_HEADER_END_RE = re.compile(rb"\n\s*\d+\s+\d+\s*\n")


@dataclass
class OfpRecord:
    """One optical-fibre-probe acquisition."""

    path: Path
    header: Dict[str, str]
    counts: np.ndarray  # (N, 2) float64, ADC counts
    bed_height_cm: Optional[int] = None
    radial_position: Optional[float] = None
    velocity: Optional[float] = None

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def fs_hz(self) -> float:
        raw = self.header.get("采样频率(kHz)", "50")
        try:
            return float(re.sub(r"[^\d.]", "", raw)) * 1000.0
        except ValueError:
            return EXPECTED_FS_HZ

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs_hz if self.fs_hz else float("nan")

    def solids_fraction(self, calibration: Tuple[float, float] = DEFAULT_CALIBRATION,
                        channel: int = 1) -> np.ndarray:
        """Convert ADC counts to instantaneous local solids fraction.

        Parameters
        ----------
        calibration
            ``(a, b)`` in ``eps_s = a * exp(b * V)`` with ``V`` in volts.
        channel
            0-based column index (0 = probe channel 1A, 1 = channel 1B).  The
            legacy scripts used column 1.
        """
        a, b = calibration
        volts = self.counts[:, channel] / 1000.0  # counts are millivolts
        return a * np.exp(b * volts)

    def mean_solids_fraction(self, calibration: Tuple[float, float] = DEFAULT_CALIBRATION,
                             channel: int = 1) -> float:
        return float(np.mean(self.solids_fraction(calibration, channel)))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_header(text: str) -> Dict[str, str]:
    """Parse the Chinese key/value header lines into a dict."""
    header: Dict[str, str] = {}
    for line in text.splitlines():
        if "," not in line:
            continue
        key, _, value = line.partition(",")
        key, value = key.strip(), value.strip()
        if key and value:
            header[key] = value
    return header


def read_pvd(path: Path, max_samples: Optional[int] = None) -> OfpRecord:
    """Read a ``.pvd`` file into an :class:`OfpRecord`.

    Parameters
    ----------
    max_samples
        Optionally read only the first ``max_samples`` rows (useful for quick
        coverage scans over hundreds of files).
    """
    path = Path(path)
    raw = path.read_bytes()
    match = _HEADER_END_RE.search(raw)
    if match is None:
        raise ValueError(f"{path.name}: could not locate the end of the text header")
    header = parse_header(raw[: match.start()].decode("gbk", errors="replace"))
    # start *at* the match so the first data row (which the pattern also matched)
    # is not swallowed by the header
    body = raw[match.start():].decode("ascii", errors="replace")
    values = np.array(body.split(), dtype=np.float64)
    if values.size % 2:
        values = values[:-1]
    counts = values.reshape(-1, 2)
    if max_samples is not None:
        counts = counts[: int(max_samples)]
    rec = OfpRecord(path=path, header=header, counts=counts)
    height, radial, velocity = parse_location(path)
    rec.bed_height_cm, rec.radial_position, rec.velocity = height, radial, velocity
    return rec


def parse_location(path: Path) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Extract ``(bed_height_cm, r/R, Ug)`` from the folder tree of ``path``.

    The campaign is laid out as ``<height_cm>/<r/R>/<Ug>/<record>.pvd``, so the
    three grand-parent directories carry the coordinates; locating them
    structurally is far more robust than trying to guess them from the numbers.
    """
    velocity_dir = Path(path).parent
    radial_dir = velocity_dir.parent
    height_dir = radial_dir.parent

    def _num(name: str) -> Optional[float]:
        try:
            return float(name)
        except ValueError:
            return None

    velocity = _num(velocity_dir.name)
    radial = _num(radial_dir.name)
    height_raw = _num(height_dir.name)
    height = int(height_raw) if height_raw is not None and height_raw.is_integer() else None
    if velocity is not None and not (0.0 < velocity <= 2.0):
        velocity = None
    if radial is not None and not (0.0 <= radial <= 1.0):
        radial = None
    return height, radial, velocity


# ---------------------------------------------------------------------------
# inventory / scanning
# ---------------------------------------------------------------------------
def iter_pvd_files(directory: Optional[Path] = None) -> List[Path]:
    directory = paths.ofp_dir() if directory is None else Path(directory)
    return sorted(directory.rglob("*.pvd"))


def coverage_table(directory: Optional[Path] = None) -> "object":
    """Availability matrix of the probe campaign (repeats per condition)."""
    import pandas as pd

    directory = paths.ofp_dir() if directory is None else Path(directory)
    rows: List[dict] = []
    for height_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for radial_dir in sorted(p for p in height_dir.iterdir() if p.is_dir()):
            for velocity_dir in sorted(p for p in radial_dir.iterdir() if p.is_dir()):
                files = sorted(velocity_dir.glob("*.pvd"))
                size = sum(f.stat().st_size for f in files)
                rows.append(
                    dict(
                        bed_height_cm=int(height_dir.name),
                        radial_position=float(radial_dir.name),
                        velocity=float(velocity_dir.name),
                        n_records=len(files),
                        megabytes=round(size / 1e6, 2),
                    )
                )
    return pd.DataFrame(rows)


def scan_headers(directory: Optional[Path] = None, limit: Optional[int] = None) -> "object":
    """Read only the headers of every record -> campaign metadata table."""
    import pandas as pd

    files = iter_pvd_files(directory)
    if limit is not None:
        files = files[: int(limit)]
    rows: List[dict] = []
    for path in files:
        try:
            # only the first bytes are needed -- avoids reading ~3 GB of samples
            with open(path, "rb") as handle:
                head = handle.read(4096)
            match = _HEADER_END_RE.search(head)
            header = parse_header(head[: match.start()].decode("gbk", errors="replace")) if match else {}
            height, radial, velocity = parse_location(path)
            rows.append(
                dict(
                    file=path.name,
                    bed_height_cm=height,
                    radial_position=radial,
                    velocity=velocity,
                    fs_hz=float(re.sub(r"[^\d.]", "", header.get("采样频率(kHz)", "50") or "50")) * 1000,
                    n_samples=int(float(re.sub(r"[^\d.]", "", header.get("数据量", "0") or "0"))),
                    probe_spacing_mm=header.get("测点间距"),
                    sensor_id=header.get("传感器编号"),
                    header_mean_ch1=header.get("平均值", "").split(",")[0] if header.get("平均值") else None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(dict(file=path.name, bed_height_cm=None, radial_position=None,
                             velocity=None, fs_hz=None, n_samples=None,
                             probe_spacing_mm=None, sensor_id=None,
                             header_mean_ch1=None, error=type(exc).__name__))
    return pd.DataFrame(rows)
