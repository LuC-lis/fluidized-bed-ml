"""
Data inventory: what exists, where, how big, and how (in)complete it is.

Produces the "asset register" that any reusable release needs: one row per raw
record family, plus the completeness matrix of the probe campaign.  This is the
module that turns a folder of recordings into a *documented dataset*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data_ofp, data_pressure, data_vibration, paths


@dataclass
class DatasetSummary:
    key: str
    folder: str
    n_files: int
    bytes_total: int
    extensions: str

    @property
    def megabytes(self) -> float:
        return self.bytes_total / 1e6


#: Which file extensions belong to which dataset.
DATASET_EXTENSIONS: Dict[str, tuple] = {
    "pressure": (".xlsx",),
    "vibration": (".mat", ".bin"),
    "ofp": (".pvd", ".ini", ".txt", ".xls"),
    "tabular": (".xlsx",),
}


def summarise_dataset(key: str) -> DatasetSummary:
    """Count files / bytes for one dataset folder."""
    folder = paths.dataset_path(key)
    extensions = DATASET_EXTENSIONS.get(key, ())
    n_files = 0
    total = 0
    if folder.is_dir():
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            n_files += 1
            total += path.stat().st_size
    return DatasetSummary(
        key=key,
        folder=folder.name,
        n_files=n_files,
        bytes_total=total,
        extensions=",".join(extensions),
    )


def full_inventory() -> pd.DataFrame:
    """Asset register for every dataset of the project."""
    rows = []
    for key in ("pressure", "vibration", "ofp", "tabular"):
        s = summarise_dataset(key)
        rows.append(
            dict(
                dataset=s.key,
                folder=s.folder,
                n_files=s.n_files,
                size_mb=round(s.megabytes, 1),
                extensions=s.extensions,
                exists=paths.dataset_path(key).is_dir(),
            )
        )
    return pd.DataFrame(rows)


def file_level_inventory(key: str) -> pd.DataFrame:
    """One row per raw file (name, size, parsed coordinates where available)."""
    folder = paths.dataset_path(key)
    rows: List[dict] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            dict(
                dataset=key,
                relative_path=str(path.relative_to(folder)),
                name=path.name,
                suffix=path.suffix.lower(),
                size_mb=round(path.stat().st_size / 1e6, 3),
            )
        )
    return pd.DataFrame(rows)


def pressure_summary() -> pd.DataFrame:
    """Per-velocity shape check of the pressure logs."""
    rows = []
    for velocity, rec in sorted(data_pressure.load_all_pressure().items()):
        rows.append(
            dict(
                velocity=velocity,
                n_rows=rec.signals.shape[0],
                sampling_hz=round(rec.sampling_hz, 2),
                duration_s=round(rec.duration_s, 2),
                n_channels=rec.signals.shape[1],
                source=rec.source.name,
            )
        )
    return pd.DataFrame(rows)


def vibration_audit() -> pd.DataFrame:
    """Health report for every accelerometer file (catches the corrupt ones)."""
    rows: List[dict] = []
    for condition in data_vibration.list_conditions():
        rows.extend(data_vibration.audit_folder(condition))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["velocity_label"] = df["condition"].map(
            lambda name: data_vibration.label_velocity(Path(name))
        )
    return df


def ofp_coverage() -> pd.DataFrame:
    """Completeness matrix of the optical-probe campaign."""
    df = data_ofp.coverage_table()
    if df.empty:
        return df
    df["complete"] = df["n_records"] >= 5
    return df.sort_values(["bed_height_cm", "radial_position", "velocity"]).reset_index(drop=True)


def ofp_header_scan(limit: Optional[int] = None) -> pd.DataFrame:
    """Metadata scan of the probe campaign (headers only, fast)."""
    return data_ofp.scan_headers(limit=limit)


def coverage_gaps(coverage: pd.DataFrame) -> pd.DataFrame:
    """Conditions with fewer than 5 usable repeats -- i.e. what must be documented."""
    if coverage.empty:
        return coverage
    return coverage[coverage["n_records"] < 5].reset_index(drop=True)
