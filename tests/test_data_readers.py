"""Tests for the raw-data readers (they must be robust to damaged recordings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fbfml import data_ofp, data_pressure, data_vibration, paths


# ---------------------------------------------------------------------------
# pressure
# ---------------------------------------------------------------------------
def test_pressure_record_shape():
    if not paths.pressure_dir().is_dir():
        pytest.skip("no pressure data")
    records = data_pressure.load_all_pressure()
    if not records:
        pytest.skip("no pressure files")
    for velocity, rec in records.items():
        assert rec.signals.shape[1] == 6
        assert rec.sampling_hz == pytest.approx(500.0, rel=1e-6)
        assert rec.duration_s == pytest.approx(120.0, abs=0.01)
        assert 0.0 < velocity <= 1.0


def test_pressure_channel_helpers():
    assert data_pressure.channel_key("通道14") == "T6"
    assert "T1" in data_pressure.channel_label("通道9")
    assert data_pressure.CHANNELS_LOW_TO_HIGH[0] == "通道9"
    assert data_pressure.CHANNELS_LOW_TO_HIGH[-1] == "通道14"
    assert data_pressure.velocity_from_filename(Path("0.3.xlsx")) == 0.3
    assert data_pressure.velocity_from_filename(Path("energy_0.3.xlsx")) is None


# ---------------------------------------------------------------------------
# vibration
# ---------------------------------------------------------------------------
def test_velocity_label_parsing():
    assert data_vibration.label_velocity(Path("0.1 20250609_104345")) == 0.1
    assert data_vibration.label_velocity(Path("20250609_104345")) is None


def test_condition_timestamp_parsing():
    ts = data_vibration.condition_timestamp(Path("0.1 20250609_104345"))
    assert ts is not None and (ts.year, ts.hour, ts.minute, ts.second) == (2025, 10, 43, 45)


@pytest.mark.slow
def test_timestamp_rule_matches_the_labelled_folders():
    """The inferred velocity ordering must reproduce both labelled folders.

    This is the guard-rail for the single most fragile inference in the project.
    """
    if not paths.vibration_dir().is_dir():
        pytest.skip("no vibration data")
    assignments = data_vibration.velocity_assignments()
    verdict = data_vibration.validate_velocity_rule(assignments)
    assert verdict["n_labelled"] >= 1, "expected at least one labelled folder"
    assert verdict["rule_consistent"], verdict
    sweep = assignments.dropna(subset=["velocity_assigned"])
    assert len(sweep) == 9, f"expected a nine-point sweep, got {len(sweep)}"


# ---------------------------------------------------------------------------
# optical fibre probe
# ---------------------------------------------------------------------------
def test_ofp_header_and_samples():
    files = data_ofp.iter_pvd_files()
    if not files:
        pytest.skip("no probe records")
    rec = data_ofp.read_pvd(files[0])
    assert rec.counts.shape[1] == 2
    assert rec.n_samples == data_ofp.EXPECTED_SAMPLES
    assert rec.fs_hz == pytest.approx(50000.0)
    assert rec.duration_s == pytest.approx(10.48576, rel=1e-6)
    assert rec.header.get("测点间距") == "1.7"


def test_ofp_location_parsing():
    files = data_ofp.iter_pvd_files()
    if not files:
        pytest.skip("no probe records")
    height, radial, velocity = data_ofp.parse_location(files[0])
    assert height in data_ofp.BED_HEIGHTS_CM
    assert radial in data_ofp.RADIAL_POSITIONS
    assert 0.0 < velocity <= 2.0


def test_ofp_calibration_is_monotone():
    files = data_ofp.iter_pvd_files()
    if not files:
        pytest.skip("no probe records")
    rec = data_ofp.read_pvd(files[0], max_samples=1000)
    a = rec.solids_fraction(data_ofp.DEFAULT_CALIBRATION)
    b = rec.solids_fraction(data_ofp.ALTERNATIVE_CALIBRATION)
    assert np.all(a > 0) and np.all(b > 0)
    # the two recovered constant sets must agree to within a few per-cent
    assert np.median(np.abs(a - b) / a) < 0.25
    # monotone in voltage
    order = np.argsort(rec.counts[:, 1])
    assert np.all(np.diff(a[order]) >= 0)


@pytest.mark.slow
def test_ofp_coverage_matches_the_directory_tree():
    if not paths.ofp_dir().is_dir():
        pytest.skip("no probe data")
    coverage = data_ofp.coverage_table()
    n_leaves = len(list(p for p in paths.ofp_dir().glob("*/*/*") if p.is_dir()))
    assert len(coverage) == n_leaves
    assert coverage["n_records"].sum() == len(data_ofp.iter_pvd_files())
