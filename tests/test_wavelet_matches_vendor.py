"""
Regression test for the wavelet feature definition.

The definition in ``fbfml.features_wavelet`` must keep reproducing the stored
band-energy tables exactly.  This test fails if it ever drifts.

Run it with the raw data present:

    pytest tests/test_wavelet_matches_vendor.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from fbfml import data_pressure, features_wavelet, paths

VELOCITIES = [0.1, 0.5, 0.9]


def _require_pressure_data():
    if not paths.pressure_dir().is_dir():
        pytest.skip(f"raw pressure data not available at {paths.pressure_dir()}")


@pytest.mark.parametrize("velocity", VELOCITIES)
def test_band_energies_match_stored_tables(velocity: float) -> None:
    """Recomputed db4 12-level energies must equal the stored ``energy_*.xlsx``."""
    _require_pressure_data()
    source = paths.pressure_dir() / f"{velocity:g}.xlsx"
    stored = paths.pressure_dir() / f"energy_{velocity:g}.xlsx"
    if not (source.exists() and stored.exists()):
        pytest.skip(f"missing {source.name} or {stored.name}")

    record = data_pressure.load_pressure_record(source)
    vendor = data_pressure.load_vendor_energy_table(stored)

    for i, channel in enumerate(data_pressure.CHANNEL_COLUMNS):
        mine = features_wavelet.band_energies(record.signals[channel].to_numpy())
        theirs = vendor.iloc[i, :12].to_numpy(dtype=float)
        np.testing.assert_allclose(
            mine, theirs, rtol=1e-9, atol=1e-9,
            err_msg=f"{velocity} m/s, {channel}: band energies differ",
        )
        assert mine.sum() == pytest.approx(float(vendor.iloc[i, 12]), rel=1e-9)


def test_feature_row_is_self_consistent() -> None:
    """Ratios must sum to one and the total must equal the sum of the bands."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=4096)
    row = features_wavelet.feature_row(signal)
    bands = np.array([row[f"Level{i}"] for i in range(1, 13)])
    ratios = np.array([row[f"Level{i}_ratio"] for i in range(1, 13)])
    assert row["TotalEnergy"] == pytest.approx(bands.sum(), rel=1e-12)
    assert ratios.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(bands >= 0)


def test_config_defaults_are_the_validated_ones() -> None:
    config = features_wavelet.WaveletConfig()
    assert (config.wavelet, config.level, config.mode) == ("db4", 12, "symmetric")


def test_scale_frequency_bands_are_descending() -> None:
    high_band = features_wavelet.scale_frequency_hz(1, 500.0)
    low_band = features_wavelet.scale_frequency_hz(6, 500.0)
    assert high_band[0] > low_band[1]  # level 1 covers higher frequencies
