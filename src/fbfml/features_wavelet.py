"""
Wavelet energy features -- definition and exact verification.

Feature definition
------------------
The models were trained on a table whose columns were ``Level1..Level12`` (band
energies), ``Level1_ratio..level12_ratio`` and a ``TotalEnergy``.  The definition
below reproduces the stored band-energy tables
(``Pressure_Energy/energy_0.X.xlsx``) for all 6 pressure channels x 9 gas
velocities with **zero** discrepancy:

    x_d = x - mean(x)                              # DC removal, NOT detrending
    c   = pywt.wavedec(x_d, 'db4', level=12, mode='symmetric')
    #       c == [cA12, cD12, cD11, ..., cD1]
    Level_i  = sum(cD_i ** 2)        for i = 1..12   (cD1 -> Level1)
    Total    = sum(Level_1..Level_12)                (cA12 is NOT included)
    Ratio_i  = Level_i / Total

Notes
-----
* ``pywt.wavedec`` returns the approximation band **first**, so ``c[1:]`` is
  ``[cD12, ..., cD1]`` and must be reversed before being named Level1..Level12.
* The vendor table's ``TotalEnergy`` excludes the level-12 approximation band;
  it equals the sum of the 12 detail bands, not the total signal energy.
* ``mode='symmetric'`` (not ``periodization``/``zero``) is what reproduces the
  stored values.
* Sanity check: ``Total`` is within a few per-cent of the sum of squares of the
  DC-removed signal, because the db4 wavelet is orthonormal up to boundary
  handling.

See ``tests/test_wavelet_matches_vendor.py`` for the regression test.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
import pywt

#: Defaults calibrated against the group's stored tables.  Do not change these
#: without re-running the validation script.
DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 12
DEFAULT_MODE = "symmetric"

LEVEL_NAMES: List[str] = [f"Level{i}" for i in range(1, DEFAULT_LEVEL + 1)]
RATIO_NAMES: List[str] = [f"Level{i}_ratio" for i in range(1, DEFAULT_LEVEL + 1)]


@dataclass(frozen=True)
class WaveletConfig:
    """Configuration of the discrete wavelet decomposition."""

    wavelet: str = DEFAULT_WAVELET
    level: int = DEFAULT_LEVEL
    mode: str = DEFAULT_MODE

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def band_energies(signal: Sequence[float], config: WaveletConfig = WaveletConfig()) -> np.ndarray:
    """Return the 12 detail-band energies of ``signal``, ordered Level1..Level12.

    Parameters
    ----------
    signal
        1-D pressure (or vibration) time series, raw units.
    config
        Wavelet / decomposition settings.

    Returns
    -------
    numpy.ndarray
        Shape ``(config.level,)`` -- ``[Level1, ..., Level12]``.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty signal")
    x = x - x.mean()  # DC removal -- required to match the stored tables
    coeffs = pywt.wavedec(x, config.wavelet, level=config.level, mode=config.mode)
    # coeffs[0] is the approximation (cA_L); the rest are cD_L ... cD_1
    details = coeffs[1:][::-1]  # -> cD_1 ... cD_L  == Level1 ... Level12
    return np.array([float(np.sum(np.square(d))) for d in details], dtype=np.float64)


def feature_row(
    signal: Sequence[float],
    config: WaveletConfig = WaveletConfig(),
    extra: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return one flat feature dictionary for a single signal.

    Keys: ``Level1..Level12``, ``Level1_ratio..Level12_ratio``, ``TotalEnergy``,
    ``Average``, ``StandardDeviation`` (+ anything in ``extra``).
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    energies = band_energies(x, config)
    total = float(energies.sum())
    row: Dict[str, float] = {name: float(v) for name, v in zip(LEVEL_NAMES, energies)}
    ratios = energies / total if total > 0 else np.zeros_like(energies)
    row.update({name: float(v) for name, v in zip(RATIO_NAMES, ratios)})
    row["TotalEnergy"] = total
    row["Average"] = float(np.mean(np.abs(x)))
    row["StandardDeviation"] = float(np.std(x))
    if extra:
        row.update(extra)
    return row


def feature_table(
    signals: Dict[str, Sequence[float]],
    config: WaveletConfig = WaveletConfig(),
) -> "object":
    """Build a :class:`pandas.DataFrame` with one row per entry of ``signals``.

    ``signals`` maps a row label (e.g. ``"ch9"``) to a time series.
    """
    import pandas as pd  # local import keeps this module import-light

    rows = {label: feature_row(sig, config) for label, sig in signals.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "source"
    return df


def scale_frequency_hz(level: int, fs_hz: float, wavelet: str = DEFAULT_WAVELET) -> tuple:
    """Frequency band ``(low, high)`` in Hz covered by detail level ``level``.

    Useful for interpreting which physical phenomenon a band carries; the
    result is exact only in the ideal (unpadded) DWT case.
    """
    nyq = fs_hz / 2.0
    high = nyq / (2 ** (level - 1))
    low = nyq / (2 ** level)
    return low, high
