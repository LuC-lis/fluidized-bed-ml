"""
Descriptive statistics used as model inputs and as physical indicators.

Kept separate from :mod:`fbfml.features_wavelet` because these are the
"classical" time-domain descriptors (mean, RMS, standard deviation, skewness,
kurtosis, zero-crossing rate, dominant frequency) that the original feature
table also carried (``Average`` / ``StandardDeviation``).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats as sp_stats


def basic_stats(signal: Sequence[float]) -> Dict[str, float]:
    """Mean, |mean|, RMS, standard deviation, skewness and kurtosis."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty signal")
    return {
        "mean": float(np.mean(x)),
        "mean_abs": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "skewness": float(sp_stats.skew(x)) if x.size > 2 else 0.0,
        "kurtosis": float(sp_stats.kurtosis(x)) if x.size > 3 else 0.0,
        "peak_to_peak": float(np.ptp(x)),
    }


def dominant_frequency_hz(signal: Sequence[float], fs_hz: float) -> float:
    """Frequency of the largest PSD peak (Hann-windowed periodogram)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    if x.size < 16:
        return float("nan")
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    if psd.size < 2:
        return float("nan")
    return float(freqs[int(np.argmax(psd[1:])) + 1])


def band_power_ratio(signal: Sequence[float], fs_hz: float,
                     band: tuple, ) -> float:
    """Fraction of the signal power inside ``band = (f_low, f_high)`` Hz."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    if x.size < 16:
        return float("nan")
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    total = psd.sum()
    return float(psd[mask].sum() / total) if total > 0 else float("nan")


def rolling_std_trend(signal: Sequence[float], fs_hz: float, window_s: float = 1.0) -> np.ndarray:
    """Standard deviation inside consecutive, non-overlapping windows."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    w = max(1, int(fs_hz * window_s))
    n = x.size // w
    if n == 0:
        return np.array([], dtype=float)
    return x[: n * w].reshape(n, w).std(axis=1, ddof=1)
