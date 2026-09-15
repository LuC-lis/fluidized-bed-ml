"""
Feature extraction from a single optical-probe record.

A dual-tip optical probe inside a fluidized bed sees the passage of bubbles as
sharp drops in the local solids fraction: the *rate* and the *depth* of those
drops are direct regime indicators.  The features below capture exactly that --
distribution shape, crossing rate, spectral content -- instead of the single
time-mean voltage that the legacy pipeline computed.

Frequency context: the probe samples at 50 kHz, so a 12-level db4 decomposition
puts Level1 at 12.5-25 kHz and Level12 at 6-12 Hz.  Bubble passage frequencies in
a bubbling/turbulent bed live in the 1-10 Hz range, i.e. the deepest levels --
which is what one would expect to carry the regime information.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import features_wavelet

#: Sampling rate of the probe (Hz), fixed by the instrument.
PROBE_FS_HZ = 50000.0
PROBE_WAVELET_CONFIG = features_wavelet.WaveletConfig(wavelet="db4", level=12, mode="symmetric")


def probe_features(counts_1d: np.ndarray, fs_hz: float = PROBE_FS_HZ,
                   config: features_wavelet.WaveletConfig = PROBE_WAVELET_CONFIG) -> Dict[str, float]:
    """Full feature set for one probe channel.

    Parameters
    ----------
    counts_1d
        Raw probe signal in mV.
    fs_hz
        Sampling rate.
    config
        Wavelet settings (db4 / 12 levels by default).
    """
    x = np.asarray(counts_1d, dtype=np.float64).ravel()
    if x.size < 16:
        raise ValueError("signal too short")
    xd = x - x.mean()

    features: Dict[str, float] = {
        "p_mean": float(x.mean()),
        "p_std": float(x.std(ddof=1)),
        "p_min": float(x.min()),
        "p_max": float(x.max()),
        "p_ptp": float(np.ptp(x)),
        "p_skew": float(sp_stats.skew(x)),
        "p_kurtosis": float(sp_stats.kurtosis(x)),
        "p_q05": float(np.percentile(x, 5)),
        "p_q25": float(np.percentile(x, 25)),
        "p_median": float(np.median(x)),
        "p_q75": float(np.percentile(x, 75)),
        "p_q95": float(np.percentile(x, 95)),
        "p_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
    }

    # --- crossing statistics: a proxy for bubble-passage frequency -----------
    thr = float(np.std(xd))
    features["p_cross_rate"] = float(np.mean(np.abs(np.diff(np.signbit(xd - thr)))) )
    features["p_frac_below_neg1sigma"] = float(np.mean(xd < -thr))
    features["p_frac_above_pos1sigma"] = float(np.mean(xd > thr))
    features["p_mean_dip_depth"] = float(x.mean() - x[x < x.mean() - thr].mean()) \
        if np.any(x < x.mean() - thr) else 0.0

    # --- spectral content ---------------------------------------------------
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    psd = np.abs(np.fft.rfft(xd * np.hanning(x.size))) ** 2
    total = psd.sum()
    if total > 0:
        features["p_dominant_freq_hz"] = float(freqs[int(np.argmax(psd[1:])) + 1])
        centroid = float((freqs * psd).sum() / total)
        features["p_spectral_centroid_hz"] = centroid
        for lo, hi, name in ((0, 2, "p_band_0_2hz"),
                             (2, 5, "p_band_2_5hz"),
                             (5, 20, "p_band_5_20hz"),
                             (20, 100, "p_band_20_100hz"),
                             (100, fs_hz / 2, "p_band_gt100hz")):
            mask = (freqs >= lo) & (freqs < hi)
            features[name] = float(psd[mask].sum() / total)
    else:  # pragma: no cover - degenerate signal
        features["p_dominant_freq_hz"] = 0.0
        features["p_spectral_centroid_hz"] = 0.0
        for name in ("p_band_0_2hz", "p_band_2_5hz", "p_band_5_20hz",
                     "p_band_20_100hz", "p_band_gt100hz"):
            features[name] = 0.0

    # --- wavelet band energies (same convention as the pressure features) ---
    energies = features_wavelet.band_energies(x, config)
    total_e = float(energies.sum())
    for i, value in enumerate(energies, start=1):
        features[f"Level{i}_ratio"] = float(value / total_e) if total_e > 0 else 0.0
    return features


def records_to_features(paths: List, progress: bool = True, max_samples: int | None = None) -> pd.DataFrame:
    """Feature table for a list of ``.pvd`` paths (one row per record)."""
    from . import data_ofp

    if progress:
        try:
            from tqdm import tqdm

            paths = tqdm(paths, desc="probe features", unit="file")
        except ImportError:  # pragma: no cover
            pass

    rows: List[dict] = []
    for path in paths:
        try:
            rec = data_ofp.read_pvd(path, max_samples=max_samples)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(dict(file=str(path), error=f"{type(exc).__name__}"))
            continue
        row = probe_features(rec.counts[:, 1], rec.fs_hz)
        row.update(
            dict(
                file=path.name,
                bed_height_cm=rec.bed_height_cm,
                radial_position=rec.radial_position,
                velocity=rec.velocity,
                n_samples=rec.n_samples,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


#: Regime labels used by the classification script.
BUBBLING_MAX = 0.5
TRANSITION_VELOCITY = 0.6
TURBULENT_MIN = 0.7


def regime_label(velocity: float) -> str:
    """Map a gas velocity to the regime classes claimed by the project poster."""
    if velocity <= BUBBLING_MAX:
        return "bubbling"
    if abs(velocity - TRANSITION_VELOCITY) < 1e-9:
        return "transition"
    return "turbulent"
