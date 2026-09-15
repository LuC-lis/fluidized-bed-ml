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


# ---------------------------------------------------------------------------
# bubble-passage statistics
# ---------------------------------------------------------------------------
def bubble_features(counts_1d: np.ndarray, fs_hz: float = PROBE_FS_HZ,
                    n_sigma: float = 0.5, psd_band: tuple = (0.5, 20.0),
                    target_fs_hz: float = 50.0) -> Dict[str, float]:
    """Bubble-passage descriptors from one probe channel.

    A dual-tip optical probe sees a bubble as a *dip* in the local solids
    fraction, so bubble passage is a sequence of dips and the two quantities that
    matter physically are **how often** they occur and **how long** they last.

    The probe samples at 50 kHz but bubble passage lives below ~20 Hz, so the raw
    signal is first **block-averaged down to** ``target_fs_hz``.  Without that
    step, single-sample noise excursions are counted as bubbles and the rate comes
    out in the hundreds of hertz, which is physically impossible.

    Parameters
    ----------
    counts_1d
        Probe signal (mV).
    fs_hz
        Sampling rate of ``counts_1d``.
    n_sigma
        Depth threshold for "inside a bubble", in standard deviations below the
        mean of the decimated signal.
    psd_band
        Frequency window for the spectral estimate of the passage rate.
    target_fs_hz
        Decimation target (50 Hz keeps everything up to Nyquist = 25 Hz).

    Returns
    -------
    dict
        ``bubble_rate_hz``, ``bubble_duty_cycle``, ``bubble_dip_depth_norm``,
        ``bubble_duration_s``, ``bubble_freq_psd_hz``, ``bubble_decorrelation_s``,
        ``bubble_n_events``.
    """
    x = np.asarray(counts_1d, dtype=np.float64).ravel()
    if x.size < 64:
        raise ValueError("signal too short")

    # --- anti-aliased decimation -------------------------------------------
    factor = max(1, int(round(fs_hz / target_fs_hz)))
    usable = (x.size // factor) * factor
    y = x[:usable].reshape(-1, factor).mean(axis=1)
    fs = fs_hz / factor

    yd = y - y.mean()
    sigma = float(yd.std(ddof=1)) or 1.0
    threshold = -n_sigma * sigma

    # --- discrete dip events ------------------------------------------------
    below = yd < threshold
    transitions = np.diff(below.astype(np.int8))
    starts = np.flatnonzero(transitions == 1) + 1
    stops = np.flatnonzero(transitions == -1) + 1
    if below[0]:
        starts = np.r_[0, starts]
    if below[-1]:
        stops = np.r_[stops, below.size]
    n_events = int(min(starts.size, stops.size))
    duration_s = y.size / fs
    dwell = (stops[:n_events] - starts[:n_events]) / fs if n_events else np.array([])
    depths = yd[starts[:n_events]] / sigma if n_events else np.array([])

    # --- spectral estimate of the passage rate ------------------------------
    freqs = np.fft.rfftfreq(y.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(yd * np.hanning(y.size))) ** 2
    mask = (freqs >= psd_band[0]) & (freqs <= psd_band[1])
    f_dom = float(freqs[mask][int(np.argmax(psd[mask]))]) if mask.any() else float("nan")

    # --- decorrelation time (first zero crossing of the autocorrelation) ----
    n_lag = min(int(5 * fs), y.size // 2)
    segment = yd[: 2 * n_lag]
    ac = np.correlate(segment, segment, mode="full")[n_lag - 1:]
    ac = ac / ac[0] if ac[0] > 0 else ac
    zeros = np.flatnonzero(np.diff(np.signbit(ac)))
    tau = float(zeros[0] / fs) if zeros.size else float("nan")

    return dict(
        bubble_rate_hz=float(n_events / duration_s),
        bubble_n_events=int(n_events),
        bubble_duty_cycle=float(below.mean()),
        bubble_dip_depth_norm=float(depths.mean()) if depths.size else 0.0,
        bubble_duration_s=float(np.median(dwell)) if dwell.size else 0.0,
        bubble_freq_psd_hz=f_dom,
        bubble_decorrelation_s=tau,
        bubble_decimated_fs_hz=fs,
    )


def records_to_bubble_features(paths: List, progress: bool = True) -> pd.DataFrame:
    """Bubble-passage features for a list of ``.pvd`` paths (one row per record)."""
    from . import data_ofp

    if progress:
        try:
            from tqdm import tqdm

            paths = tqdm(paths, desc="bubble features", unit="file")
        except ImportError:  # pragma: no cover
            pass

    rows: List[dict] = []
    for path in paths:
        try:
            rec = data_ofp.read_pvd(path)
            row = bubble_features(rec.counts[:, 1], rec.fs_hz)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(dict(file=str(path), error=f"{type(exc).__name__}"))
            continue
        row.update(dict(file=path.name, bed_height_cm=rec.bed_height_cm,
                        radial_position=rec.radial_position, velocity=rec.velocity))
        rows.append(row)
    return pd.DataFrame(rows)
