"""Multi-band spectral analysis and stereo width computation."""
from __future__ import annotations

import numpy as np
import librosa

BAND_EDGES_HZ = [20, 60, 250, 500, 2000, 4000, 8000, 16000, 20000]
BAND_NAMES = ["sub", "low", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]


def compute_band_energies(y: np.ndarray, sr: int) -> dict[str, float]:
    """Compute average energy in 8 frequency bands (dB). Returns dict of band_name -> dB."""
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    result = {}
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_energy = float(S[mask, :].mean())
            result[name] = round(float(10 * np.log10(band_energy + 1e-12)), 2)
        else:
            result[name] = -96.0
    return result


def compute_spectral_comparison(
    query_bands: dict[str, float],
    ref_bands: dict[str, float],
) -> dict[str, float]:
    """Compare two band energy profiles. Returns per-band delta in dB (query - ref)."""
    return {name: round(query_bands.get(name, -96) - ref_bands.get(name, -96), 2) for name in BAND_NAMES}


def compute_energy_curve(y: np.ndarray, sr: int, hop_s: float = 0.5) -> dict:
    """
    Compute per-window energy features over the full track.
    Returns dict with arrays: times, lufs, centroid, onset_density, low_ratio.
    """
    hop_samples = int(hop_s * sr)
    n_windows = max(1, len(y) // hop_samples)
    times = []
    lufs_vals = []
    centroid_vals = []
    onset_vals = []
    low_ratio_vals = []

    for i in range(n_windows):
        start = i * hop_samples
        end = min(start + hop_samples, len(y))
        seg = y[start:end]
        if seg.size == 0:
            continue

        t = float(start) / sr
        times.append(round(t, 2))

        rms = float(np.sqrt(np.mean(seg ** 2)))
        lufs_vals.append(round(float(20 * np.log10(rms + 1e-9)), 2))

        cent = librosa.feature.spectral_centroid(y=seg, sr=sr)
        centroid_vals.append(round(float(cent.mean()), 1))

        onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
        onset_vals.append(round(float(onset_env.mean()), 3))

        S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = freqs < 250
        total_e = float(S.sum()) + 1e-12
        low_e = float(S[low_mask, :].sum())
        low_ratio_vals.append(round(low_e / total_e, 4))

    return {
        "times": times,
        "lufs": lufs_vals,
        "centroid": centroid_vals,
        "onset_density": onset_vals,
        "low_ratio": low_ratio_vals,
    }


def compute_stereo_features(y_stereo: np.ndarray, sr: int) -> dict:
    """
    Compute stereo width features from a stereo signal (2, N).
    Returns correlation, mid_side_ratio, and per-band width.
    """
    if y_stereo.ndim == 1:
        return {"correlation": 1.0, "mid_side_ratio": 1.0, "width_by_band": {n: 0.0 for n in BAND_NAMES}}

    if y_stereo.shape[0] != 2:
        y_stereo = y_stereo.T
    if y_stereo.shape[0] != 2:
        return {"correlation": 1.0, "mid_side_ratio": 1.0, "width_by_band": {n: 0.0 for n in BAND_NAMES}}

    left, right = y_stereo[0], y_stereo[1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    corr_num = float(np.sum(left * right))
    corr_den = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + 1e-12)
    correlation = round(corr_num / corr_den, 4)

    mid_energy = float(np.sum(mid ** 2))
    side_energy = float(np.sum(side ** 2))
    ms_ratio = round(mid_energy / (mid_energy + side_energy + 1e-12), 4)

    S_mid = np.abs(librosa.stft(mid, n_fft=4096, hop_length=512)) ** 2
    S_side = np.abs(librosa.stft(side, n_fft=4096, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    width_by_band = {}
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            m_e = float(S_mid[mask, :].sum())
            s_e = float(S_side[mask, :].sum())
            width_by_band[name] = round(s_e / (m_e + s_e + 1e-12), 4)
        else:
            width_by_band[name] = 0.0

    return {
        "correlation": correlation,
        "mid_side_ratio": ms_ratio,
        "width_by_band": width_by_band,
    }


def detect_section_label(
    rms_dbfs: float,
    onset_density: float,
    hf_perc_ratio: float,
    flatness: float,
    position_ratio: float,
    bpm: float | None = None,
) -> str:
    """
    Heuristic section labeling based on audio features + position.
    Returns one of: intro, verse, buildup, drop, breakdown, outro.
    """
    if position_ratio < 0.05:
        return "intro"
    if position_ratio > 0.92:
        return "outro"

    is_high_energy = rms_dbfs > -12
    is_dense = onset_density > 0.5
    is_percussive = hf_perc_ratio > 0.04

    if is_high_energy and is_dense and is_percussive:
        return "drop"

    if not is_high_energy and not is_dense:
        if position_ratio > 0.4:
            return "breakdown"
        return "verse"

    if is_high_energy and not is_percussive:
        return "buildup"

    if is_dense and not is_high_energy:
        return "buildup"

    return "verse"