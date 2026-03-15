"""Multi-band spectral analysis, deep profiling, and stereo width computation."""
from __future__ import annotations

import numpy as np
import librosa

BAND_EDGES_HZ = [20, 60, 250, 500, 2000, 4000, 8000, 16000, 20000]
BAND_NAMES = ["sub", "low", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]


# --------------- 64-band mel profiles (for search ranking) ---------------

def compute_eq_profile(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """
    64-band log-mel average profile with temporal and spectral smoothing.
    Returns L2-normalized float32 vector. Used for search ranking.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, power=2.0, center=False)
    M = np.log1p(S)
    if M.shape[1] >= 3:
        k_t = np.ones(5, dtype=np.float32) / 5
        M = np.apply_along_axis(lambda x: np.convolve(x, k_t, mode="same"), axis=1, arr=M)
    v = M.mean(axis=1)
    if v.size >= 3:
        k_f = np.ones(3, dtype=np.float32) / 3
        v = np.convolve(v, k_f, mode="same")
    v = v.astype(np.float32, copy=False)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def compute_eq_profile_peak(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """
    64-band log-mel PEAK profile -- max energy per band across time.
    Captures headroom and arrangement density (loud transient events).
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, power=2.0, center=False)
    M = np.log1p(S)
    v = M.max(axis=1)
    if v.size >= 3:
        k_f = np.ones(3, dtype=np.float32) / 3
        v = np.convolve(v, k_f, mode="same")
    v = v.astype(np.float32, copy=False)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def compute_eq_profile_variance(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """
    64-band log-mel VARIANCE profile -- how much each band moves over time.
    High variance = rhythmic content (hats, percussion). Low = sustained (pads, bass).
    Distinguishes 1/16th hats from 1/8th hats: same avg energy but different variance.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, power=2.0, center=False)
    M = np.log1p(S)
    v = M.var(axis=1)
    v = v.astype(np.float32, copy=False)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


# --------------- Per-band crest & transient density (8-band) ---------------

def compute_band_crest(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Per-band crest factor (peak-to-RMS ratio in dB) across 8 frequency bands.
    Low crest = compressed/limited. High crest = dynamic/unprocessed.
    """
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    result = {}
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band = S[mask, :]
            rms = float(np.sqrt(np.mean(band ** 2)))
            peak = float(band.max())
            crest = float(20 * np.log10((peak + 1e-9) / (rms + 1e-9)))
            result[name] = round(crest, 2)
        else:
            result[name] = 0.0
    return result


def compute_band_transient_density(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Per-band transient density (onset events per second) across 8 frequency bands.
    Distinguishes arrangement density from EQ balance: 1/16th hats produce
    high transient density in brilliance/air bands, while 1/8th hats produce lower.
    """
    duration = len(y) / sr
    if duration < 0.1:
        return {name: 0.0 for name in BAND_NAMES}

    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    result = {}
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_energy = S[mask, :].sum(axis=0)
            onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(band_energy[np.newaxis, :] ** 2), sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
            result[name] = round(len(onsets) / duration, 2)
        else:
            result[name] = 0.0
    return result


# --------------- Mastering state detection ---------------

def detect_mastering_state(y: np.ndarray, sr: int) -> str:
    """
    Classify a track as 'mastered', 'pre_master', or 'unknown' based on
    overall crest factor, peak level, and loudness histogram characteristics.
    """
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_db = float(20 * np.log10((peak + 1e-9) / (rms + 1e-9)))
    peak_dbfs = float(20 * np.log10(peak + 1e-9))

    # Compute short-term loudness histogram
    hop = int(0.4 * sr)
    n_frames = max(1, len(y) // hop)
    frame_rms = []
    for i in range(n_frames):
        frame = y[i * hop:(i + 1) * hop]
        if frame.size > 0:
            frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))
    frame_dbfs = [20 * np.log10(r + 1e-9) for r in frame_rms]
    loudness_range = max(frame_dbfs) - min(frame_dbfs) if frame_dbfs else 100

    # Mastered tracks: low crest (<8dB), peak near 0dBFS (>-1dB), narrow loudness range (<12dB)
    mastered_signals = 0
    if crest_db < 8.0:
        mastered_signals += 1
    if peak_dbfs > -1.0:
        mastered_signals += 1
    if loudness_range < 12.0:
        mastered_signals += 1

    if mastered_signals >= 2:
        return "mastered"
    if crest_db > 14.0 or peak_dbfs < -3.0:
        return "pre_master"
    return "unknown"


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
    track_rms_percentiles: dict | None = None,
    track_onset_percentiles: dict | None = None,
    track_hf_percentiles: dict | None = None,
) -> tuple[str, float]:
    """
    Section labeling using relative thresholds within the track's own dynamic range.
    Returns (label, confidence) where confidence is 0.0-1.0.

    If percentiles are provided, thresholds are relative to the track.
    Otherwise falls back to absolute thresholds (less accurate).
    """
    # Position-based labels (high confidence at extremes)
    if position_ratio < 0.03:
        return "intro", 0.9
    if position_ratio > 0.95:
        return "outro", 0.9

    # Use relative thresholds if available, else absolute fallbacks
    if track_rms_percentiles:
        is_high_energy = rms_dbfs >= track_rms_percentiles.get("p75", -12)
        is_low_energy = rms_dbfs <= track_rms_percentiles.get("p25", -24)
        energy_extremity = abs(rms_dbfs - track_rms_percentiles.get("p50", -18)) / max(
            track_rms_percentiles.get("p75", -12) - track_rms_percentiles.get("p25", -24), 1e-6
        )
    else:
        is_high_energy = rms_dbfs > -12
        is_low_energy = rms_dbfs < -24
        energy_extremity = 0.5

    if track_onset_percentiles:
        is_dense = onset_density >= track_onset_percentiles.get("p75", 0.5)
    else:
        is_dense = onset_density > 0.5

    if track_hf_percentiles:
        is_percussive = hf_perc_ratio >= track_hf_percentiles.get("p75", 0.04)
    else:
        is_percussive = hf_perc_ratio > 0.04

    # Score each label candidate
    scores = {}

    # Drop: high energy + dense + percussive
    drop_score = (0.4 * float(is_high_energy) + 0.3 * float(is_dense) + 0.3 * float(is_percussive))
    if is_high_energy and is_dense and is_percussive:
        drop_score = min(1.0, drop_score + 0.3)
    scores["drop"] = drop_score

    # Breakdown: low energy, not dense, mid-to-late position
    bd_score = (0.4 * float(is_low_energy) + 0.3 * float(not is_dense) + 0.3 * float(position_ratio > 0.3))
    scores["breakdown"] = bd_score

    # Buildup: rising energy, moderate density, not the highest energy
    bu_score = 0.3 * float(not is_low_energy and not is_high_energy) + 0.3 * float(is_dense) + 0.2 * float(not is_percussive)
    if position_ratio > 0.1 and not is_high_energy and is_dense:
        bu_score += 0.2
    scores["buildup"] = bu_score

    # Verse: moderate energy, moderate density
    verse_score = 0.3 * float(not is_high_energy and not is_low_energy) + 0.2 * float(not is_dense)
    if position_ratio < 0.4:
        verse_score += 0.15
    scores["verse"] = verse_score

    # Intro: low-moderate energy near the start
    intro_score = 0.5 * max(0, 0.15 - position_ratio) / 0.15 + 0.3 * float(is_low_energy)
    scores["intro"] = intro_score

    # Outro: low-moderate energy near the end
    outro_score = 0.5 * max(0, position_ratio - 0.85) / 0.15 + 0.3 * float(is_low_energy)
    scores["outro"] = outro_score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    # Confidence based on how clearly the top label wins
    confidence = min(1.0, max(0.1, margin * 2 + best_score * 0.3))

    return best_label, round(confidence, 2)