"""Multi-band spectral analysis, deep profiling, and stereo width computation."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import librosa

# Load trained section classifier if available
_CLASSIFIER = None
_CLASSIFIER_PATH = Path(__file__).resolve().parents[2] / "models" / "section_classifier.pkl"
if _CLASSIFIER_PATH.exists():
    try:
        with open(_CLASSIFIER_PATH, "rb") as _f:
            _CLASSIFIER = pickle.load(_f)
    except Exception:
        pass

BAND_EDGES_HZ = [20, 60, 250, 500, 2000, 4000, 8000, 16000, 20000]
BAND_NAMES = ["sub", "low", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]


# --------------- 64-band mel profiles (for search ranking) ---------------

def _compute_log_mel(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """Compute log-mel spectrogram (shared across eq profile functions)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, power=2.0, center=False)
    return np.log1p(S)


def compute_eq_profiles(y: np.ndarray, sr: int, n_bands: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all three 64-band eq profiles (avg, peak, variance) from a single
    mel spectrogram. Returns (avg, peak, variance) as L2-normalized float32 vectors.
    """
    M = _compute_log_mel(y, sr, n_bands)
    k_f = np.ones(3, dtype=np.float32) / 3

    # Average profile with temporal + spectral smoothing
    M_smooth = M
    if M.shape[1] >= 3:
        k_t = np.ones(5, dtype=np.float32) / 5
        M_smooth = np.apply_along_axis(lambda x: np.convolve(x, k_t, mode="same"), axis=1, arr=M)
    v_avg = M_smooth.mean(axis=1)
    if v_avg.size >= 3:
        v_avg = np.convolve(v_avg, k_f, mode="same")
    v_avg = v_avg.astype(np.float32, copy=False)
    v_avg /= (np.linalg.norm(v_avg) + 1e-12)

    # Peak profile
    v_peak = M.max(axis=1)
    if v_peak.size >= 3:
        v_peak = np.convolve(v_peak, k_f, mode="same")
    v_peak = v_peak.astype(np.float32, copy=False)
    v_peak /= (np.linalg.norm(v_peak) + 1e-12)

    # Variance profile
    v_var = M.var(axis=1).astype(np.float32, copy=False)
    v_var /= (np.linalg.norm(v_var) + 1e-12)

    return v_avg, v_peak, v_var


def compute_eq_profile(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """64-band log-mel average profile. For standalone use (e.g. search)."""
    avg, _, _ = compute_eq_profiles(y, sr, n_bands)
    return avg


def compute_eq_profile_peak(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """64-band log-mel peak profile. For standalone use."""
    _, peak, _ = compute_eq_profiles(y, sr, n_bands)
    return peak


def compute_eq_profile_variance(y: np.ndarray, sr: int, n_bands: int = 64) -> np.ndarray:
    """64-band log-mel variance profile. For standalone use."""
    _, _, var = compute_eq_profiles(y, sr, n_bands)
    return var


# --------------- Per-band crest & transient density (8-band) ---------------

def compute_band_crest_and_transient_density(
    y: np.ndarray, sr: int
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute per-band crest factor AND transient density from a single STFT.
    Returns (band_crest, band_transient_density).
    """
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    duration = len(y) / sr

    crest_result = {}
    td_result = {}
    for i, name in enumerate(BAND_NAMES):
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            crest_result[name] = 0.0
            td_result[name] = 0.0
            continue

        band = S[mask, :]

        # Crest factor
        rms = float(np.sqrt(np.mean(band ** 2)))
        peak = float(band.max())
        crest_result[name] = round(float(20 * np.log10((peak + 1e-9) / (rms + 1e-9))), 2)

        # Transient density
        if duration >= 0.1:
            band_power = (band ** 2).sum(axis=0)
            band_db = librosa.power_to_db(band_power[np.newaxis, :], ref=np.max)
            onset_env = librosa.onset.onset_strength(S=band_db, sr=sr, hop_length=hop_length)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='time'
            )
            td_result[name] = round(len(onsets) / duration, 2)
        else:
            td_result[name] = 0.0

    return crest_result, td_result


def compute_band_crest(y: np.ndarray, sr: int) -> dict[str, float]:
    """Per-band crest factor. For standalone use."""
    crest, _ = compute_band_crest_and_transient_density(y, sr)
    return crest


def compute_band_transient_density(y: np.ndarray, sr: int) -> dict[str, float]:
    """Per-band transient density. For standalone use."""
    _, td = compute_band_crest_and_transient_density(y, sr)
    return td


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


def compute_stem_energies(
    stems: dict,
    sr: int,
    start_sample: int,
    end_sample: int,
) -> dict[str, float]:
    """Compute RMS in dBFS for each stem in a given sample range."""
    result = {}
    for name in ("drums", "bass", "vocals", "other"):
        if name in stems:
            seg = stems[name][start_sample:end_sample]
            if seg.shape[0] > 0:
                rms = float(np.sqrt(np.mean(seg ** 2)))
                result[name] = round(float(20 * np.log10(rms + 1e-12)), 2)
            else:
                result[name] = -96.0
        else:
            result[name] = -96.0
    return result


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
    stem_energies: dict | None = None,
    stem_percentiles: dict | None = None,
    rms_delta: float = 0.0,
    rms_trend: float = 0.0,
) -> tuple[str, float]:
    """
    Section labeling using stem energy, energy trajectory, and relative thresholds.
    rms_delta: dB change vs previous section (positive = rising).
    rms_trend: dB change over a longer window (3 sections), positive = sustained rise.
    """
    p = track_rms_percentiles or {}
    rms_p25, rms_p50, rms_p75 = p.get("p25", -24), p.get("p50", -18), p.get("p75", -12)
    rms_range = max(rms_p75 - rms_p25, 1e-6)
    energy_norm = (rms_dbfs - rms_p25) / rms_range

    # Stem signals -- normalize using p5/p95 range for full dynamic picture
    se = stem_energies or {}
    sp = stem_percentiles or {}

    def stem_norm(name):
        val = se.get(name, -96)
        sp_data = sp.get(name, {})
        s_lo = sp_data.get("p5", -40)
        s_hi = sp_data.get("p95", -10)
        s_range = s_hi - s_lo
        if s_range < 1.0:
            return None  # stem is essentially constant
        return (val - s_lo) / s_range

    drums_n = stem_norm("drums")
    bass_n = stem_norm("bass")
    vocals_n = stem_norm("vocals")
    other_n = stem_norm("other")

    has_vocals = vocals_n is not None and vocals_n > 0.4
    has_strong_vocals = vocals_n is not None and vocals_n > 0.7
    has_drums = drums_n is not None and drums_n > 0.3
    has_strong_drums = drums_n is not None and drums_n > 0.7
    has_bass = bass_n is not None and bass_n > 0.3
    has_strong_bass = bass_n is not None and bass_n > 0.7
    no_drums = drums_n is not None and drums_n < 0.15
    no_bass = bass_n is not None and bass_n < 0.15

    # Trajectory
    is_rising = rms_delta > 1.0
    is_falling = rms_delta < -1.0
    trend_rising = rms_trend > 2.0
    trend_falling = rms_trend < -2.0

    scores = {}

    # --- DROP: high energy, strong drums + bass ---
    drop_s = 0.0
    if energy_norm > 0.5:
        drop_s += 0.2
    if energy_norm > 0.8:
        drop_s += 0.2
    if has_strong_drums:
        drop_s += 0.15
    elif has_drums:
        drop_s += 0.08
    if has_strong_bass:
        drop_s += 0.15
    elif has_bass:
        drop_s += 0.08
    if not is_rising and not is_falling and energy_norm > 0.5:
        drop_s += 0.1  # stable high = sustained drop
    if has_strong_vocals:
        drop_s -= 0.1  # vocals in drop less common in EDM
    scores["drop"] = max(0, drop_s)

    # --- BREAKDOWN: stripped arrangement, energy drop-off ---
    # Key: breakdowns require BOTH low energy AND stripped arrangement (no drums)
    # Moderate energy with drums = NOT a breakdown (probably verse or buildup)
    bd_s = 0.0
    if energy_norm < 0.25:
        bd_s += 0.25
    elif energy_norm < 0.4:
        bd_s += 0.1
    if no_drums:
        bd_s += 0.25
    elif drums_n is not None and drums_n < 0.2:
        bd_s += 0.1
    if no_bass:
        bd_s += 0.1
    if is_falling or trend_falling:
        bd_s += 0.1
    # Penalize breakdown when arrangement isn't actually stripped
    if has_drums and energy_norm > 0.3:
        bd_s -= 0.15
    if has_vocals:
        bd_s -= 0.2  # vocals present = much more likely verse
    if has_strong_vocals:
        bd_s -= 0.15  # strong vocals = definitely not breakdown
    scores["breakdown"] = max(0, bd_s)

    # --- BUILDUP: sustained rising energy ---
    bu_s = 0.0
    if is_rising:
        bu_s += 0.2
    if trend_rising:
        bu_s += 0.25  # sustained rise is the strongest buildup signal
    if 0.15 < energy_norm < 0.85:
        bu_s += 0.1
    if has_drums and not has_strong_drums:
        bu_s += 0.1
    if energy_norm > 0.85:
        bu_s -= 0.15  # if already at peak, probably drop not buildup
    scores["buildup"] = max(0, bu_s)

    # --- VERSE: vocals present, moderate energy ---
    # Verse is primarily defined by vocal presence. Without vocals, verse is unlikely.
    verse_s = 0.0
    if has_vocals:
        verse_s += 0.3
    if has_strong_vocals:
        verse_s += 0.2
    if 0.1 < energy_norm < 0.7:
        verse_s += 0.1
    if not has_strong_drums:
        verse_s += 0.05
    # Without meaningful vocal stem data, verse score stays very low
    scores["verse"] = max(0, verse_s)

    # --- INTRO: early in the track ---
    # For EDM, the first ~15% is often intro/mix-in regardless of energy
    intro_s = 0.0
    if position_ratio < 0.15:
        intro_s += 0.3 * max(0, (0.15 - position_ratio) / 0.15)
    if energy_norm < 0.4 and position_ratio < 0.15:
        intro_s += 0.2
    elif position_ratio < 0.08:
        intro_s += 0.15  # very start gets intro boost even at high energy
    if no_drums and position_ratio < 0.15:
        intro_s += 0.1
    scores["intro"] = max(0, intro_s)

    # --- OUTRO: final portion of track ---
    # Outro detection must balance: EDM mix-outs are full-energy drops at the end.
    # But legitimate drops also happen at 55-75% position. We use:
    # - Strong position signal for the very end (>85%)
    # - Energy decline + position for the 70-85% range
    # - Position alone is NOT enough to override a clear drop at 60-75%
    outro_s = 0.0
    if position_ratio > 0.85:
        outro_s += 0.4
    elif position_ratio > 0.7:
        pos_weight = (position_ratio - 0.7) / 0.15
        outro_s += 0.15 * pos_weight
    if position_ratio > 0.92:
        outro_s += 0.2
    # Declining energy in the tail is the strongest signal
    if trend_falling and position_ratio > 0.6:
        outro_s += 0.2
    if is_falling and position_ratio > 0.7:
        outro_s += 0.15
    if energy_norm < 0.3 and position_ratio > 0.7:
        outro_s += 0.15
    scores["outro"] = max(0, outro_s)

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    confidence = min(1.0, max(0.1, margin * 1.5 + best_score * 0.4))

    return best_label, round(confidence, 2)


def classify_section_label(
    energy_norm: float,
    position_ratio: float,
    drums_n: float,
    bass_n: float,
    vocals_n: float,
    other_n: float,
    rms_delta: float,
    rms_trend: float,
    future_delta: float,
    crest_db: float,
    hf_perc_ratio: float,
    flatness: float,
) -> tuple[str, float]:
    """Use trained classifier for section labeling. Falls back to rule-based if no model."""
    if _CLASSIFIER is None:
        return "unknown", 0.0

    feature_vec = np.array([[
        energy_norm,
        position_ratio,
        drums_n,
        bass_n,
        vocals_n,
        other_n,
        rms_delta,
        rms_trend,
        future_delta,
        crest_db,
        hf_perc_ratio,
        flatness,
        1.0 if drums_n > 0.7 else 0.0,
        1.0 if bass_n > 0.7 else 0.0,
        1.0 if vocals_n > 0.4 else 0.0,
        1.0 if drums_n < 0.15 else 0.0,
        position_ratio * energy_norm,
    ]])

    label = _CLASSIFIER.predict(feature_vec)[0]
    proba = _CLASSIFIER.predict_proba(feature_vec)[0]
    confidence = float(max(proba))
    return str(label), round(confidence, 2)