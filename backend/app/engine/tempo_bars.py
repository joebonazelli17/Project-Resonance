# app/tempo_bars.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import numpy as np

# --- Essentia only: gold standard pipeline ---
try:
    import essentia
    import essentia.standard as es
except Exception as e:
    raise ImportError(
        "Essentia is required for the high-accuracy pipeline. "
        "Install the compiled Essentia (with standard) before running."
    ) from e


@dataclass
class SectionMeta:
    start_s: float
    end_s: float
    bars: int
    bpm: float
    key: Optional[str] = None
    scale: Optional[str] = None  # 'major' / 'minor' (or None)


# -----------------------------
# Core high-accuracy primitives
# -----------------------------

def _beats_essentia(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Beat tracking with Essentia (multifeature).
    Returns (beat_times_sec, bpm_est).
    """
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    # RhythmExtractor gives BPM and beat ticks directly (in seconds)
    rex = es.RhythmExtractor2013(method="multifeature")
    bpm, ticks, _, _, _ = rex(y)
    beats = np.asarray(ticks, dtype=np.float32)

    # Secondary fallback (rare): if few ticks, try BeatTrackerMultiFeature
    if beats.size < 2:
        beats = np.array(es.BeatTrackerMultiFeature()(y), dtype=np.float32)
        if beats.size >= 2:
            ibi = np.median(np.diff(beats))
            if ibi > 0:
                bpm = float(60.0 / ibi)

    return beats, float(bpm)


def _tempo_octave_correction(beats: np.ndarray, bpm_hint: float) -> float:
    """
    Correct classic half/double tempo errors by comparing median IBI-derived BPM
    vs. candidates {bpm_hint, bpm_hint*2, bpm_hint/2} and picking nearest.
    """
    if beats.size < 2:
        return bpm_hint
    ibi_med = float(np.median(np.diff(beats)))
    if ibi_med <= 0:
        return bpm_hint

    bpm_from_ibi = 60.0 / ibi_med
    cands = np.array([bpm_hint / 2.0, bpm_hint, bpm_hint * 2.0], dtype=float)
    return float(cands[np.argmin(np.abs(cands - bpm_from_ibi))])


def _estimate_key_essentia(y: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """
    Key estimation tuned for modern/electronic material.
    """
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    # Prefer EDMA (electronic/dance) profile; polyphonic for full mixes.
    try:
        key, scale, _ = es.KeyExtractor(profileType="edma", usePolyphony=True)(y)
        return key, scale
    except Exception:
        pass

    # Robust fallback within Essentia (still high-quality)
    try:
        key, scale, _ = es.KeyExtractor(profileType="harmonic", usePolyphony=True)(y)
        return key, scale
    except Exception:
        return None, None


def _best_bar_grid(
    beats_sec: np.ndarray,
    bpm: float,
    time_sig_candidates: Sequence[int] = (4, 3, 6),
) -> Tuple[int, int, float]:
    """
    Choose the best beats_per_bar (time signature) and downbeat phase.
    Returns (beats_per_bar, phase_offset, sec_per_bar).
    Scoring favors bar durations close to the BPM-implied bar length and low variance.
    """
    beats = np.asarray(beats_sec, dtype=float)
    if beats.size < 4:
        # default to 4/4 if too few beats
        sec_per_beat = 60.0 / max(bpm, 1e-6)
        return 4, 0, sec_per_beat * 4

    sec_per_beat = 60.0 / bpm
    best = None  # (score, bpb, phase, sec_per_bar)

    for bpb in time_sig_candidates:
        sec_per_bar = sec_per_beat * bpb
        for phase in range(bpb):
            bars_bound = beats[phase::bpb]
            if bars_bound.size < 2:
                continue
            durs = np.diff(bars_bound)
            mean_err = float(np.abs(np.median(durs) - sec_per_bar))
            var = float(np.var(durs)) if durs.size > 1 else 0.0
            score = mean_err + 0.5 * var
            # Strong 4/4 prior: non-4/4 must be substantially better to win.
            # With evenly-spaced beats, all groupings score ~0, so without
            # this bias the first candidate in the list wins arbitrarily.
            if bpb != 4:
                score += 0.1
            cand = (score, bpb, phase, sec_per_bar)
            if (best is None) or (score < best[0]):
                best = cand

    if best is None:
        # fallback heuristics
        return 4, 0, sec_per_beat * 4

    _, bpb, phase, sec_per_bar = best
    return bpb, phase, sec_per_bar

def slice_by_bars_from_beats(
    beats_sec: np.ndarray,
    bars: int,
    hop_bars: int = 8,
    beats_per_bar: int = 4,
    phase_offset: int = 0,
) -> list[tuple[float, float, int, int]]:
    """
    Beat-aligned slicing that also returns bar indices.
    Each tuple = (start_sec, end_sec, start_bar_idx, end_bar_idx)
    """
    beats = np.asarray(beats_sec, dtype=float)
    if beats.size < max(beats_per_bar, 2):
        return []

    bars_bound = beats[phase_offset::beats_per_bar]
    if bars_bound.size < 1:
        return []

    beat_dur = float(np.median(np.diff(beats))) if beats.size >= 2 else 60.0 / 120.0
    sec_per_bar = beat_dur * beats_per_bar
    out = []
    i = 0
    last_time = float(beats[-1] + beat_dur)

    while i < len(bars_bound):
        s = float(bars_bound[i])
        j = i + bars
        e = float(bars_bound[j]) if j < len(bars_bound) else min(s + bars * sec_per_bar, last_time)
        if e > s:
            out.append((s, e, i, i + bars))
        i += hop_bars
    return out


# -----------------------------
# Public API (drop-in)
# -----------------------------

def detect_beats_bpm_key(
    y: np.ndarray,
    sr: int,
    *,
    prefer_time_sigs: Sequence[int] = (4, 3, 6),
    lock_time_sig: Optional[int] = None,
) -> tuple[np.ndarray, float, Optional[str], Optional[str], int, int]:
    """
    Returns (beat_times_sec, bpm, key, scale, beats_per_bar, phase_offset)
    using Essentia-only high-accuracy pipeline.

    - Tempo & beats: RhythmExtractor2013(multifeature) (+ octave correction)
    - Key: KeyExtractor(edma, polyphonic)
    - Time signature & phase: best among prefer_time_sigs (or lock to a value)
    """
    # Ensure mono float32 input (Essentia expects 32-bit float)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    beats, bpm0 = _beats_essentia(y)
    if beats.size < 2:
        raise RuntimeError("Beat tracking failed to produce enough ticks for robust slicing.")

    bpm = _tempo_octave_correction(beats, bpm0)

    if lock_time_sig is not None:
        bpb = int(lock_time_sig)
        # choose phase that yields the most stable bars for this fixed bpb
        sec_per_bar = (60.0 / bpm) * bpb
        best_phase, best_score = 0, float("inf")
        for phase in range(bpb):
            bars_bound = beats[phase::bpb]
            if bars_bound.size < 2:
                continue
            durs = np.diff(bars_bound)
            mean_err = float(np.abs(np.median(durs) - sec_per_bar))
            var = float(np.var(durs)) if durs.size > 1 else 0.0
            score = mean_err + 0.5 * var
            if score < best_score:
                best_score = score
                best_phase = phase
        phase = best_phase
    else:
        bpb, phase, _ = _best_bar_grid(beats, bpm, prefer_time_sigs)

    key, scale = _estimate_key_essentia(y)
    return beats.astype(np.float32), float(bpm), key, scale, int(bpb), int(phase)


def sections_from_audio(
    y: np.ndarray,
    sr: int,
    bars_list: Sequence[int] = (2, 4, 8, 16),
    hop_bars: int = 8,
    time_sig_beats_per_bar: Optional[int] = None,
) -> List[Tuple[int, int, SectionMeta]]:
    """
    Build sample-indexed windows with high-accuracy bar/grid alignment.
    If time_sig_beats_per_bar is None, auto-detect best among (4,3,6).
    """
    beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(
        y, sr,
        prefer_time_sigs=(4, 3, 6) if time_sig_beats_per_bar is None else (time_sig_beats_per_bar,),
        lock_time_sig=time_sig_beats_per_bar,
    )

    out: List[Tuple[int, int, SectionMeta]] = []
    for bars in bars_list:
        wins = slice_by_bars_from_beats(
            beats_sec=beats,
            bars=bars,
            hop_bars=hop_bars,
            beats_per_bar=bpb,
            phase_offset=phase,
        )
        for (s, e, b0, b1) in wins:
            s_i, e_i = int(round(s * sr)), int(round(e * sr))
            out.append(
                (s_i, e_i, SectionMeta(start_s=s, end_s=e, bars=bars, bpm=bpm, key=key, scale=scale))
            )
    return out
