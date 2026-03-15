from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import SyncSession, async_session
from app.core.storage import download_to_path
from app.models.track import Track, TrackSection, TrackStatus

STEM_NAMES = ("drums", "bass", "vocals", "other")


def _sync_update_status(track_id: str, status: TrackStatus, error: str | None = None, **kwargs):
    with SyncSession() as db:
        result = db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
        track = result.scalar_one_or_none()
        if track:
            track.status = status
            track.error_message = error
            for k, v in kwargs.items():
                if hasattr(track, k):
                    setattr(track, k, v)
            db.commit()


def _embedding_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()


def _compute_percentiles(values: list[float]) -> dict:
    """Compute p25/p50/p75 for a list of values."""
    if not values:
        return {"p25": 0, "p50": 0, "p75": 0}
    arr = np.array(values)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def run_analysis(track_id: str) -> None:
    """Full analysis pipeline with deep spectral profiling and mastering detection."""

    with SyncSession() as db:
        result = db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
        track = result.scalar_one_or_none()
        if not track:
            return
        filename = track.filename
        s3_key = track.s3_key

    _sync_update_status(track_id, TrackStatus.ANALYZING)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / filename
            download_to_path(s3_key, local_path)

            from app.engine.pipeline import load_mono, _extra_features, _slice_by_bpm_fallback
            from app.engine.tempo_bars import detect_beats_bpm_key, slice_by_bars_from_beats
            from app.engine.embeddings import embed_audio_batch
            from app.engine.spectral import (
                compute_band_energies, compute_energy_curve,
                compute_stereo_features, detect_section_label,
                compute_eq_profile, compute_eq_profile_peak, compute_eq_profile_variance,
                compute_band_crest, compute_band_transient_density,
                detect_mastering_state,
            )
            import librosa as _lr

            y, sr = load_mono(local_path)
            duration_s = float(y.shape[0] / sr) if y.size else 0.0
            beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)

            energy_curve = compute_energy_curve(y, sr, hop_s=0.5)
            mastering_state = detect_mastering_state(y, sr)

            y_stereo = None
            try:
                y_st, _ = _lr.load(str(local_path), sr=sr, mono=False)
                if y_st.ndim == 2 and y_st.shape[0] == 2:
                    y_stereo = y_st
            except Exception:
                pass

            stems = {}
            try:
                from app.engine.stems import separate_stems
                stems = separate_stems(y, sr)
                print(f"[analyze] Demucs separated {list(stems.keys())} for {filename}")
            except Exception as ex:
                print(f"[analyze] Demucs failed, continuing without stems: {ex}")

            # --- Pass 1: collect all segments and raw features for percentile computation ---
            raw_segments = []  # (seg, s, e, b0, b1, bars, s_i, e_i)

            for bars in settings.DEFAULT_BARS_LIST:
                wins = slice_by_bars_from_beats(
                    beats, bars=bars, hop_bars=settings.DEFAULT_HOP_BARS,
                    beats_per_bar=bpb, phase_offset=phase,
                )
                if not wins:
                    wins = _slice_by_bpm_fallback(bpm, duration_s, bars, settings.DEFAULT_HOP_BARS, bpb)
                    if wins:
                        sec_per_bar = (60.0 / float(bpm)) * bpb if bpm else None
                        wins = [
                            (s, e,
                             int(round(s / sec_per_bar)) if sec_per_bar else 0,
                             int(round(e / sec_per_bar)) if sec_per_bar else 0)
                            for (s, e) in wins
                        ]

                for win in wins:
                    s, e, b0, b1 = win if len(win) == 4 else (*win, 0, 0)
                    s_i, e_i = int(round(s * sr)), int(round(e * sr))
                    seg = y[s_i:e_i]
                    if seg.shape[0] < sr:
                        continue
                    raw_segments.append((seg, s, e, b0, b1, bars, s_i, e_i))

            # Compute features for percentile calculation
            all_rms = []
            all_onset = []
            all_hf = []
            pre_feats = []
            for (seg, s, e, b0, b1, bars, s_i, e_i) in raw_segments:
                feats = _extra_features(seg, sr)
                onset_env = _lr.onset.onset_strength(y=seg, sr=sr)
                onset_d = float(onset_env.mean())
                all_rms.append(feats["rms_dbfs"])
                all_onset.append(onset_d)
                all_hf.append(feats["hf_perc_ratio"])
                pre_feats.append((feats, onset_d))

            rms_pct = _compute_percentiles(all_rms)
            onset_pct = _compute_percentiles(all_onset)
            hf_pct = _compute_percentiles(all_hf)

            # --- Pass 2: compute all features with relative thresholds ---
            seg_audio = []
            seg_stem_audio = {name: [] for name in STEM_NAMES}
            seg_meta = []

            for idx, (seg, s, e, b0, b1, bars, s_i, e_i) in enumerate(raw_segments):
                feats, onset_d = pre_feats[idx]

                band_e = compute_band_energies(seg, sr)
                band_cr = compute_band_crest(seg, sr)
                band_td = compute_band_transient_density(seg, sr)

                eq_avg = compute_eq_profile(seg, sr)
                eq_peak = compute_eq_profile_peak(seg, sr)
                eq_var = compute_eq_profile_variance(seg, sr)

                stereo_f = None
                if y_stereo is not None:
                    st_seg = y_stereo[:, s_i:e_i]
                    if st_seg.shape[1] >= sr:
                        stereo_f = compute_stereo_features(st_seg, sr)

                pos_ratio = float(s) / max(duration_s, 1e-6)
                label, confidence = detect_section_label(
                    rms_dbfs=feats["rms_dbfs"],
                    onset_density=onset_d,
                    hf_perc_ratio=feats["hf_perc_ratio"],
                    flatness=feats["flatness"],
                    position_ratio=pos_ratio,
                    bpm=bpm,
                    track_rms_percentiles=rms_pct,
                    track_onset_percentiles=onset_pct,
                    track_hf_percentiles=hf_pct,
                )

                seg_audio.append(seg)
                seg_meta.append({
                    "start_s": s, "end_s": e, "bars": bars,
                    "bar_start": b0, "bar_end": b1,
                    "bpm": bpm, "key": key, "scale": scale,
                    "section_label": label,
                    "section_label_confidence": confidence,
                    "band_energies": band_e,
                    "stereo_features": stereo_f,
                    "band_crest": band_cr,
                    "band_transient_density": band_td,
                    "_eq_profile": eq_avg,
                    "_eq_profile_peak": eq_peak,
                    "_eq_profile_variance": eq_var,
                    **feats,
                })

                for name in STEM_NAMES:
                    if name in stems:
                        stem_seg = stems[name][s_i:e_i]
                        seg_stem_audio[name].append(stem_seg if stem_seg.shape[0] >= sr else None)
                    else:
                        seg_stem_audio[name].append(None)

            # --- Batch embed ---
            mix_embeddings = embed_audio_batch(seg_audio, sr) if seg_audio else None

            stem_embeddings = {}
            for name in STEM_NAMES:
                valid_segs = [(i, s) for i, s in enumerate(seg_stem_audio[name]) if s is not None]
                if valid_segs:
                    indices, segs = zip(*valid_segs)
                    embs = embed_audio_batch(list(segs), sr)
                    stem_embeddings[name] = {idx: embs[j] for j, idx in enumerate(indices)}
                else:
                    stem_embeddings[name] = {}

            # --- Persist ---
            with SyncSession() as db:
                result = db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
                track = result.scalar_one_or_none()
                if track:
                    track.duration_s = duration_s
                    track.bpm = bpm
                    track.key = key
                    track.scale = scale
                    track.beats_per_bar = bpb
                    track.status = TrackStatus.READY
                    track.energy_curve = energy_curve
                    track.mastering_state = mastering_state

                    for i, meta in enumerate(seg_meta):
                        eq_avg = meta.pop("_eq_profile")
                        eq_peak = meta.pop("_eq_profile_peak")
                        eq_var = meta.pop("_eq_profile_variance")

                        mix_emb = _embedding_to_bytes(mix_embeddings[i]) if mix_embeddings is not None else None
                        drums_emb = _embedding_to_bytes(stem_embeddings["drums"][i]) if i in stem_embeddings.get("drums", {}) else None
                        bass_emb = _embedding_to_bytes(stem_embeddings["bass"][i]) if i in stem_embeddings.get("bass", {}) else None
                        vocals_emb = _embedding_to_bytes(stem_embeddings["vocals"][i]) if i in stem_embeddings.get("vocals", {}) else None
                        other_emb = _embedding_to_bytes(stem_embeddings["other"][i]) if i in stem_embeddings.get("other", {}) else None

                        section = TrackSection(
                            track_id=uuid.UUID(track_id),
                            embedding=mix_emb,
                            embedding_drums=drums_emb,
                            embedding_bass=bass_emb,
                            embedding_vocals=vocals_emb,
                            embedding_other=other_emb,
                            eq_profile=_embedding_to_bytes(eq_avg),
                            eq_profile_peak=_embedding_to_bytes(eq_peak),
                            eq_profile_variance=_embedding_to_bytes(eq_var),
                            **meta,
                        )
                        db.add(section)
                    db.commit()

            print(f"[analyze] {filename}: {len(seg_meta)} sections, mastering={mastering_state}, energy_curve={len(energy_curve.get('times', []))} pts")

    except Exception as exc:
        _sync_update_status(track_id, TrackStatus.FAILED, error=str(exc))
        raise


async def run_search(query_track_id: str, bars: int = 4, hop_bars: int = 2, k: int = 5):
    """Blended similarity search: CLAP embeddings + EQ profile + feature gating."""
    from app.schemas.track import SearchResponse, SearchWindowResult, SearchMatch

    async with async_session() as db:
        result = await db.execute(select(Track).where(Track.id == uuid.UUID(query_track_id)))
        query_track = result.scalar_one_or_none()
        if not query_track:
            raise ValueError(f"Query track {query_track_id} not found")
        lib_result = await db.execute(
            select(Track).options(selectinload(Track.sections)).where(Track.status == TrackStatus.READY)
        )
        lib_tracks = lib_result.scalars().all()

    if not lib_tracks:
        return SearchResponse(query_track_id=uuid.UUID(query_track_id), bars=bars, results=[])

    all_sections = []
    corpus_clap = []
    corpus_eq = []
    for t in lib_tracks:
        for sec in t.sections:
            if sec.bars == bars and sec.embedding is not None:
                all_sections.append((t, sec))
                corpus_clap.append(_bytes_to_embedding(sec.embedding))
                if sec.eq_profile is not None:
                    corpus_eq.append(_bytes_to_embedding(sec.eq_profile))
                else:
                    corpus_eq.append(np.zeros(64, dtype=np.float32))

    if not all_sections:
        return SearchResponse(query_track_id=uuid.UUID(query_track_id), bars=bars, results=[])

    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / query_track.filename
        download_to_path(query_track.s3_key, local_path)

        from app.engine.pipeline import load_mono, _extra_features, _slice_by_bpm_fallback
        from app.engine.tempo_bars import detect_beats_bpm_key, slice_by_bars_from_beats
        from app.engine.embeddings import embed_audio_batch
        from app.engine.spectral import compute_eq_profile

        y, sr = load_mono(local_path)
        duration_s = float(y.shape[0] / sr) if y.size else 0.0
        beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)

        wins = slice_by_bars_from_beats(beats, bars=bars, hop_bars=hop_bars, beats_per_bar=bpb, phase_offset=phase)
        if not wins:
            wins = _slice_by_bpm_fallback(bpm, duration_s, bars, hop_bars, bpb)
            if wins:
                sec_per_bar = (60.0 / float(bpm)) * bpb if bpm else None
                wins = [(s, e, int(round(s / sec_per_bar)) if sec_per_bar else 0, int(round(e / sec_per_bar)) if sec_per_bar else 0) for (s, e) in wins]

        q_segments, q_meta, q_feats_list = [], [], []
        for win in wins:
            s, e, b0, b1 = win if len(win) == 4 else (*win, 0, 0)
            s_i, e_i = int(round(s * sr)), int(round(e * sr))
            seg = y[s_i:e_i]
            if seg.shape[0] < sr:
                continue
            q_segments.append(seg)
            q_meta.append({"start_s": s, "end_s": e, "b0": b0, "b1": b1})
            q_feats_list.append(_extra_features(seg, sr))

        if not q_segments:
            return SearchResponse(query_track_id=uuid.UUID(query_track_id), query_bpm=bpm, query_key=key, bars=bars, results=[])

        Q_clap = embed_audio_batch(q_segments, sr).astype(np.float32)
        Q_eq = np.stack([compute_eq_profile(seg, sr) for seg in q_segments]).astype(np.float32)

        C_clap = np.stack(corpus_clap).astype(np.float32)
        C_eq = np.stack(corpus_eq).astype(np.float32)

        # Blended scoring: CLAP cosine + EQ profile cosine
        EQ_WEIGHT = 0.3
        HF_GATE_THRESHOLD = 0.08

        window_results = []
        for qi, meta in enumerate(q_meta):
            clap_sims = C_clap @ Q_clap[qi] / (np.linalg.norm(C_clap, axis=1) * np.linalg.norm(Q_clap[qi]) + 1e-12)
            eq_sims = C_eq @ Q_eq[qi] / (np.linalg.norm(C_eq, axis=1) * np.linalg.norm(Q_eq[qi]) + 1e-12)
            blended = (1.0 - EQ_WEIGHT) * clap_sims + EQ_WEIGHT * eq_sims

            # HF percussive gating
            q_hf = q_feats_list[qi]["hf_perc_ratio"]
            for j, (_, sec) in enumerate(all_sections):
                if abs(sec.hf_perc_ratio - q_hf) > HF_GATE_THRESHOLD:
                    blended[j] = -1.0

            top_k = min(k, len(all_sections))
            top_indices = np.argsort(blended)[::-1][:top_k]

            matches = []
            for idx in top_indices:
                if blended[idx] < 0:
                    continue
                lib_track, lib_sec = all_sections[int(idx)]
                matches.append(SearchMatch(
                    match_track_id=lib_track.id, match_filename=lib_track.original_filename,
                    match_section_id=lib_sec.id, match_start_s=lib_sec.start_s, match_end_s=lib_sec.end_s,
                    match_bars=lib_sec.bars, match_bar_start=lib_sec.bar_start, match_bar_end=lib_sec.bar_end,
                    match_bpm=lib_sec.bpm, match_key=lib_sec.key, similarity=float(blended[int(idx)]),
                ))

            window_results.append(SearchWindowResult(
                query_start_s=meta["start_s"], query_end_s=meta["end_s"], query_bars=bars,
                query_bar_start=meta["b0"], query_bar_end=meta["b1"], matches=matches,
            ))
        return SearchResponse(query_track_id=uuid.UUID(query_track_id), query_bpm=bpm, query_key=key, bars=bars, results=window_results)