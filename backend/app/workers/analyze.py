from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import async_session
from app.core.storage import download_to_path
from app.models.track import Track, TrackSection, TrackStatus


async def _update_status(track_id: str, status: TrackStatus, error: str | None = None, **kwargs):
    async with async_session() as db:
        result = await db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
        track = result.scalar_one_or_none()
        if track:
            track.status = status
            track.error_message = error
            for k, v in kwargs.items():
                if hasattr(track, k):
                    setattr(track, k, v)
            await db.commit()


def run_analysis(track_id: str) -> None:
    """Synchronous background task: download from S3, analyze, store sections."""
    import asyncio

    async def _inner():
        async with async_session() as db:
            result = await db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
            track = result.scalar_one_or_none()
            if not track:
                return

        await _update_status(track_id, TrackStatus.ANALYZING)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_path = Path(tmp) / track.filename
                download_to_path(track.s3_key, local_path)

                from app.engine.pipeline import load_mono
                from app.engine.tempo_bars import detect_beats_bpm_key, slice_by_bars_from_beats
                from app.engine.pipeline import _extra_features, _slice_by_bpm_fallback

                y, sr = load_mono(local_path)
                duration_s = float(y.shape[0] / sr) if y.size else 0.0
                beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)

                sections = []
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
                                 int([math]::Round(s / sec_per_bar)) if sec_per_bar else 0,
                                 int([math]::Round(e / sec_per_bar)) if sec_per_bar else 0)
                                for (s, e) in wins
                            ]

                    for win in wins:
                        s, e, b0, b1 = win if len(win) == 4 else (*win, 0, 0)
                        s_i, e_i = int(round(s * sr)), int(round(e * sr))
                        seg = y[s_i:e_i]
                        if seg.shape[0] < sr:
                            continue
                        feats = _extra_features(seg, sr)
                        sections.append(TrackSection(
                            track_id=uuid.UUID(track_id),
                            start_s=s, end_s=e, bars=bars,
                            bar_start=b0, bar_end=b1,
                            bpm=bpm, key=key, scale=scale,
                            **feats,
                        ))

                async with async_session() as db:
                    result = await db.execute(select(Track).where(Track.id == uuid.UUID(track_id)))
                    track = result.scalar_one_or_none()
                    if track:
                        track.duration_s = duration_s
                        track.bpm = bpm
                        track.key = key
                        track.scale = scale
                        track.beats_per_bar = bpb
                        track.status = TrackStatus.READY
                        for sec in sections:
                            db.add(sec)
                        await db.commit()

        except Exception as exc:
            await _update_status(track_id, TrackStatus.FAILED, error=str(exc))
            raise

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_inner())
    finally:
        loop.close()


async def run_search(query_track_id: str, bars: int = 4, hop_bars: int = 2, k: int = 5):
    """Run similarity search against the library."""
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

    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / query_track.filename
        download_to_path(query_track.s3_key, local_path)

        from app.engine.pipeline import load_mono, _extra_features, _slice_by_bpm_fallback
        from app.engine.tempo_bars import detect_beats_bpm_key, slice_by_bars_from_beats
        from app.engine.embeddings import embed_audio_batch
        from app.engine.index import build_index

        y, sr = load_mono(local_path)
        duration_s = float(y.shape[0] / sr) if y.size else 0.0
        beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)

        wins = slice_by_bars_from_beats(beats, bars=bars, hop_bars=hop_bars, beats_per_bar=bpb, phase_offset=phase)
        if not wins:
            wins = _slice_by_bpm_fallback(bpm, duration_s, bars, hop_bars, bpb)
            if wins:
                sec_per_bar = (60.0 / float(bpm)) * bpb if bpm else None
                wins = [(s, e, int(round(s / sec_per_bar)) if sec_per_bar else 0, int(round(e / sec_per_bar)) if sec_per_bar else 0) for (s, e) in wins]

        q_segments, q_meta = [], []
        for win in wins:
            s, e, b0, b1 = win if len(win) == 4 else (*win, 0, 0)
            s_i, e_i = int(round(s * sr)), int(round(e * sr))
            seg = y[s_i:e_i]
            if seg.shape[0] < sr:
                continue
            q_segments.append(seg)
            q_meta.append({"start_s": s, "end_s": e, "b0": b0, "b1": b1})

        if not q_segments:
            return SearchResponse(query_track_id=uuid.UUID(query_track_id), query_bpm=bpm, query_key=key, bars=bars, results=[])

        Q = embed_audio_batch(q_segments, sr).astype(np.float32)
        all_sections = [(t, sec) for t in lib_tracks for sec in t.sections if sec.bars == bars]

        if not all_sections:
            return SearchResponse(query_track_id=uuid.UUID(query_track_id), query_bpm=bpm, query_key=key, bars=bars, results=[])

        C = np.zeros((len(all_sections), 512), dtype=np.float32)
        index = build_index(C)
        k_real = min(k, len(all_sections))
        sims, ids = index.search(Q, k_real)

        window_results = []
        for qi, meta in enumerate(q_meta):
            matches = []
            for rank in range(k_real):
                idx = int(ids[qi][rank])
                if idx < 0:
                    continue
                lib_track, lib_sec = all_sections[idx]
                matches.append(SearchMatch(
                    match_track_id=lib_track.id, match_filename=lib_track.original_filename,
                    match_section_id=lib_sec.id, match_start_s=lib_sec.start_s, match_end_s=lib_sec.end_s,
                    match_bars=lib_sec.bars, match_bar_start=lib_sec.bar_start, match_bar_end=lib_sec.bar_end,
                    match_bpm=lib_sec.bpm, match_key=lib_sec.key, similarity=float(sims[qi][rank]),
                ))
            window_results.append(SearchWindowResult(
                query_start_s=meta["start_s"], query_end_s=meta["end_s"], query_bars=bars,
                query_bar_start=meta["b0"], query_bar_end=meta["b1"], matches=matches,
            ))
        return SearchResponse(query_track_id=uuid.UUID(query_track_id), query_bpm=bpm, query_key=key, bars=bars, results=window_results)