from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.storage import upload_file
from app.models.track import Track, TrackSection, TrackStatus
from app.schemas.track import SearchResponse, TextSearchResponse, TextSearchMatch, StemSearchRequest, StemSearchResponse
from app.workers.analyze import run_analysis, run_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_similar(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    bars: int = Query(default=4, ge=1, le=32),
    hop_bars: int = Query(default=2, ge=1),
    k: int = Query(default=5, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Upload a query track and find similar sections in the library."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif"}:
        raise HTTPException(400, f"Unsupported audio format: {ext}")

    track_id = uuid.uuid4()
    s3_key = f"queries/{track_id}{ext}"
    data = await file.read()
    upload_file(data, s3_key, content_type=file.content_type or "application/octet-stream")

    track = Track(
        id=track_id,
        filename=f"{track_id}{ext}",
        original_filename=file.filename or "unknown",
        s3_key=s3_key,
        content_type=file.content_type or "application/octet-stream",
        file_size=len(data),
        status=TrackStatus.PENDING,
    )
    db.add(track)
    await db.commit()

    results = await run_search(str(track_id), bars=bars, hop_bars=hop_bars, k=k)
    return results


@router.get("/text", response_model=TextSearchResponse)
async def search_by_text(
    q: str = Query(..., min_length=2, max_length=500, description="Natural language description"),
    bars: int | None = Query(default=None, ge=1, le=32),
    k: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Search the library by text description using CLAP's shared audio-text embedding space."""
    import numpy as np
    from sqlalchemy.orm import selectinload
    from app.engine.embeddings import embed_text
    from app.engine.index import build_index

    lib_result = await db.execute(
        select(Track)
        .options(selectinload(Track.sections))
        .where(Track.status == TrackStatus.READY)
    )
    lib_tracks = lib_result.scalars().all()

    all_sections = []
    corpus_vecs = []
    for t in lib_tracks:
        for sec in t.sections:
            if sec.embedding is None:
                continue
            if bars is not None and sec.bars != bars:
                continue
            all_sections.append((t, sec))
            corpus_vecs.append(np.frombuffer(sec.embedding, dtype=np.float32).copy())

    if not all_sections:
        return TextSearchResponse(query=q, matches=[])

    Q = embed_text([q])
    C = np.stack(corpus_vecs).astype(np.float32)
    index = build_index(C)
    k_real = min(k, len(all_sections))
    sims, ids = index.search(Q, k_real)

    matches = []
    for rank in range(k_real):
        idx = int(ids[0][rank])
        if idx < 0:
            continue
        lib_track, lib_sec = all_sections[idx]
        matches.append(TextSearchMatch(
            track_id=lib_track.id,
            filename=lib_track.original_filename,
            section_id=lib_sec.id,
            start_s=lib_sec.start_s,
            end_s=lib_sec.end_s,
            bars=lib_sec.bars,
            bar_start=lib_sec.bar_start,
            bar_end=lib_sec.bar_end,
            bpm=lib_sec.bpm,
            key=lib_sec.key,
            scale=lib_sec.scale,
            similarity=float(sims[0][rank]),
        ))

    return TextSearchResponse(query=q, matches=matches)


@router.post("/stems", response_model=StemSearchResponse)
async def search_by_stems(
    body: StemSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """Search with per-stem weighted similarity. Upload a query text or provide weights."""
    import numpy as np
    from app.engine.embeddings import embed_text
    from app.engine.index import build_index

    weights = body.weights
    w_total = sum(weights.values())
    if w_total <= 0:
        raise HTTPException(400, "At least one stem weight must be > 0")
    weights = {k: v / w_total for k, v in weights.items()}

    stem_col_map = {
        "mix": "embedding",
        "drums": "embedding_drums",
        "bass": "embedding_bass",
        "vocals": "embedding_vocals",
        "other": "embedding_other",
    }

    lib_result = await db.execute(
        select(Track).options(selectinload(Track.sections)).where(Track.status == TrackStatus.READY)
    )
    lib_tracks = lib_result.scalars().all()

    all_sections = []
    for t in lib_tracks:
        for sec in t.sections:
            if body.bars is not None and sec.bars != body.bars:
                continue
            if sec.embedding is None:
                continue
            all_sections.append((t, sec))

    if not all_sections:
        return StemSearchResponse(query=body.query, weights=weights, matches=[])

    Q_text = embed_text([body.query])

    scores = np.zeros(len(all_sections), dtype=np.float32)
    for stem_name, w in weights.items():
        if w <= 0:
            continue
        col = stem_col_map.get(stem_name, f"embedding_{stem_name}")
        vecs = []
        for _, sec in all_sections:
            emb_data = getattr(sec, col, None) if col != "embedding" else sec.embedding
            if emb_data is not None:
                vecs.append(np.frombuffer(emb_data, dtype=np.float32).copy())
            else:
                vecs.append(np.zeros(512, dtype=np.float32))

        C = np.stack(vecs).astype(np.float32)
        sims = (C @ Q_text[0]) / (np.linalg.norm(C, axis=1) * np.linalg.norm(Q_text[0]) + 1e-12)
        scores += w * sims

    top_k = min(body.k, len(all_sections))
    top_indices = np.argsort(scores)[::-1][:top_k]

    matches = []
    for idx in top_indices:
        lib_track, lib_sec = all_sections[int(idx)]
        matches.append(TextSearchMatch(
            track_id=lib_track.id,
            filename=lib_track.original_filename,
            section_id=lib_sec.id,
            start_s=lib_sec.start_s,
            end_s=lib_sec.end_s,
            bars=lib_sec.bars,
            bar_start=lib_sec.bar_start,
            bar_end=lib_sec.bar_end,
            bpm=lib_sec.bpm,
            key=lib_sec.key,
            scale=lib_sec.scale,
            similarity=float(scores[int(idx)]),
        ))

    return StemSearchResponse(query=body.query, weights=weights, matches=matches)


@router.get("/compare/{section_a_id}/{section_b_id}")
async def compare_sections(
    section_a_id: uuid.UUID,
    section_b_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Smart comparison with actionable, mastering-aware recommendations.
    Section A = user's track, Section B = reference.
    """
    from app.engine.spectral import BAND_NAMES, BAND_EDGES_HZ

    result_a = await db.execute(
        select(TrackSection).where(TrackSection.id == section_a_id)
    )
    result_b = await db.execute(
        select(TrackSection).where(TrackSection.id == section_b_id)
    )
    sec_a = result_a.scalar_one_or_none()
    sec_b = result_b.scalar_one_or_none()
    if not sec_a or not sec_b:
        raise HTTPException(404, "Section not found")

    # Get parent tracks for mastering state
    track_a_result = await db.execute(select(Track).where(Track.id == sec_a.track_id))
    track_b_result = await db.execute(select(Track).where(Track.id == sec_b.track_id))
    track_a = track_a_result.scalar_one_or_none()
    track_b = track_b_result.scalar_one_or_none()

    mastering_a = track_a.mastering_state if track_a else "unknown"
    mastering_b = track_b.mastering_state if track_b else "unknown"
    mastering_mismatch = (mastering_a != mastering_b) and "mastered" in (mastering_a or "", mastering_b or "")

    bands_a = sec_a.band_energies or {}
    bands_b = sec_b.band_energies or {}
    crest_a = sec_a.band_crest or {}
    crest_b = sec_b.band_crest or {}
    td_a = sec_a.band_transient_density or {}
    td_b = sec_b.band_transient_density or {}
    stereo_a = sec_a.stereo_features or {}
    stereo_b = sec_b.stereo_features or {}

    # Relative spectral shape (anchor at 1kHz = mid band)
    mid_a = bands_a.get("mid", -30)
    mid_b = bands_b.get("mid", -30)
    spectral_shape_delta = {}
    for name in BAND_NAMES:
        rel_a = bands_a.get(name, -96) - mid_a
        rel_b = bands_b.get(name, -96) - mid_b
        spectral_shape_delta[name] = round(rel_a - rel_b, 2)

    # Generate actionable recommendations
    recommendations = []

    for i, name in enumerate(BAND_NAMES):
        delta = spectral_shape_delta[name]
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        td_delta = td_a.get(name, 0) - td_b.get(name, 0)

        if abs(delta) < 1.5:
            continue

        if abs(td_delta) > 1.0 and abs(delta) > 2.0:
            # Transient density difference -- arrangement issue, not EQ
            if td_delta < 0:
                recommendations.append({
                    "band": name,
                    "freq_range": f"{lo}-{hi}Hz",
                    "type": "arrangement",
                    "message": f"Reference has {abs(td_delta):.1f} more transient events/sec at {lo}-{hi}Hz. "
                               f"This suggests more rhythmic content (e.g., faster hat patterns), not an EQ issue. "
                               f"Adding EQ boost here would make existing elements too bright.",
                    "severity": "info",
                })
            else:
                recommendations.append({
                    "band": name,
                    "freq_range": f"{lo}-{hi}Hz",
                    "type": "arrangement",
                    "message": f"Your track has {abs(td_delta):.1f} more transient events/sec at {lo}-{hi}Hz than the reference.",
                    "severity": "info",
                })
        elif delta < -1.5:
            recommendations.append({
                "band": name,
                "freq_range": f"{lo}-{hi}Hz",
                "type": "eq",
                "message": f"Consider a ~{abs(delta):.1f}dB boost around {lo}-{hi}Hz to match the reference's tonal balance.",
                "severity": "suggestion",
            })
        elif delta > 1.5:
            recommendations.append({
                "band": name,
                "freq_range": f"{lo}-{hi}Hz",
                "type": "eq",
                "message": f"Your track is ~{abs(delta):.1f}dB hotter at {lo}-{hi}Hz. Consider a gentle cut.",
                "severity": "suggestion",
            })

    # Dynamics recommendations (mastering-aware)
    dynamics_delta = {
        "rms_dbfs": round(sec_a.rms_dbfs - sec_b.rms_dbfs, 2),
        "peak_dbfs": round(sec_a.peak_dbfs - sec_b.peak_dbfs, 2),
        "crest_db": round(sec_a.crest_db - sec_b.crest_db, 2),
    }

    if mastering_mismatch:
        recommendations.append({
            "type": "mastering_context",
            "message": f"Note: your track appears to be '{mastering_a}' while the reference is '{mastering_b}'. "
                       f"Differences in overall loudness, crest factor, and compression are expected and should "
                       f"not be corrected at the mix stage. Focus on relative spectral balance and stereo image.",
            "severity": "warning",
        })
    else:
        if dynamics_delta["crest_db"] > 4:
            recommendations.append({
                "type": "dynamics",
                "message": f"Your crest factor is {dynamics_delta['crest_db']:.1f}dB higher -- consider bus compression to increase density.",
                "severity": "suggestion",
            })

    # Stereo recommendations
    if stereo_a and stereo_b:
        ms_diff = stereo_a.get("mid_side_ratio", 1) - stereo_b.get("mid_side_ratio", 1)
        if abs(ms_diff) > 0.1:
            direction = "narrower" if ms_diff > 0 else "wider"
            recommendations.append({
                "type": "stereo",
                "message": f"Your stereo image is {direction} than the reference (mid/side ratio diff: {ms_diff:.2f}).",
                "severity": "suggestion",
            })

    return {
        "section_a": str(section_a_id),
        "section_b": str(section_b_id),
        "mastering_state_a": mastering_a,
        "mastering_state_b": mastering_b,
        "mastering_mismatch": mastering_mismatch,
        "spectral_shape_delta": spectral_shape_delta,
        "band_crest_a": crest_a,
        "band_crest_b": crest_b,
        "transient_density_a": td_a,
        "transient_density_b": td_b,
        "stereo_a": stereo_a,
        "stereo_b": stereo_b,
        "dynamics_delta": dynamics_delta,
        "recommendations": recommendations,
    }


@router.get("/library/stats")
async def library_stats(db: AsyncSession = Depends(get_db)):
    """Return high-level stats about the indexed library."""
    result = await db.execute(
        select(Track).where(Track.status == TrackStatus.READY)
    )
    tracks = result.scalars().all()

    result_sections = await db.execute(select(TrackSection))
    sections = result_sections.scalars().all()

    return {
        "total_tracks": len(tracks),
        "total_sections": len(sections),
        "bars_distribution": {},
    }
