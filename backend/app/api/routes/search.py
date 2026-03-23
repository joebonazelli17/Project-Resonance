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
from app.schemas.track import (
    ReferenceCoachAnchor,
    ReferenceCoachMatch,
    ReferenceCoachResponse,
    ReferenceCoachSection,
    SearchResponse,
    StemSearchRequest,
    StemSearchResponse,
    TextSearchMatch,
    TextSearchResponse,
)
from app.workers.analyze import run_analysis, run_search

router = APIRouter(prefix="/search", tags=["search"])


def _match_basis(query_label: str | None, candidate_has_same_label: bool) -> str:
    if query_label and candidate_has_same_label:
        return "same_label"
    if query_label:
        return "fallback_any_label"
    return "any_label"


def _coach_match(track: Track, section: TrackSection, similarity: float, match_basis: str) -> ReferenceCoachMatch:
    return ReferenceCoachMatch(
        track_id=track.id,
        filename=track.original_filename,
        section_id=section.id,
        start_s=section.start_s,
        end_s=section.end_s,
        bars=section.bars,
        bar_start=section.bar_start,
        bar_end=section.bar_end,
        bpm=section.bpm,
        key=section.key,
        scale=section.scale,
        section_label=section.section_label,
        similarity=float(similarity),
        match_basis=match_basis,
    )


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

    try:
        results = await run_search(str(track_id), bars=bars, hop_bars=hop_bars, k=k)
        return results
    finally:
        # Clean up temporary query track record and S3 object
        try:
            from app.core.database import async_session as _async_session
            async with _async_session() as cleanup_db:
                query_track = await cleanup_db.get(Track, track_id)
                if query_track:
                    await cleanup_db.delete(query_track)
                    await cleanup_db.commit()
        except Exception:
            pass
        try:
            from app.core.storage import delete_file
            delete_file(s3_key)
        except Exception:
            pass


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


@router.get("/reference-coach/{track_id}", response_model=ReferenceCoachResponse)
async def reference_coach(
    track_id: uuid.UUID,
    bars: int = Query(default=8, ge=1, le=32),
    alternates: int = Query(default=2, ge=0, le=5),
    db: AsyncSession = Depends(get_db),
):
    import numpy as np

    result = await db.execute(
        select(Track).options(selectinload(Track.sections)).where(Track.id == track_id)
    )
    query_track = result.scalar_one_or_none()
    if not query_track:
        raise HTTPException(404, "Track not found")

    query_sections = sorted(
        [sec for sec in query_track.sections if sec.bars == bars],
        key=lambda sec: (sec.start_s, sec.bar_start, sec.bar_end),
    )
    pending_embedding_sections = sum(1 for sec in query_sections if sec.embedding is None)
    matching_ready = pending_embedding_sections == 0

    lib_result = await db.execute(
        select(Track)
        .options(selectinload(Track.sections))
        .where(Track.status == TrackStatus.READY, Track.id != track_id)
    )
    lib_tracks = lib_result.scalars().all()

    candidate_rows: list[tuple[Track, TrackSection]] = []
    candidate_vecs: list[np.ndarray] = []
    label_to_candidate_indices: dict[str, list[int]] = {}

    for lib_track in lib_tracks:
        for sec in lib_track.sections:
            if sec.bars != bars or sec.embedding is None:
                continue
            idx = len(candidate_rows)
            candidate_rows.append((lib_track, sec))
            candidate_vecs.append(np.frombuffer(sec.embedding, dtype=np.float32).copy())
            if sec.section_label:
                label_to_candidate_indices.setdefault(sec.section_label, []).append(idx)

    if candidate_vecs:
        corpus = np.stack(candidate_vecs).astype(np.float32)
        corpus_norms = np.linalg.norm(corpus, axis=1) + 1e-12
    else:
        corpus = None
        corpus_norms = None

    track_score_sums: dict[str, float] = {}
    track_weight_sums: dict[str, float] = {}
    track_match_counts: dict[str, int] = {}
    track_lookup = {str(track.id): track for track in lib_tracks}
    comparable_sections = 0
    query_entries = []

    for sec in query_sections:
        if sec.embedding is None or corpus is None or corpus_norms is None:
            query_entries.append({
                "section": sec,
                "sims": None,
                "ranked_indices": [],
                "best_idx_by_track": {},
                "match_basis": _match_basis(sec.section_label, False),
            })
            continue

        comparable_sections += 1
        q_vec = np.frombuffer(sec.embedding, dtype=np.float32).copy()
        q_norm = float(np.linalg.norm(q_vec) + 1e-12)
        sims = (corpus @ q_vec) / (corpus_norms * q_norm)

        same_label_indices = label_to_candidate_indices.get(sec.section_label or "", [])
        use_same_label = bool(sec.section_label and same_label_indices)
        valid_indices = (
            np.asarray(same_label_indices, dtype=np.int32)
            if use_same_label
            else np.arange(len(candidate_rows), dtype=np.int32)
        )
        ranked_indices = valid_indices[np.argsort(sims[valid_indices])[::-1]]

        best_idx_by_track: dict[str, int] = {}
        weight = max(float(sec.section_label_confidence or 0.5), 0.25)
        for candidate_idx in ranked_indices:
            lib_track, _ = candidate_rows[int(candidate_idx)]
            score = max(float(sims[int(candidate_idx)]), 0.0)
            if score <= 0:
                continue
            track_key = str(lib_track.id)
            if track_key in best_idx_by_track:
                continue
            best_idx_by_track[track_key] = int(candidate_idx)
            track_score_sums[track_key] = track_score_sums.get(track_key, 0.0) + (score * weight)
            track_weight_sums[track_key] = track_weight_sums.get(track_key, 0.0) + weight
            track_match_counts[track_key] = track_match_counts.get(track_key, 0) + 1

        query_entries.append({
            "section": sec,
            "sims": sims,
            "ranked_indices": ranked_indices.tolist(),
            "best_idx_by_track": best_idx_by_track,
            "match_basis": _match_basis(sec.section_label, use_same_label),
        })

    anchor_track_id: str | None = None
    anchor_track_payload: ReferenceCoachAnchor | None = None
    if comparable_sections > 0 and track_score_sums:
        best_rank = None
        for candidate_track_id, score_sum in track_score_sums.items():
            weight_sum = max(track_weight_sums.get(candidate_track_id, 0.0), 1e-6)
            match_count = track_match_counts.get(candidate_track_id, 0)
            coverage_ratio = match_count / max(comparable_sections, 1)
            avg_similarity = score_sum / weight_sum
            rank = (avg_similarity * coverage_ratio, coverage_ratio, avg_similarity, match_count)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                anchor_track_id = candidate_track_id

        if anchor_track_id:
            anchor_track = track_lookup[anchor_track_id]
            weight_sum = max(track_weight_sums.get(anchor_track_id, 0.0), 1e-6)
            match_count = track_match_counts.get(anchor_track_id, 0)
            coverage_ratio = match_count / max(comparable_sections, 1)
            avg_similarity = track_score_sums[anchor_track_id] / weight_sum
            anchor_track_payload = ReferenceCoachAnchor(
                track_id=anchor_track.id,
                filename=anchor_track.original_filename,
                mastering_state=anchor_track.mastering_state,
                avg_similarity=float(avg_similarity),
                coverage_ratio=float(coverage_ratio),
                matched_sections=match_count,
            )

    response_sections: list[ReferenceCoachSection] = []
    matched_sections = 0
    for entry in query_entries:
        sec: TrackSection = entry["section"]
        sims = entry["sims"]
        match_basis = entry["match_basis"]
        anchor_match = None
        if anchor_track_id and sims is not None:
            anchor_idx = entry["best_idx_by_track"].get(anchor_track_id)
            if anchor_idx is not None:
                lib_track, lib_sec = candidate_rows[anchor_idx]
                anchor_match = _coach_match(
                    lib_track,
                    lib_sec,
                    max(float(sims[anchor_idx]), 0.0),
                    match_basis,
                )
                matched_sections += 1

        alternate_matches: list[ReferenceCoachMatch] = []
        if sims is not None and alternates > 0:
            seen_tracks = {anchor_track_id} if anchor_track_id else set()
            for candidate_idx in entry["ranked_indices"]:
                lib_track, lib_sec = candidate_rows[int(candidate_idx)]
                track_key = str(lib_track.id)
                similarity = max(float(sims[int(candidate_idx)]), 0.0)
                if similarity <= 0 or track_key in seen_tracks:
                    continue
                seen_tracks.add(track_key)
                alternate_matches.append(_coach_match(lib_track, lib_sec, similarity, match_basis))
                if len(alternate_matches) >= alternates:
                    break

        response_sections.append(ReferenceCoachSection(
            query_section_id=sec.id,
            query_start_s=sec.start_s,
            query_end_s=sec.end_s,
            query_bar_start=sec.bar_start,
            query_bar_end=sec.bar_end,
            query_section_label=sec.section_label,
            query_section_label_confidence=sec.section_label_confidence,
            anchor_match=anchor_match,
            alternate_matches=alternate_matches,
        ))

    return ReferenceCoachResponse(
        track_id=query_track.id,
        bars=bars,
        mastering_state=query_track.mastering_state,
        matching_ready=matching_ready,
        pending_embedding_sections=pending_embedding_sections,
        total_sections=len(query_sections),
        matched_sections=matched_sections,
        anchor_track=anchor_track_payload,
        sections=response_sections,
    )


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
    stem_a = sec_a.stem_energies or {}
    stem_b = sec_b.stem_energies or {}

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
    stem_balance_delta = {
        name: round(stem_a.get(name, -96.0) - stem_b.get(name, -96.0), 2)
        for name in ("drums", "bass", "vocals", "other")
    }

    for stem_name, delta in stem_balance_delta.items():
        if abs(delta) < 2.5:
            continue
        if delta < 0:
            recommendations.append({
                "type": "stem_balance",
                "stem": stem_name,
                "message": f"Your {stem_name} energy is about {abs(delta):.1f}dB lower than the reference. "
                           f"If this section feels smaller, the gap is likely component balance or arrangement, not just EQ.",
                "severity": "suggestion",
            })
        else:
            recommendations.append({
                "type": "stem_balance",
                "stem": stem_name,
                "message": f"Your {stem_name} energy is about {abs(delta):.1f}dB hotter than the reference.",
                "severity": "info",
            })

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
        "section_label_a": sec_a.section_label,
        "section_label_b": sec_b.section_label,
        "stem_energies_a": stem_a,
        "stem_energies_b": stem_b,
        "stem_balance_delta": stem_balance_delta,
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

    bars_dist: dict[int, int] = {}
    for sec in sections:
        bars_dist[sec.bars] = bars_dist.get(sec.bars, 0) + 1

    mastering_dist: dict[str, int] = {}
    for t in tracks:
        state = t.mastering_state or "unknown"
        mastering_dist[state] = mastering_dist.get(state, 0) + 1

    return {
        "total_tracks": len(tracks),
        "total_sections": len(sections),
        "bars_distribution": dict(sorted(bars_dist.items())),
        "mastering_distribution": mastering_dist,
    }
