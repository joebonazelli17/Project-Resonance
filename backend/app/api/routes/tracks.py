from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.storage import upload_file, download_file, generate_presigned_url, delete_file
from app.models.track import Track, TrackStatus
from app.schemas.track import TrackOut, TrackDetailOut, TrackWithCurveOut
from app.workers.analyze import spawn_analysis

router = APIRouter(prefix="/tracks", tags=["tracks"])

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif"}


@router.post("/upload", response_model=TrackOut)
async def upload_track(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}. Supported: {', '.join(sorted(AUDIO_EXTS))}")

    track_id = uuid.uuid4()
    s3_key = f"uploads/{track_id}{ext}"
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
    await db.refresh(track)

    # Worker picks up PENDING tracks automatically.
    # Fall back to subprocess spawn if worker isn't running.
    spawn_analysis(str(track_id))

    return track


@router.post("/{track_id}/reanalyze")
async def reanalyze_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete existing sections and re-run full analysis from scratch."""
    from app.models.track import TrackSection
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    from sqlalchemy import delete
    await db.execute(delete(TrackSection).where(TrackSection.track_id == track_id))
    track.status = TrackStatus.PENDING
    await db.commit()
    spawn_analysis(str(track_id))
    return {"status": "reanalysis started", "track_id": str(track_id)}


@router.post("/{track_id}/relabel")
async def relabel_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Re-run section classifier on existing features. No audio reprocessing."""
    from app.models.track import TrackSection
    from app.engine.spectral import classify_section_label, _CLASSIFIER
    from sqlalchemy.orm import selectinload
    import numpy as np

    if _CLASSIFIER is None:
        raise HTTPException(500, "Section classifier not loaded")

    result = await db.execute(
        select(Track).options(selectinload(Track.sections)).where(Track.id == track_id)
    )
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")

    duration = track.duration_s or 1.0
    MIN_STEM_RANGE = 8.0

    # Group sections by bar size
    by_bars: dict[int, list[TrackSection]] = {}
    for sec in track.sections:
        by_bars.setdefault(sec.bars, [])
        by_bars[sec.bars].append(sec)

    # Compute track-wide percentiles from all sections
    all_rms = [s.rms_dbfs for s in track.sections]
    rms_pct = _pct(all_rms)
    rms_range = max(rms_pct["p75"] - rms_pct["p25"], 1e-6)

    stem_pcts = {}
    for stem in ["drums", "bass", "vocals", "other"]:
        vals = [s.stem_energies.get(stem, -96) for s in track.sections
                if s.stem_energies and s.stem_energies.get(stem, -96) > -90]
        stem_pcts[stem] = _pct(vals) if vals else {"p25": -40, "p50": -25, "p75": -10}

    def snorm(val, sp):
        sr = sp["p75"] - sp["p25"]
        return -1.0 if sr < MIN_STEM_RANGE else (val - sp["p25"]) / sr

    updated = 0
    for bars_key, secs_list in by_bars.items():
        secs_sorted = sorted(secs_list, key=lambda s: s.start_s)
        rms_list = [s.rms_dbfs for s in secs_sorted]

        for i, sec in enumerate(secs_sorted):
            se = sec.stem_energies or {}
            pos = sec.start_s / max(duration, 1e-6)
            en = (sec.rms_dbfs - rms_pct["p25"]) / rms_range

            dn = snorm(se.get("drums", -96), stem_pcts["drums"])
            bn = snorm(se.get("bass", -96), stem_pcts["bass"])
            vn = snorm(se.get("vocals", -96), stem_pcts["vocals"])
            on = snorm(se.get("other", -96), stem_pcts["other"])

            rd = (rms_list[i] - rms_list[i - 1]) if i >= 1 else 0.0
            rt = (rms_list[i] - rms_list[i - 3]) / 3.0 if i >= 3 else rd if i >= 1 else 0.0
            n2 = rms_list[i + 2] if i + 2 < len(rms_list) else rms_list[i]
            fd = (n2 - rms_list[i]) / 2.0

            label, confidence = classify_section_label(
                energy_norm=en, position_ratio=pos,
                drums_n=dn, bass_n=bn, vocals_n=vn, other_n=on,
                rms_delta=rd, rms_trend=rt, future_delta=fd,
                crest_db=sec.crest_db, hf_perc_ratio=sec.hf_perc_ratio,
                flatness=sec.flatness,
            )
            sec.section_label = label
            sec.section_label_confidence = confidence
            updated += 1

    await db.commit()
    return {"status": "relabeled", "track_id": str(track_id), "sections_updated": updated}


@router.post("/relabel-all")
async def relabel_all_tracks(db: AsyncSession = Depends(get_db)):
    """Re-run section classifier on all tracks. No audio reprocessing."""
    result = await db.execute(select(Track).where(Track.status == TrackStatus.READY))
    tracks = result.scalars().all()
    results = []
    for track in tracks:
        resp = await relabel_track(track.id, db)
        results.append(resp)
    return {"status": "all relabeled", "tracks": results}


def _pct(vals):
    if not vals:
        return {"p25": 0, "p50": 0, "p75": 0}
    s = sorted(vals)
    n = len(s)
    return {"p25": s[n // 4], "p50": s[n // 2], "p75": s[3 * n // 4]}


@router.get("", response_model=list[TrackOut])
async def list_tracks(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(Track).order_by(Track.created_at.desc()).offset(offset).limit(limit)
    if status:
        q = q.where(Track.status == status)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{track_id}", response_model=TrackDetailOut)
async def get_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Track).options(selectinload(Track.sections)).where(Track.id == track_id)
    )
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    return track


@router.get("/{track_id}/energy", response_model=TrackWithCurveOut)
async def get_track_energy(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get a track's energy curve (LUFS, centroid, onset density over time)."""
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    return track


@router.get("/{track_id}/stream")
async def stream_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return a URL for audio playback. Points to presigned S3 for browser-native
    formats, or to the /audio endpoint for transcoding (AIFF etc)."""
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")

    ext = Path(track.filename).suffix.lower()
    if ext in (".mp3", ".wav", ".flac", ".ogg"):
        return {"url": generate_presigned_url(track.s3_key)}

    # For AIFF and other non-browser formats, use the transcode endpoint
    return {"url": f"/api/tracks/{track_id}/audio"}


@router.get("/{track_id}/audio")
async def audio_transcode(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Transcode audio to WAV for browser playback."""
    import subprocess, tempfile, os
    from fastapi.responses import FileResponse

    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")

    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, track.filename)
    wav_path = os.path.join(tmp_dir, f"{track.id}.wav")

    with open(src_path, "wb") as f:
        f.write(download_file(track.s3_key))

    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "2", "-ar", "44100", wav_path],
        capture_output=True, timeout=120,
    )

    if os.path.exists(wav_path):
        return FileResponse(wav_path, media_type="audio/wav", filename=f"{track.id}.wav")

    raise HTTPException(500, "Transcoding failed")


@router.delete("/{track_id}")
async def delete_track(
    track_id: uuid.UUID,
    permanent: bool = Query(default=False, description="Permanently delete track and audio file"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    if permanent:
        try:
            delete_file(track.s3_key)
        except Exception:
            pass
        await db.delete(track)
        await db.commit()
        return {"deleted": str(track_id), "permanent": True}
    else:
        track.status = TrackStatus.DELETED
        await db.commit()
        return {"deleted": str(track_id), "permanent": False}
