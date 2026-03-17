from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.storage import upload_file, download_file, generate_presigned_url, delete_file
from app.models.track import Track, TrackStatus
from app.schemas.track import TrackOut, TrackDetailOut, TrackWithCurveOut
from app.workers.analyze import run_analysis

router = APIRouter(prefix="/tracks", tags=["tracks"])

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif"}


@router.post("/upload", response_model=TrackOut)
async def upload_track(
    background_tasks: BackgroundTasks,
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

    background_tasks.add_task(run_analysis, str(track_id))

    return track


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
async def delete_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    try:
        delete_file(track.s3_key)
    except Exception:
        pass
    await db.delete(track)
    await db.commit()
    return {"deleted": str(track_id)}
