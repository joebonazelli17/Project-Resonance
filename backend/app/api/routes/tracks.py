from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import get_db
from app.core.storage import upload_file, generate_presigned_url, delete_file
from app.models.track import Track, TrackStatus
from app.schemas.track import TrackOut, TrackDetailOut
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


@router.get("/{track_id}/stream")
async def stream_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, "Track not found")
    url = generate_presigned_url(track.s3_key)
    return {"url": url}


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
