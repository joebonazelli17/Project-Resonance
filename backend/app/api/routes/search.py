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
from app.schemas.track import SearchResponse
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
