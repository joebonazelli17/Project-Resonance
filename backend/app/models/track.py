from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class TrackStatus(str, enum.Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(128), default="audio/aiff")
    file_size: Mapped[int] = mapped_column(Integer, default=0)

    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    key: Mapped[str | None] = mapped_column(String(8), nullable=True)
    scale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    beats_per_bar: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[TrackStatus] = mapped_column(
        Enum(TrackStatus), default=TrackStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    sections: Mapped[list[TrackSection]] = relationship(back_populates="track", cascade="all, delete-orphan")


class TrackSection(Base):
    __tablename__ = "track_sections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="CASCADE"))

    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    bars: Mapped[int] = mapped_column(Integer, nullable=False)
    bar_start: Mapped[int] = mapped_column(Integer, default=0)
    bar_end: Mapped[int] = mapped_column(Integer, default=0)

    bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    key: Mapped[str | None] = mapped_column(String(8), nullable=True)
    scale: Mapped[str | None] = mapped_column(String(16), nullable=True)

    hf_perc_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    rms_dbfs: Mapped[float] = mapped_column(Float, default=0.0)
    peak_dbfs: Mapped[float] = mapped_column(Float, default=0.0)
    crest_db: Mapped[float] = mapped_column(Float, default=0.0)
    flatness: Mapped[float] = mapped_column(Float, default=0.0)

    embedding_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    track: Mapped[Track] = relationship(back_populates="sections")
