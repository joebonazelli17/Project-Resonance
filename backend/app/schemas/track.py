from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class TrackOut(BaseModel):
    id: uuid.UUID
    filename: str
    original_filename: str
    duration_s: float | None = None
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    beats_per_bar: int | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrackSectionOut(BaseModel):
    id: uuid.UUID
    track_id: uuid.UUID
    start_s: float
    end_s: float
    bars: int
    bar_start: int
    bar_end: int
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    hf_perc_ratio: float
    rms_dbfs: float
    peak_dbfs: float
    crest_db: float
    flatness: float

    model_config = {"from_attributes": True}


class TrackDetailOut(TrackOut):
    sections: list[TrackSectionOut] = []


class SearchMatch(BaseModel):
    match_track_id: uuid.UUID
    match_filename: str
    match_section_id: uuid.UUID
    match_start_s: float
    match_end_s: float
    match_bars: int
    match_bar_start: int
    match_bar_end: int
    match_bpm: float | None = None
    match_key: str | None = None
    similarity: float


class SearchWindowResult(BaseModel):
    query_start_s: float
    query_end_s: float
    query_bars: int
    query_bar_start: int
    query_bar_end: int
    matches: list[SearchMatch]


class SearchResponse(BaseModel):
    query_track_id: uuid.UUID
    query_bpm: float | None = None
    query_key: str | None = None
    bars: int
    results: list[SearchWindowResult]
