from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, field_validator


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
    mastering_state: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def lowercase_status(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    model_config = {"from_attributes": True}


class TrackWithCurveOut(TrackOut):
    energy_curve: dict | None = None


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
    section_label: str | None = None
    section_label_confidence: float | None = None
    band_energies: dict | None = None
    stereo_features: dict | None = None
    band_crest: dict | None = None
    band_transient_density: dict | None = None
    stem_energies: dict | None = None

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


class ReferenceCoachMatch(BaseModel):
    track_id: uuid.UUID
    filename: str
    section_id: uuid.UUID
    start_s: float
    end_s: float
    bars: int
    bar_start: int
    bar_end: int
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    section_label: str | None = None
    similarity: float
    match_basis: str = "any_label"


class ReferenceCoachSection(BaseModel):
    query_section_id: uuid.UUID
    query_start_s: float
    query_end_s: float
    query_bar_start: int
    query_bar_end: int
    query_section_label: str | None = None
    query_section_label_confidence: float | None = None
    anchor_match: ReferenceCoachMatch | None = None
    alternate_matches: list[ReferenceCoachMatch] = []


class ReferenceCoachAnchor(BaseModel):
    track_id: uuid.UUID
    filename: str
    mastering_state: str | None = None
    avg_similarity: float
    coverage_ratio: float
    matched_sections: int


class ReferenceCoachResponse(BaseModel):
    track_id: uuid.UUID
    bars: int
    mastering_state: str | None = None
    matching_ready: bool
    pending_embedding_sections: int
    total_sections: int
    matched_sections: int
    anchor_track: ReferenceCoachAnchor | None = None
    sections: list[ReferenceCoachSection] = []


class TextSearchMatch(BaseModel):
    track_id: uuid.UUID
    filename: str
    section_id: uuid.UUID
    start_s: float
    end_s: float
    bars: int
    bar_start: int
    bar_end: int
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    similarity: float


class TextSearchResponse(BaseModel):
    query: str
    matches: list[TextSearchMatch]


class StemSearchRequest(BaseModel):
    query: str
    weights: dict[str, float] = {"mix": 0.2, "drums": 0.3, "bass": 0.2, "vocals": 0.2, "other": 0.1}
    bars: int | None = None
    k: int = 10


class StemSearchResponse(BaseModel):
    query: str
    weights: dict[str, float]
    matches: list[TextSearchMatch]
