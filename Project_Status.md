# Project Resonance -- Status & Architecture

## What Is This

An AI Music Intelligence Platform built for music producers and DJs. The core product is a **Reference Track Engine** that lets producers search their music library by audio similarity, text description, or stem-weighted queries -- at the section level (bars), not just whole tracks.

The long-term vision includes an AI Mix Coach (loudness/spectral/dynamics feedback vs genre norms), but the current build focuses entirely on the Reference Engine.

## Tech Stack

### Backend (`backend/`)
- **FastAPI** (async, Python 3.11+)
- **PostgreSQL** via SQLAlchemy 2.0 async + asyncpg
- **MinIO** (S3-compatible) for audio file storage
- **CLAP** (LAION, HTSAT-base) for audio + text embeddings (512-dim shared space)
- **Essentia** for beat tracking, BPM, key, time signature detection
- **Demucs** (htdemucs) for stem separation (drums/bass/vocals/other)
- **FAISS** (flat inner-product) for vector similarity search
- **librosa** for spectral analysis, onset detection, mel features

### Frontend (`frontend/`)
- **Next.js 15** + React 19 + TypeScript
- **Tailwind CSS** with custom `resonance` color palette (dark theme)
- **wavesurfer.js 7** with Regions plugin for waveform display + section highlighting
- **lucide-react** for icons

### Infrastructure
- **docker-compose.yml** orchestrates 4 services: Postgres 16, MinIO, FastAPI backend, Next.js frontend
- **Dockerfiles** for both backend and frontend (standalone Next.js output)
- `.env.example` with all config variables

## Architecture

```
Project-Resonance/
├── backend/
│   ├── app/
│   │   ├── api/routes/         # health.py, tracks.py, search.py
│   │   ├── core/               # config.py, database.py, storage.py
│   │   ├── engine/             # Audio analysis engine
│   │   │   ├── embeddings.py   # CLAP audio/text embedding + DAE hybrid
│   │   │   ├── index.py        # FAISS index builder
│   │   │   ├── pipeline.py     # Original CLI pipeline (migrated)
│   │   │   ├── spectral.py     # Multi-band analysis, energy curves, stereo, section labels
│   │   │   ├── stems.py        # Demucs stem separation
│   │   │   └── tempo_bars.py   # Essentia beat/BPM/key/time-sig detection
│   │   ├── models/track.py     # SQLAlchemy models (Track, TrackSection)
│   │   ├── schemas/track.py    # Pydantic response schemas
│   │   ├── workers/analyze.py  # Background analysis + search logic
│   │   └── main.py             # FastAPI app entry point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                # Next.js App Router pages
│   │   │   ├── page.tsx        # Library (upload + track list)
│   │   │   ├── search/page.tsx # Search (text / stems / audio tabs)
│   │   │   └── tracks/[id]/    # Track detail (waveform + sections)
│   │   ├── components/
│   │   │   └── WaveformPlayer.tsx  # wavesurfer.js component
│   │   └── lib/api.ts          # API client + TypeScript types
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── .env.example
├── app/                        # Original CLI engine code (pre-platform)
└── data/                       # Local corpus + cache (gitignored)
```

## Database Models

### Track
- id, filename, original_filename, s3_key, content_type, file_size
- duration_s, bpm, key, scale, beats_per_bar
- status (pending/analyzing/ready/failed), error_message
- energy_curve (JSON: times, lufs, centroid, onset_density, low_ratio)
- created_at, updated_at

### TrackSection
- id, track_id (FK)
- start_s, end_s, bars, bar_start, bar_end
- bpm, key, scale
- hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
- section_label (intro/verse/buildup/drop/breakdown/outro)
- band_energies (JSON: 8-band dB profile -- sub through air)
- stereo_features (JSON: correlation, mid_side_ratio, width_by_band)
- embedding (LargeBinary: 512 float32, full mix CLAP)
- embedding_drums, embedding_bass, embedding_vocals, embedding_other (per-stem CLAP)

## API Endpoints

### Tracks
- `POST /api/tracks/upload` -- upload audio file, triggers background analysis
- `GET /api/tracks` -- list all tracks (filterable by status)
- `GET /api/tracks/{id}` -- track detail with all sections
- `GET /api/tracks/{id}/energy` -- track with energy curve data
- `GET /api/tracks/{id}/stream` -- presigned S3 URL for playback
- `DELETE /api/tracks/{id}` -- delete track + S3 object

### Search
- `POST /api/search` -- audio similarity search (upload query file)
- `GET /api/search/text?q=...&bars=...&k=...` -- text-to-audio search via CLAP
- `POST /api/search/stems` -- stem-weighted text search (JSON body with weights)
- `GET /api/search/compare/{a}/{b}` -- spectral + stereo + dynamics comparison of two sections
- `GET /api/search/library/stats` -- library overview stats

## Analysis Pipeline (what happens on upload)

1. File stored in S3 (MinIO)
2. Track record created in Postgres (status=pending)
3. Background worker:
   a. Downloads file from S3 to temp dir
   b. Loads mono float32 at 44.1kHz
   c. Essentia beat tracking -> BPM, key, scale, time signature, downbeat phase
   d. Computes track-level energy curve (LUFS/centroid/onset/low-ratio over time)
   e. Attempts Demucs stem separation -> drums, bass, vocals, other
   f. Loads stereo version for stereo analysis (if stereo source)
   g. For each bar size (2, 4, 8, 16):
      - Beat-aligned window slicing (with fallback to BPM-based)
      - For each window:
        - Extra features: hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
        - 8-band spectral energy profile
        - Stereo features (if stereo source)
        - Section label heuristic (intro/verse/buildup/drop/breakdown/outro)
        - Slices corresponding stem windows
   h. Batch CLAP embedding of all mix segments (N, 512)
   i. Batch CLAP embedding of each stem's segments
   j. Stores everything to Postgres
   k. Status -> ready

## Key Features Built

1. **Text-to-Audio Search** -- type "dark minimal techno kick" and search via CLAP's shared text-audio embedding space
2. **Stem-Level Similarity** -- Demucs separates tracks into drums/bass/vocals/other, each embedded separately. Search with per-stem weight sliders
3. **Multi-Band Spectral Profiles** -- 8-band energy (sub/low/low-mid/mid/high-mid/presence/brilliance/air) per section with visual bar charts
4. **Arrangement Energy Curves** -- track-level LUFS, spectral centroid, onset density, low-energy ratio over time
5. **Semantic Section Labels** -- heuristic classification as intro/verse/buildup/drop/breakdown/outro. Filterable in UI
6. **Stereo Width Analysis** -- L-R correlation, mid/side ratio, per-band stereo width
7. **DAE Hybrid Embeddings** -- function to concatenate CLAP + engineered features for composite similarity (ready but not yet wired to search)
8. **Waveform Player** -- wavesurfer.js with colored region overlays, play/pause, section click-to-play
9. **Track Detail Page** -- waveform + section cards with feature bars, spectral mini-charts, label badges, stereo width, bar/label filtering

## What Still Needs Doing

- **Docker setup**: Docker Desktop needs to be installed on the dev machine. Then `docker compose up --build` from root
- **CLAP model checkpoint**: needs to be mounted into the Docker container at the path in config (or set CLAP_CKPT env var)
- **Essentia**: needs to be installed in the Docker image (the Dockerfile may need `essentia` added -- it uses apt or pip depending on platform)
- **Git commit/push**: Droid-Shield blocked automated commits due to default dev credentials in .env.example and docker-compose.yml. User must commit manually
- **Alembic migrations**: not yet set up -- currently using `create_all` on startup
- **Authentication**: none yet -- open API
- **MuQ-MuLan**: architecture is ready for drop-in swap but not implemented
- **Energy curve visualization**: data is stored and served via API but no frontend chart component yet

## How to Run

```bash
# Prerequisites: Docker Desktop installed and running

# From project root:
docker compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/api/health
# MinIO console: http://localhost:9001 (minioadmin/minioadmin)
# Postgres: localhost:5432 (resonance/resonance)

# For frontend dev only (no backend):
cd frontend && npm install && npm run dev
```

## Git Info

- Repo: https://github.com/joebonazelli17/Project-Resonance
- Branch: main
- Last pushed commit: Platform scaffold (FastAPI + Next.js + Docker)
- Unpushed work: All enhancements (text search, stems, spectral, energy curves, labels, stereo, DAE, waveform player, track detail page)