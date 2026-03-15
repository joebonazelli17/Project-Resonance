# Project Resonance -- Status & Architecture

## The Vision

An AI Music Intelligence Platform for music producers. Two core products:

1. **Reference Track Engine** (current focus) -- A producer uploads their track. The system breaks it into sections (2, 4, 8, 16 bars -- eventually user-configurable) and for each section, finds the most similar sections from a library of professional reference tracks. Different reference tracks can match different sections. The goal: a huge library of professional reference tracks where the system intelligently matches section-by-section, giving producers a precise map of "your verse sounds most like this reference's verse, your drop matches this other reference's drop."

2. **AI Mix Coach** (future) -- Loudness/spectral/dynamics feedback vs genre norms. Tells producers what to adjust and why.

The long-term goal is that advanced professional producers say "I can't work without this." It needs to be gold-standard, novel, and deeply useful -- not just a tech demo.

## Product Philosophy

### What makes this different from existing tools

Existing tools (Mastering The Mix REFERENCE, iZotope Audiolens) do static A/B comparison between your track and a single reference. They show average spectral curves and say "boost here." But:

- **Average spectral curves are misleading.** If the reference has 1/16th hats and you have 1/8th hats, the reference will show more high-end energy. But boosting your existing hats to match the average makes them too bright -- the real difference is *arrangement density* (more rhythmic events), not EQ balance. Our tool needs to distinguish between tonal differences (EQ-fixable) and arrangement differences (not EQ-fixable).

- **Mastering state skews everything.** A mastered reference has been squashed through a limiter. Comparing a pre-master mix to that will always show "undercompressed." REFERENCE plugin's compression feedback led the creator of this project to over-compress early mixes -- the tool was comparing mastering-stage characteristics to a mix-stage track. Our tool must detect mastering state and contextualize all feedback accordingly.

- **Section-level matching is novel.** No existing tool finds reference matches per-section across a library. They require you to manually choose a reference. We automate discovery.

### Key insights to preserve

1. **Peak vs average vs variance per frequency band**: Two tracks can have identical average spectral curves but completely different transient content and arrangement density. We need to analyze all three dimensions to give accurate feedback.

2. **Per-band transient density**: The 1/16th vs 1/8th hat problem. Counting onset events per second per frequency band reveals whether a spectral difference is due to more events or louder events.

3. **Per-band crest factor**: Reveals compression state per frequency range. A mastered track has low crest across all bands. A mix might have high crest in the sub (uncompressed bass) but low crest in mids (compressed vocals).

4. **Mastering-aware recommendations**: If reference is mastered and user track isn't, exclude dynamics/loudness advice. Focus on spectral shape (relative balance), stereo decisions, and arrangement density.

5. **Spectral shape comparison should be relative**: Compare curves relative to a 1kHz anchor point, not in absolute dB. Absolute dB is meaningless when one track has been through a limiter.

6. **Mastering detection must run on raw audio**: Peak normalization (which the pipeline applies for analysis consistency) makes every track's peak ~0dBFS, defeating peak-level-based mastering detection. Detection runs before normalization.

### Future enhancements (captured for continuity)

- **User-configurable chunk size**: Let users choose bar size for section matching
- **Match confirmation flow**: Queue up each section pair (user + reference), let user confirm/reject similarity
- **Self-learning feedback loop**: When users agree/disagree with match quality, use that signal to tune scoring weights and thresholds over time. This is a major differentiator -- the system gets better with use.
- **MuQ-MuLan model swap**: Architecture ready for drop-in replacement of CLAP with newer embedding models

## Tech Stack

### Backend (`backend/`)
- **FastAPI** (async, Python 3.11+)
- **PostgreSQL** via SQLAlchemy 2.0 (async via asyncpg for API routes, sync via psycopg2 for background workers)
- **MinIO** (S3-compatible) for audio file storage
- **CLAP** (LAION, HTSAT-base) for audio + text embeddings (512-dim shared space)
- **Essentia 2.1b6** for beat tracking, BPM, key, time signature detection (prebuilt x86_64 wheel, pinned version)
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
- **Backend forced to `linux/amd64`** in docker-compose for cross-platform compatibility (essentia has no Linux aarch64 wheels; QEMU emulation on M1 Mac, native on Windows)
- **Dockerfiles** for both backend and frontend (standalone Next.js output)

## Database Models

### Track
- id, filename, original_filename, s3_key, content_type, file_size
- duration_s, bpm, key, scale, beats_per_bar
- status (pending/analyzing/ready/failed), error_message
- energy_curve (JSON: times, lufs, centroid, onset_density, low_ratio)
- mastering_state (string: "mastered", "pre_master", "unknown")
- created_at, updated_at

### TrackSection
- id, track_id (FK)
- start_s, end_s, bars, bar_start, bar_end
- bpm, key, scale
- hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
- section_label, section_label_confidence
- band_energies (JSON: 8-band dB profile -- for UI display)
- stereo_features (JSON: correlation, mid_side_ratio, width_by_band)
- band_crest (JSON: 8-band peak-to-RMS ratio -- compression state per band)
- band_transient_density (JSON: 8-band onset events per second)
- embedding (LargeBinary: 512 float32, full mix CLAP)
- embedding_drums/bass/vocals/other (per-stem CLAP)
- eq_profile (LargeBinary: 64 float32, average mel profile -- for search ranking)
- eq_profile_peak (LargeBinary: 64 float32, peak mel profile)
- eq_profile_variance (LargeBinary: 64 float32, spectral movement)

## Analysis Pipeline

1. File stored in S3 (MinIO), track record created (status=pending)
2. Background worker:
   a. Download from S3
   b. **Mastering state detection on raw audio** (crest factor, peak level, loudness histogram) -- before normalization
   c. Load mono float32 at 44.1kHz, peak-normalize to 0.95
   d. Essentia beat tracking -> BPM, key, scale, time signature, downbeat phase
   e. Track-level energy curve (LUFS/centroid/onset/low-ratio over time)
   f. Demucs stem separation -> drums, bass, vocals, other
   g. Stereo load for stereo analysis (if stereo source)
   h. **Two-pass section analysis** for each bar size (2, 4, 8, 16):
      - Pass 1: Collect all segments, compute per-segment features, build percentile distributions (p25/p50/p75 for RMS, onset density, HF ratio)
      - Pass 2: Compute all features with relative thresholds using track percentiles:
        - Core features: hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
        - 64-band mel profiles: average, peak, variance (for search ranking)
        - 8-band spectral energy (for UI), per-band crest, per-band transient density
        - Stereo features (if stereo source)
        - Section label using relative thresholds + confidence scoring
        - Stem window slicing for per-stem embeddings
   i. Batch CLAP embedding of all mix + stem segments
   j. Store everything to Postgres, status -> ready

## API Endpoints

### Tracks
- `POST /api/tracks/upload` -- Upload audio file, triggers background analysis
- `GET /api/tracks` -- List tracks (filterable by status)
- `GET /api/tracks/{id}` -- Track detail with all sections
- `GET /api/tracks/{id}/energy` -- Energy curve data
- `GET /api/tracks/{id}/stream` -- Presigned S3 URL for audio playback
- `DELETE /api/tracks/{id}` -- Delete track + S3 object

### Search
- `POST /api/search` -- Upload query audio, find similar sections (blended CLAP + EQ scoring). Temporary query track is cleaned up after search completes.
- `GET /api/search/text` -- Natural language search via CLAP text encoder
- `POST /api/search/stems` -- Weighted per-stem similarity search
- `GET /api/search/compare/{a}/{b}` -- Smart section comparison with mastering-aware recommendations
- `GET /api/search/library/stats` -- Library statistics (track count, bars distribution, mastering distribution)

## Search Scoring

Blended multi-signal scoring:
- `score = 0.7 * clap_cosine_sim + 0.3 * eq_profile_cosine_sim`
- HF percussive ratio hard gate (reject spectrally incompatible matches)
- DAE hybrid embeddings as optional mode (CLAP + engineered features)
- Compare endpoint: spectral shape delta relative to 1kHz anchor, distinguishes arrangement differences (transient density) from tonal differences (EQ-fixable), mastering-aware recommendations with severity levels

## Docker & Infrastructure Notes

- Backend container forced to `platform: linux/amd64` for essentia wheel compatibility
- MinIO API on host port **9002** (9000 was in use). Console on **9001**.
- Dockerfile downloads CLAP (~600MB), RoBERTa (~500MB), BERT vocab, Demucs (~80MB) at build time
- Essentia pinned to `2.1b6.dev1389` (Jul 2025 release with prebuilt x86_64 manylinux wheels)
- `main.py` monkey-patches HuggingFace `from_pretrained` to load from `/models/`
- Background worker uses sync psycopg2 (not async asyncpg) to avoid event loop conflicts
- `./backend:/app` volume mount with `--reload` -- code changes picked up live
- **Zscaler**: EY corporate proxy blocks huggingface.co. First build must be on unrestricted network (home Mac). Then `docker save` the image and transfer to work PC.

## Current Status

### Implemented & verified (code review complete)
- Full-stack scaffolding: upload, list, delete, stream tracks
- Essentia beat/BPM/key/time-sig detection (gold standard, no fallbacks)
- Demucs stem separation (drums/bass/vocals/other)
- CLAP audio + text embeddings (512-dim, L2-normalized)
- Deep spectral analysis: 64-band mel profiles (avg/peak/variance), 8-band energy/crest/transient density
- Mastering state detection (runs on raw audio before normalization)
- Two-pass section labeling with relative thresholds and confidence scores
- Blended search scoring (CLAP + EQ profile cosine + HF gating)
- Smart comparison endpoint with mastering-aware, arrangement-vs-EQ-aware recommendations
- Stereo width analysis (correlation, mid/side ratio, per-band width)
- Track energy curves (LUFS, centroid, onset density, low ratio over time)
- Text search and weighted stem search via CLAP shared embedding space
- Waveform player with section regions, click-to-play, active section highlighting
- Track detail page with section cards, bars/label filters, spectral profile visualization
- Search page with text/stem/audio modes, stem weight sliders
- Library page with drag-and-drop upload, status badges, auto-refresh

### Bug fixes applied (latest commit)
1. `sections_from_audio` 4-tuple unpacking crash -- fixed
2. Mastering detection after peak normalization (always ~0dBFS) -- fixed: now runs on raw audio
3. Orphan query track records accumulating in DB -- fixed: cleanup in finally block
4. Empty `bars_distribution` in library stats -- fixed: now computes real distributions
5. `band_transient_density` onset detection with wrong input format -- fixed: proper power-to-dB pipeline
6. Duplicated `_eq_profile` in pipeline.py vs spectral.py -- fixed: single source of truth
7. Frontend missing `band_crest`, `band_transient_density`, `section_label_confidence` types -- fixed
8. `compareSections` untyped response -- fixed: proper `ComparisonResult` interface
9. WaveformPlayer not updating region colors on section selection -- fixed: useEffect on activeSectionId
10. ~80 lines of dead commented-out code -- removed

### Pending / next steps
- Docker build on home Mac (in progress -- `docker compose up --build` with `linux/amd64`)
- First real track upload and end-to-end pipeline validation
- Transfer Docker image to Windows work PC (`docker save` / `docker load`)
- Validate search quality with real corpus
- Alembic migrations setup
- Authentication layer

## Future Roadmap

### Tier 1: Core Differentiators
- **Phrase-by-phrase cross-reference matching**: Timeline UI showing your sections mapped to best-matching reference sections across the entire library
- **Self-learning feedback loop**: User confirms/rejects matches, system tunes scoring weights per-genre over time
- **A/B spectral diff visualization**: Side-by-side spectrograms with delta heat map overlay, arrangement vs EQ distinction

### Tier 2: Producer Workflow
- **Mix Coach mode**: Guided workflow sequencing comparison recommendations into actionable checklists
- **Genre-aware analysis presets**: Different spectral weighting for EDM/hip-hop/acoustic/etc.
- **Reference collection sharing**: Curate and share community reference track libraries

### Tier 3: Technical Advancement
- **Real-time WebSocket progress**: Replace polling with live analysis progress updates
- **MuQ-MuLan model swap**: Drop-in replacement for CLAP with newer embedding model
- **Offline desktop app**: Electron wrapper with embedded Python backend
- **DAW plugin bridge**: AU/VST plugin sending audio directly to Resonance for real-time feedback

## Git Info

- Repo: https://github.com/joebonazelli17/Project-Resonance
- Branch: main