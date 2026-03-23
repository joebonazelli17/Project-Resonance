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
- **scikit-learn RandomForest** for section classification

### Frontend (`frontend/`)
- **Next.js 15** + React 19 + TypeScript
- **Tailwind CSS** with custom `resonance` color palette (dark theme)
- **wavesurfer.js 7** with Regions plugin for waveform display + section highlighting
- **lucide-react** for icons

### Infrastructure
- **docker-compose.yml** orchestrates 5 services: Postgres 16, MinIO, FastAPI backend, persistent worker, Next.js frontend
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
2. Persistent worker polls for pending tracks with models preloaded once per process (upload route still has subprocess fallback if worker is unavailable)
3. Background analysis:
   a. Download from S3
   b. **Mastering state detection on raw audio** (crest factor, peak level, loudness histogram) -- before normalization
   c. Load mono float32 at 44.1kHz, peak-normalize to 0.95
   d. Essentia beat tracking -> BPM, key, scale, time signature, downbeat phase (with strong 4/4 prior for EDM)
   e. Track-level energy curve (LUFS/centroid/onset/low-ratio over time)
   f. Demucs stem separation -> drums, bass, vocals, other (cached to MinIO for future reanalysis)
   g. Stereo load for stereo analysis (if stereo source)
   h. **Batch section analysis** for each bar size (2, 4, 8, 16):
      - Build all windows first using beat grid / fallback bar slicing
      - Compute full-track STFT/HPSS + onset envelope once, then slice per segment
      - Extract core features: hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
      - Compute 64-band mel profiles: average, peak, variance (for search ranking)
      - Compute 8-band spectral energy (for UI), per-band crest, per-band transient density
      - Compute stereo features (if stereo source)
      - Compute per-stem section energies
      - Label sections with RandomForest classifier + rule-based fallback
   i. **Phase 1**: Persist core analysis and section metadata, then mark track as READY
   j. **Phase 2**: Batch CLAP embeddings for mix + stems after the track is already usable
   k. Update existing sections with embeddings as they finish

## API Endpoints

### Tracks
- `POST /api/tracks/upload` -- Upload audio file; persistent worker picks up the job, with subprocess fallback so the API stays responsive
- `POST /api/tracks/{id}/reanalyze` -- Delete existing sections and rerun full analysis
- `POST /api/tracks/{id}/relabel` -- Re-run section classifier only (no audio reprocessing)
- `POST /api/tracks/relabel-all` -- Re-run section classifier for all ready tracks
- `GET /api/tracks` -- List tracks (filterable by status)
- `GET /api/tracks/{id}` -- Track detail with all sections
- `GET /api/tracks/{id}/energy` -- Energy curve data
- `GET /api/tracks/{id}/stream` -- Presigned S3 URL for audio playback
- `DELETE /api/tracks/{id}` -- Soft-delete track (set status=deleted). Add `?permanent=true` to permanently delete track + S3 object.
- `GET /api/tracks/{id}/audio` -- Transcode audio to WAV for browser playback (AIFF, etc.)

### Search
- `POST /api/search` -- Upload query audio, find similar sections (blended CLAP + EQ scoring). Temporary query track is cleaned up after search completes.
- `GET /api/search/text` -- Natural language search via CLAP text encoder
- `POST /api/search/stems` -- Weighted per-stem similarity search
- `GET /api/search/reference-coach/{track_id}` -- Build an anchor-track view for an analyzed track using same-label section matches
- `GET /api/search/compare/{a}/{b}` -- Smart section comparison with mastering-aware recommendations
- `GET /api/search/library/stats` -- Library statistics (track count, bars distribution, mastering distribution)

## Search Scoring

Blended multi-signal scoring:
- `score = 0.7 * clap_cosine_sim + 0.3 * eq_profile_cosine_sim`
- HF percussive ratio hard gate (reject spectrally incompatible matches)
- DAE hybrid embeddings as optional mode (CLAP + engineered features)
- Reference Coach aggregates best section matches into a single **anchor track** using similarity + coverage across the uploaded track
- Reference Coach prefers **same-label matches** when section labels are available, falling back gracefully when they are not
- Compare endpoint: spectral shape delta relative to 1kHz anchor, distinguishes arrangement differences (transient density) from tonal differences (EQ-fixable), adds stem-balance deltas, and emits mastering-aware recommendations with severity levels

## Docker & Infrastructure Notes

- Backend container forced to `platform: linux/amd64` for essentia wheel compatibility
- MinIO API on host port **9002** (9000 was in use). Console on **9001**.
- Dockerfile downloads CLAP (~600MB), RoBERTa (~500MB), BERT vocab, Demucs (~80MB) at build time
- Essentia pinned to `2.1b6.dev1389` (Jul 2025 release with prebuilt x86_64 manylinux wheels)
- `app.core.patches` module monkey-patches HuggingFace `from_pretrained` to load from `/models/` (imported by both main.py and standalone workers)
- Persistent worker preloads heavy models once, polls every few seconds for `PENDING` tracks, and reuses models across analyses
- `spawn_analysis()` still exists as a fallback path so the API stays responsive if the worker is not running
- Stem separations are cached in MinIO as `.npz` blobs to accelerate reanalysis
- Background worker uses sync psycopg2 (not async asyncpg) to avoid event loop conflicts
- `./backend:/app` volume mount with `--reload` -- code changes picked up live (note: auto-reload kills in-flight analysis)
- **Zscaler**: EY corporate proxy blocks huggingface.co. First build must be on unrestricted network (home Mac). Then `docker save` the image and transfer to work PC.

## Current Status

### Implemented & verified (code review complete)
- Full-stack scaffolding: upload, list, delete (soft-delete by default), stream tracks
- Essentia beat/BPM/key/time-sig detection (gold standard, no fallbacks)
- Demucs stem separation (drums/bass/vocals/other)
- CLAP audio + text embeddings (512-dim, L2-normalized)
- ML section classifier trained from hand-labeled ground truth tracks, with relabel-only endpoints for fast iteration
- 4/4 bar-grid fix for EDM tracks (prevents inflated bar counts from incorrect 3/4 detection)
- Deep spectral analysis: 64-band mel profiles (avg/peak/variance), 8-band energy/crest/transient density
- Mastering state detection (runs on raw audio before normalization)
- Section labeling with confidence scores, stem energies, trajectory features, and rule-based fallback
- Blended search scoring (CLAP + EQ profile cosine + HF gating)
- Smart comparison endpoint with mastering-aware, arrangement-vs-EQ-aware recommendations plus stem-balance deltas
- Stereo width analysis (correlation, mid/side ratio, per-band width)
- Track energy curves (LUFS, centroid, onset density, low ratio over time)
- Text search and weighted stem search via CLAP shared embedding space
- Waveform player with section regions, click-to-seek overlay, section card seeking
- Track detail page with section cards, bars/label filters, spectral profile visualization
- **Reference Coach** on the track detail page: anchor track summary, per-section best match, alternate matches, and actionable guidance
- Dynamic waveform regions: update when switching bar size or section type filters
- Search page with text/stem/audio modes, stem weight sliders
- Library page with drag-and-drop upload, status badges, elapsed timer, auto-refresh
- AIFF browser playback via ffmpeg transcode endpoint (AIFF -> WAV)
- Delete confirmation dialogs on library and track detail pages

### Performance optimizations (latest)
- **Two-phase analysis**: Phase 1 now stores core analysis only and marks the track READY before CLAP finishes. Mix + stem embeddings run entirely in Phase 2.
- **Persistent worker**: Models load once and stay warm across tracks instead of cold-loading every job.
- **Relabel-only workflow**: Section-classifier iteration now takes seconds instead of full audio reprocessing.
- **Stem cache**: Demucs output is cached in MinIO and reused on reanalysis.
- **Hop size 4**: Changed `DEFAULT_HOP_BARS` from 2 to 4, cutting segment count roughly in half (~260 segments for a 6-min track vs ~472).
- **Thread count 6**: Changed `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `TORCH_NUM_THREADS` from 1 to 6 on a 6-core Ryzen.
- **Parallel Demucs + beat detection**: ThreadPoolExecutor runs stem separation and beat/BPM/key detection concurrently
- **Pre-resample to 48kHz**: Full track + stems resampled once upfront, then sliced per section (eliminates per-segment resample operations)
- **Cached mel spectrograms**: `compute_eq_profiles()` computes avg/peak/variance from single mel spectrogram (was 3 separate spectrograms per section)
- **Combined STFT**: `compute_band_crest_and_transient_density()` computes both from single STFT (was 2 per section)
- **Batch feature extraction**: full-track STFT/HPSS + onset envelope are computed once and sliced per segment, cutting feature extraction roughly 3x on a 252-section track
- **CLAP deferred from Phase 1**: section matching embeddings no longer block initial usability
- **GPU auto-detection**: Startup logs platform, torch version, GPU availability, thread count
- **Timing instrumentation**: Each pipeline step logs elapsed time for profiling
- **Observed performance on current CPU setup**: ~2.3 min to READY for cached reanalysis, ~8.7 min to READY for a brand-new track; remaining first-pass floor is mostly Demucs CPU time

### Infrastructure fixes (latest)
- **Next.js API proxy**: Frontend uses `rewrites()` in next.config.ts to proxy `/api/*` to backend. Eliminates CORS and IPv4/IPv6 issues. Upload goes directly to backend to bypass proxy body size limits.
- **HF patches extracted to `app.core.patches`**: Shared module imported by both `main.py` and standalone workers -- fixes Zscaler SSL errors when running analysis outside uvicorn
- **CORS**: Allows both `localhost:3000` and `127.0.0.1:3000` origins
- **Enum case fix**: TrackStatus values uppercase to match Postgres enum, schema validator lowercases for frontend
- **Soft delete**: Delete endpoint sets status to DELETED by default, permanent delete requires `?permanent=true`
- **Docker frontend disabled for dev**: Stop Docker frontend container, run `npx next dev` locally for live code changes

### Bug fixes applied
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
11. AIFF browser playback failure -- fixed: transcode endpoint via ffmpeg
12. Waveform overcrowded with overlapping regions -- fixed: default 8-bar filter, non-overlapping dedup
13. Section card click not seeking waveform -- fixed: activeStartS + seekCounter props
14. Waveform click-to-seek blocked by region overlays -- fixed: transparent click overlay at z-10

### UI fixes (latest)
- **Spectral profile A-weighted**: Raw band energies are perceptually weighted (A-weighting curve) and peak-normalized per section, so the display matches what you'd see on a spectrum analyzer. Without this, low frequencies always dominated due to physics.
- **Section cards sorted**: Cards now sorted by start time (ascending), then by bar size.
- **Waveform regions on "All" filter**: When no specific bar filter is selected, waveform regions use the largest available bar size to avoid overlap. Previously, mixing all bar sizes made most regions invisible.
- **Reference Coach auto-refresh**: Track page now refreshes coaching automatically while analysis / embeddings finish, instead of making the user manually reload.

### Completed milestones
- Docker build on home Mac with `linux/amd64` (all models downloaded successfully)
- First real track upload and end-to-end pipeline validation (328 sections, all stems, CLAP embeddings)
- Docker image transfer to Windows work PC via `docker save` / `docker load`
- All 5 services running on Windows (db, minio, backend, worker, frontend)
- Waveform playback working with AIFF transcoding and click-to-seek
- Analysis pipeline optimized (persistent worker, stem caching, batch features, deferred CLAP)
- Ground-truth-labeled training set assembled and classifier retrained from real tracks
- Reference Coach slice shipped on the analyzed-track page
- Spectral profile visualization calibrated with A-weighting for perceptually accurate display

### Pending / next steps (prioritized)
- **Broaden labeled evaluation set**: current classifier is useful, but it still needs more hand-labeled tracks to prove generalization across more song structures
- **Deepen the Reference Coach UX**: A/B auditioning, anchor-track review flow, and better “what should I do next?” sequencing
- **Improve recommendation orchestration**: move from raw per-comparison tips to smarter section-level coaching summaries across the full song
- **Tighten section-type gating across all search modes**: Reference Coach now uses same-label matching; the general search endpoints still need the same discipline
- **Validate search quality with real corpus (multiple tracks)**
- Controlled ground truth testing with modified stems (see testing strategy below)
- Speed/infrastructure path for scale: GPU-backed analysis + bulk ingest queue for large libraries
- Alembic migrations setup
- Authentication layer

## Product Vision: Component-Level Referencing

### Core concept
Evolving from track-level similarity to component-level similarity. Instead of "what track sounds like mine?" we ask "which parts of my track match which parts of professional tracks?"

### Anchor Track + Component Insights (critical UX model)
Avoid "Frankenstein references" -- we do NOT recommend "use Track A drums, Track B bass, Track C vocals." This breaks musical cohesion.

Status: **first implementation slice shipped**. The analyzed-track page now selects an anchor track, shows best per-section matches, alternates, and comparison guidance. The full product vision below is still the target state.

**Step 1 -- Anchor Track (Primary Reference):**
Select best overall match, gated by section type (drop matches drop, verse matches verse) and similar BPM/structure.

**Step 2 -- Component Comparison:**
Compare user track vs anchor per-stem:
```
Drums:    42%
Bass:     85%
Melodics: 91%
Vocals:   78%
```

**Step 3 -- Insight Layer:**
System translates metrics into human guidance:
- "Melodic elements are already near professional level."
- "Drums have significantly lower energy and transient density."
- "Bass is stylistically aligned but lacks punch."

**Step 4 -- Optional Enhancement Suggestions:**
If a component is weak, show closest component-specific matches as *secondary references*, not replacements:
```
Drums are below reference quality.
Closest drum matches:
  - Track X (91%)
  - Track Y (88%)
```

### Section-type gating (required constraint)
All similarity search must be gated by section label: drop matches drop, verse matches verse, build matches build. Otherwise results become misleading.

Current state:
- **Reference Coach** now prefers same-label section matches when labels are available
- General search endpoints still mostly match by bar count / embeddings / HF gating and need stronger section-type constraints

### User-controlled stem weighting
User can specify focus areas: "Drums + Bass focus", "Ignore vocals", "Everything except vocals." Already partially implemented in stem search -- needs to be integrated into the anchor track flow.

### Version Comparison Mode
User uploads two versions of their own track (v1, v2). System outputs per-component diff:
```
Drums: +4.2 dB energy
Transient density: +18%
Stereo width: unchanged
```
This transforms Resonance into a mix iteration tool in addition to a reference tool.

### UX Principles
1. Start with anchor track (cohesion)
2. Layer component insights (diagnosis)
3. Offer optional alternatives (optimization)
4. Never overwhelm user with raw metrics
5. Translate everything into actionable guidance

## Testing Strategy

### Controlled ground truth testing
Take a professional reference track, split stems, modify specific elements, re-upload. Verify the system detects the right changes.

### Test types
1. **Stem gain tests**: drums +6dB, bass -6dB. Expected: drum similarity increases, bass similarity decreases.
2. **EQ tests**: boost highs, cut lows. Expected: spectral shift detected, transient density unchanged.
3. **Transient density tests**: double hats (1/8 to 1/16), remove percussive elements. Expected: transient density changes without large spectral change.
4. **Compression tests**: apply limiter, change crest factor. Expected: crest decreases, loudness increases.
5. **Arrangement tests**: remove stems entirely, change section structure.

### Validation goal
System should be directionally correct -- not necessarily perfectly precise (Demucs separation is imperfect), but consistently pointing in the right direction.

## Future Roadmap

### Tier 1: Core Differentiators
- **Expand Reference Coach into full Anchor Track + Component Breakdown**: richer anchor selection review, better component deltas, and cleaner section-by-section coaching workflow
- **Phrase-by-phrase cross-reference matching**: Timeline UI showing your sections mapped to best-matching reference sections across the entire library
- **Self-learning feedback loop**: User confirms/rejects matches, system tunes scoring weights per-genre over time
- **A/B spectral diff visualization**: Side-by-side spectrograms with delta heat map overlay, arrangement vs EQ distinction

### Tier 2: Producer Workflow
- **Mix Coach mode**: Guided workflow sequencing comparison recommendations into actionable checklists
- **Version Comparison Mode**: Upload v1 and v2 of a mix, get per-component diff showing what changed
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