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

6. **The original CLI pipeline had features we dropped**: The 64-band mel EQ profile and blended search scoring (CLAP + EQ + HF gating) were more sophisticated than what we replaced them with. The 8-band display profile is fine for UI but too coarse for search. We need both.

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
   a. Download from S3, load mono float32 at 44.1kHz
   b. Essentia beat tracking -> BPM, key, scale, time signature, downbeat phase
   c. Track-level energy curve (LUFS/centroid/onset/low-ratio over time)
   d. Mastering state detection (crest factor, peak-to-LUFS, loudness histogram)
   e. Demucs stem separation -> drums, bass, vocals, other
   f. Stereo load for stereo analysis (if stereo source)
   g. For each bar size (2, 4, 8, 16), beat-aligned window slicing:
      - Core features: hf_perc_ratio, rms_dbfs, peak_dbfs, crest_db, flatness
      - 64-band mel profiles: average, peak, variance
      - 8-band spectral energy (for UI), crest, transient density
      - Stereo features, section label (relative thresholds) + confidence
      - Stem window slicing
   h. Batch CLAP embedding of all mix + stem segments
   i. Store everything to Postgres, status -> ready

## Search Scoring

Blended multi-signal scoring:
- `score = w_clap * clap_sim + w_eq * eq_profile_cosine_sim + w_feat * feature_sim`
- HF percussive ratio hard gate (skip spectrally incompatible matches)
- Loudness normalization before comparison
- DAE hybrid embeddings as optional mode (CLAP + engineered features)
- All weights configurable per search request

## Docker & Infrastructure Notes

- MinIO API on host port **9002** (9000 was in use). Console on **9001**.
- Dockerfile downloads CLAP (~600MB), RoBERTa (~500MB), BERT vocab, Demucs (~80MB) at build time.
- `main.py` monkey-patches HuggingFace `from_pretrained` to load from `/models/`.
- Background worker uses sync psycopg2 (not async asyncpg) to avoid event loop conflicts.
- `./backend:/app` volume mount with `--reload` -- code changes picked up live.
- **Zscaler**: EY corporate proxy blocks huggingface.co. First build must be on unrestricted network.

## Current Status

### Working
- Full-stack scaffolding, upload/list/delete tracks
- Essentia beat/BPM/key, Demucs stems, CLAP embeddings
- 8-band spectral profiles, stereo width, energy curves
- Waveform player, track detail page, text/stem search

### In progress (current sprint)
- Phase 1: Deep spectral analysis (64-band profiles, per-band crest, transient density)
- Phase 2: Mastering state detection
- Phase 3: Blended search scoring
- Phase 4: Relative section labels with confidence
- Phase 5: Smart comparison recommendations
- Phase 6: Energy curve frontend visualization

## Git Info

- Repo: https://github.com/joebonazelli17/Project-Resonance
- Branch: main