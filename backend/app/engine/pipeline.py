# app/pipeline.py
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import json, hashlib, time, os
from datetime import datetime
from app.engine.tempo_bars import detect_beats_bpm_key, slice_by_bars_from_beats
from app.engine.embeddings import embed_audio_batch, _get_clap
from app.engine.index import build_index



AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif"}
MODEL_TAG = "clap-htsat-base-512@music_audioset_epoch15"
CACHE_VERSION = "v3"

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

def _sec_to_mmss(x: float) -> str:
    """Convert seconds (float) to M:SS.ss string (e.g., 100.345 -> '1:40.35')"""
    m = int(x // 60)
    s = round(x % 60, 2)
    return f"{m}:{s:05.2f}"

def file_cache_key(fp: Path) -> str:
    st = fp.stat()
    h = hashlib.sha1()
    h.update(str(fp.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]

def track_cache_path(cache_dir: Path, fp: Path) -> Path:
    cache_dir = cache_dir / "tracks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{file_cache_key(fp)}__{fp.name}.npz"

def index_cache_paths(cache_dir: Path) -> dict:
    p = cache_dir / "index"
    p.mkdir(parents=True, exist_ok=True)
    return {
        "X": p / "X.f32.npy",
        "meta": p / "meta.parquet",
        "faiss": p / "faiss.index",
        "manifest": p / "manifest.json",
        "dir": p
    }


def current_corpus_tracklist(files: list[Path]) -> list[dict]:
    out = []
    for fp in files:
        st = fp.stat()
        out.append({
            "name": fp.name,
            "key": file_cache_key(fp),   # already incorporates path, size, mtime
            "size": int(st.st_size),
            "mtime": int(st.st_mtime),
        })
    # stable ordering
    return sorted(out, key=lambda d: (d["name"], d["key"]))

def save_track_npz(out_fp: Path, emb: np.ndarray, rows: list[dict], params: dict):
    # rename 'file' column -> 'filename' to avoid collision with np.savez_compressed(file=...)
    cols = {
        "filename": np.array([r["file"] for r in rows]),
        "start_s": np.array([r["start_s"] for r in rows]),
        "end_s": np.array([r["end_s"] for r in rows]),
        "bars": np.array([r["bars"] for r in rows]),
        "bpm": np.array([r["bpm"] for r in rows]),
        "key": np.array([r["key"] for r in rows]),
        "scale": np.array([r["scale"] for r in rows]),
        "b0": np.array([r["b0"] for r in rows]),
        "b1": np.array([r["b1"] for r in rows]),
        "hf_perc_ratio": np.array([r["hf_perc_ratio"] for r in rows]),
        "rms_dbfs": np.array([r["rms_dbfs"] for r in rows]),
        "peak_dbfs": np.array([r["peak_dbfs"] for r in rows]),
        "crest_db": np.array([r["crest_db"] for r in rows]),
        "flatness": np.array([r["flatness"] for r in rows]),
    }
    np.savez_compressed(
        out_fp,
        emb=emb.astype(np.float32),
        **cols,
        _params=json.dumps(params)
    )


def load_track_npz(fp: Path):
    z = np.load(fp, allow_pickle=True)
    emb = z["emb"].astype(np.float32)
    rows = []
    for i in range(emb.shape[0]):
        rows.append({
            "file": str(z["filename"][i]),
            "start_s": float(z["start_s"][i]),
            "end_s": float(z["end_s"][i]),
            "bars": int(z["bars"][i]),
            "bpm": (None if str(z["bpm"][i]) == "nan" else float(z["bpm"][i])),
            "key": (None if str(z["key"][i]) == "None" else str(z["key"][i])),
            "scale": (None if str(z["scale"][i]) == "None" else str(z["scale"][i])),
            "b0": int(z["b0"][i]),
            "b1": int(z["b1"][i]),
            "hf_perc_ratio": float(z["hf_perc_ratio"][i]),
            "rms_dbfs": float(z["rms_dbfs"][i]),
            "peak_dbfs": float(z["peak_dbfs"][i]),
            "crest_db": float(z["crest_db"][i]),
            "flatness": float(z["flatness"][i]),
        })
    params = json.loads(z["_params"].item() if hasattr(z["_params"], "item") else str(z["_params"]))
    return emb, rows, params



def load_mono(fp: Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load mono float32, peak-normalize, resample to 44.1k (Essentia-calibrated)."""
    y, sr = librosa.load(str(fp), sr=target_sr, mono=True)
    if y.size:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = (y / peak) * 0.95
    return y.astype(np.float32), target_sr

def _slice_by_bpm_fallback(bpm: float, duration_s: float, bars: int, hop_bars: int, beats_per_bar: int = 4):
    if not bpm or bpm <= 0:
        return []
    sec_per_bar = (60.0 / float(bpm)) * beats_per_bar
    win = sec_per_bar * bars
    hop = sec_per_bar * hop_bars
    t = 0.0
    out = []
    eps = 1e-3  # allow millisecond rounding wiggle room
    while t + win <= duration_s + eps:
        out.append((t, t + win))
        t += hop
    return out

from app.engine.spectral import compute_eq_profile as _eq_profile

def _extra_features(y: np.ndarray, sr: int) -> dict:
    """Compute quick tonal/compression cues for gating & analysis."""
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = y.astype(np.float32, copy=False)

    # HPSS split
    H, P = librosa.decompose.hpss(librosa.stft(y, n_fft=2048, hop_length=512))
    mag_H, mag_P = np.abs(H), np.abs(P)
    hf_band = int(mag_P.shape[0] * 0.8)
    hf_perc_ratio = float(mag_P[hf_band:, :].sum() / (mag_P.sum() + 1e-8))

    # RMS / peak
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest = float(20 * np.log10((peak + 1e-6) / (rms + 1e-6)))

    # Spectral flatness (overall tonality proxy)
    flat = float(librosa.feature.spectral_flatness(y=y).mean())

    return {
        "hf_perc_ratio": hf_perc_ratio,
        "rms_dbfs": 20 * np.log10(rms + 1e-9),
        "peak_dbfs": 20 * np.log10(peak + 1e-9),
        "crest_db": crest,
        "flatness": flat,
    }


class BatchFeatureExtractor:
    """Pre-compute full-track STFT/HPSS once, then extract per-segment features via slicing."""

    def __init__(self, y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
        if y.ndim > 1:
            y = y.mean(axis=0)
        self.y = y.astype(np.float32, copy=False)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

        # One STFT + HPSS for the entire track
        S = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        self.H, self.P = librosa.decompose.hpss(S)
        self.mag_H = np.abs(self.H)
        self.mag_P = np.abs(self.P)
        self.hf_band = int(self.mag_P.shape[0] * 0.8)

        # Full-track spectral flatness frames
        self.flatness_frames = librosa.feature.spectral_flatness(S=np.abs(S))[0]

    def _sample_to_frame(self, sample_idx: int) -> int:
        return sample_idx // self.hop_length

    def extract(self, s_i: int, e_i: int) -> dict:
        """Extract features for a segment defined by sample indices."""
        seg = self.y[s_i:e_i]

        # Slice pre-computed STFT frames
        f_start = self._sample_to_frame(s_i)
        f_end = self._sample_to_frame(e_i)
        f_end = max(f_end, f_start + 1)

        mag_P_seg = self.mag_P[:, f_start:f_end]
        hf_perc_ratio = float(mag_P_seg[self.hf_band:, :].sum() / (mag_P_seg.sum() + 1e-8))

        # RMS / peak from audio directly (cheap)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        peak = float(np.max(np.abs(seg)))
        crest = float(20 * np.log10((peak + 1e-6) / (rms + 1e-6)))

        # Slice pre-computed flatness
        flat_seg = self.flatness_frames[f_start:f_end]
        flat = float(flat_seg.mean()) if flat_seg.size > 0 else 0.0

        return {
            "hf_perc_ratio": hf_perc_ratio,
            "rms_dbfs": 20 * np.log10(rms + 1e-9),
            "peak_dbfs": 20 * np.log10(peak + 1e-9),
            "crest_db": crest,
            "flatness": flat,
        }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _dedup_per_track(
    out: pd.DataFrame,
    iou_thresh: float = 0.5,
    eq_weight: float = 0.0,
    keep_per_window: int = 1,
) -> pd.DataFrame:
    if out.empty:
        return out

    # Use the right ranking key
    key_col = "blend" if eq_weight > 0.0 and "blend" in out.columns else "sim"

    # Normalize a numeric window id (seconds). Your table later formats mm:ss; here we keep it float.
    # If q_start is already in seconds, this will be a no-op.
    qstart = out["q_start"]
    if qstart.dtype == object:
        # if it's already "M:SS" strings, convert back to seconds safely
        def _to_sec(x):
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x)
            if ":" in s:
                m, ss = s.split(":")
                return 60.0 * float(m) + float(ss)
            try:
                return float(s)
            except Exception:
                return np.nan
        out = out.copy()
        out["q_start_sec"] = out["q_start"].apply(_to_sec)
        out = out[pd.notna(out["q_start_sec"])]
        out["q_start_sec"] = out["q_start_sec"].astype(float)
        qstart_col = "q_start_sec"
    else:
        qstart_col = "q_start"

    kept = []

    # Dedup independently per match_file, but protect per-window coverage
    for match_name, g in out.groupby("match_file", sort=False):
        g = g.sort_values([qstart_col, key_col], ascending=[True, False]).reset_index(drop=True)

        # Track how many survivors each window still has
        win_counts = g.groupby(qstart_col).size().to_dict()

        used = np.zeros(len(g), dtype=bool)
        keep_idx = []

        for i in range(len(g)):
            if used[i]:
                continue
            # Always tentatively keep the current best
            keep_idx.append(i)
            s1, e1 = float(g.iloc[i]["m_start"]), float(g.iloc[i]["m_end"])

            # Try to suppress overlaps that would collide with this region
            for j in range(i + 1, len(g)):
                if used[j]:
                    continue
                s2, e2 = float(g.iloc[j]["m_start"]), float(g.iloc[j]["m_end"])
                inter = max(0.0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter + 1e-9
                iou = inter / union

                if iou >= iou_thresh:
                    win = g.iloc[j][qstart_col]
                    # Only drop if that window would still keep >= keep_per_window rows
                    if win_counts.get(win, 0) > keep_per_window:
                        used[j] = True
                        win_counts[win] = win_counts.get(win, 0) - 1
                    # else: protect coverage for that window; do not drop

        kept.append(g.loc[sorted(set(keep_idx + np.where(~used)[0].tolist()))])

    res = pd.concat(kept, ignore_index=True)

    # Final tidy: within each window, re-sort by your score
    res = (
        res.sort_values([qstart_col, key_col], ascending=[True, False])
           .reset_index(drop=True)
    )
    return res


def ingest_corpus(
    corpus_dir: Path,
    bars_list=(2, 4, 8, 16),
    hop_bars=8,
    beats_per_bar=4,
    cache_dir: Path = Path("data/cache"),
    use_cache: bool = True,
    rebuild_index: bool = False,
):
    t_ingest = time.time() 
    files = [p for p in sorted(Path(corpus_dir).iterdir()) if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise RuntimeError(f"No audio files found in {corpus_dir}")
    
    want_index_params = {
        "model_tag": MODEL_TAG,
        "cache_version": CACHE_VERSION,
        "bars_list": list(bars_list),
        "hop_bars": int(hop_bars),
        "beats_per_bar": int(beats_per_bar),
        "tracks": current_corpus_tracklist(files),
    }

    # Try fast path: load full index if present and allowed
    cache_root = Path(cache_dir) / f"hop{hop_bars}_bars{'-'.join(map(str, bars_list))}_bpb{beats_per_bar}"
    cache = index_cache_paths(cache_root)
    t0 = time.time()
    if use_cache and not rebuild_index and all(p.exists() for p in [cache["X"], cache["meta"], cache["faiss"], cache["manifest"]]):
        with open(cache["manifest"], "r") as f:
            man = json.load(f)
        reasons = []
        def _cmp(name, a, b):
            if a != b:
                reasons.append(f"{name} differs: cached={a} vs want={b}")
            return a == b

        same = (
            _cmp("model_tag",      man.get("model_tag"),      want_index_params["model_tag"]) and
            _cmp("cache_version",  man.get("cache_version"),  want_index_params["cache_version"]) and
            _cmp("bars_list",      man.get("bars_list"),      want_index_params["bars_list"]) and
            _cmp("hop_bars",       man.get("hop_bars"),       want_index_params["hop_bars"]) and
            _cmp("beats_per_bar",  man.get("beats_per_bar"),  want_index_params["beats_per_bar"]) and
            _cmp("tracks",         man.get("tracks"),         want_index_params["tracks"])
        )
        if same:
            try:
                import faiss
                index = faiss.read_index(str(cache["faiss"]))
                X = np.load(cache["X"], mmap_mode="r")
                df = pd.read_parquet(cache["meta"])
                if len(df) == X.shape[0] and index.d == X.shape[1]:
                    print("[cache] loaded full index cache")

                    # --- build per-bars subindices even on cache fast path ---
                    bars_values = sorted(set(map(int, df["bars"].tolist())))
                    indices_by_bars = {}
                    rowids_by_bars = {}
                    for b in bars_values:
                        row_ids = np.where(df["bars"].values.astype(int) == b)[0]
                        Xi = X[row_ids]
                        sub_index = build_index(Xi)  # no faiss_threads
                        indices_by_bars[int(b)] = sub_index
                        rowids_by_bars[int(b)] = row_ids.astype(np.int64)

                    return index, df, X, indices_by_bars, rowids_by_bars   # <-- return 5-tuple
                else:
                    print("[cache] manifest matched, but index/meta shapes differ; rebuilding.")
            except Exception as e:
                print(f"[cache] index cache load failed, rebuilding: {e}")
        else:
            print("[cache] manifest mismatch, rebuilding due to:")
            for r in reasons:
                print("   -", r)

    # Otherwise, (re)build from per-track caches, computing only missing/stale tracks
    all_rows = []
    all_vecs = []
    it = files
    if _HAVE_TQDM:
        it = tqdm(files, desc="Corpus", unit="track")
    
    for fp in it:
        tcache = track_cache_path(cache_root, fp)
        want_params = {
            "sr": 44100,
            "bars_list": list(bars_list),
            "hop_bars": int(hop_bars),
            "beats_per_bar": int(beats_per_bar),
            "model_tag": MODEL_TAG,
            "cache_version": CACHE_VERSION,
        }

        use_track_cache = False
        if use_cache and tcache.exists():
            try:
                emb, rows, params = load_track_npz(tcache)
                if params == want_params:
                    use_track_cache = True
                    all_vecs.append(emb)
                    all_rows.extend(rows)
                    print(f"[cache] loaded {fp.name}: {emb.shape[0]} segs")
            except Exception:
                pass

        if not use_track_cache:
            y, sr = load_mono(fp, target_sr=want_params["sr"])
            duration_s = y.shape[0] / float(sr) if y.size else 0.0
            beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)
            per_rows = []
            per_vecs = []  # will collect (batch) matrices and vstack later
            BATCH = 32

            for bars in bars_list:
                wins = slice_by_bars_from_beats(
                    beats, bars=bars, hop_bars=hop_bars, beats_per_bar=bpb, phase_offset=phase
                )
                if not wins:
                    # fallback produces (s, e) only
                    wins = _slice_by_bpm_fallback(bpm, duration_s, bars, hop_bars, bpb)

                # --- normalize wins to (s, e, b0, b1) ---
                if wins:
                    first = wins[0]
                    if len(first) == 4:
                        # already (s, e, b0, b1): keep as-is
                        pass
                    else:
                        # fallback case: infer bar indices from seconds
                        sec_per_bar = (60.0 / float(bpm)) * bpb if bpm else None
                        wins = [
                            (s, e,
                            int(round(s / sec_per_bar)) if sec_per_bar else 0,
                            int(round(e / sec_per_bar)) if sec_per_bar else 0)
                            for (s, e) in wins
                        ]
                batch_segments = []
                batch_rows = []

                def _flush_batch():
                    nonlocal per_vecs, per_rows, batch_segments, batch_rows
                    if batch_segments:
                        try:
                            V = embed_audio_batch(batch_segments, sr)  # (N, 512)
                            if V.size:
                                per_vecs.append(V.astype(np.float32))
                                per_rows.extend(batch_rows)
                        except Exception as ex:
                            print(f"[err] embed_audio_batch failed on {fp.name} [{len(batch_segments)} segs]: {ex}")
                        finally:
                            batch_segments = []
                            batch_rows = []
                    return batch_segments, batch_rows

                for (s, e, b0, b1) in wins:
                    s_i, e_i = int(round(s * sr)), int(round(e * sr))
                    seg = y[s_i:e_i]
                    if seg.shape[0] < sr:
                        continue

                    feats_extra = _extra_features(seg, sr)
                    sec_per_bar = (60.0 / bpm) * bpb if bpm else np.nan

                    batch_segments.append(seg)
                    batch_rows.append({
                        "file": fp.name, "start_s": s, "end_s": e, "bars": bars,
                        "bpm": bpm, "key": key, "scale": scale,
                        "b0": int(b0), "b1": int(b1),
                        "sec_per_bar": sec_per_bar,
                        **feats_extra
                    })

                    if len(batch_segments) >= BATCH:
                        batch_segments, batch_rows = _flush_batch()

                # flush remainder for this bars value
                batch_segments, batch_rows = _flush_batch()

            # after all bars processed
            if per_vecs:
                emb = np.vstack(per_vecs).astype(np.float32)
                save_track_npz(tcache, emb, per_rows, want_params)
                print(f"[cache] saved {fp.name}: {emb.shape[0]} segs")
                all_vecs.append(emb)
                all_rows.extend(per_rows)
    if _HAVE_TQDM:
        try:
            it.close()
        except Exception:
            pass


    if not all_vecs:
        raise RuntimeError("No valid sections extracted (after cache).")

    print(f"[ingest] vstack {len(all_vecs)} blocks -> array ...", flush=True)
    t = time.time()
    X = np.vstack(all_vecs).astype(np.float32)
    print(f"[ingest] vstack done in {time.time()-t:.2f}s | X shape={X.shape}", flush=True)

    df = pd.DataFrame(all_rows)
    print(f"[ingest] building FAISS index ...", flush=True)
    import faiss
    try:
        faiss.omp_set_num_threads(1)   # safe default; your shell exports can still lift this if you want
    except Exception:
        pass

    index = build_index(X)   # no faiss_threads arg
    print(f"[ingest] FAISS built in {time.time()-t:.2f}s", flush=True)
    print("[ingest] writing index & meta ...", flush=True)
    t = time.time()
    np.save(cache["X"], X)
    df.to_parquet(cache["meta"])
    faiss.write_index(index, str(cache["faiss"]))
    with open(cache["manifest"], "w") as f:
        json.dump({
            "ts": time.time(),
            "count": int(X.shape[0]),
            "dim": int(X.shape[1]),
            **want_index_params,    # ← single source of truth
        }, f)
    print(f"[ingest] wrote files in {time.time()-t:.2f}s  (total ingest {time.time()-t_ingest:.2f}s)", flush=True)

    bars_values = sorted(set(map(int, df["bars"].tolist())))
    indices_by_bars = {}
    rowids_by_bars = {}
    for b in bars_values:
        row_ids = np.where(df["bars"].values.astype(int) == b)[0]
        Xi = X[row_ids]
        sub_index = build_index(Xi)  # no faiss_threads
        indices_by_bars[int(b)] = sub_index
        rowids_by_bars[int(b)] = row_ids.astype(np.int64)
    return index, df, X, indices_by_bars, rowids_by_bars


def query_file(query_fp: Path, index, df_corpus: pd.DataFrame, X: np.ndarray,
               bars=16, hop_bars=8, k=5, beats_per_bar=4,
               eq_weight: float = 0.0, eq_bands: int = 24,
               corpus_dir: Path | None = None,
               indices_by_bars: dict[int, object] | None = None,
               rowids_by_bars: dict[int, np.ndarray] | None = None):
    """Slice the query track, embed windows in batch, search the corpus index, return matches DataFrame."""

    y, sr = load_mono(query_fp)
    duration_s = y.shape[0] / float(sr) if y.size else 0.0
    beats, bpm, key, scale, bpb, phase = detect_beats_bpm_key(y, sr)

    wins = slice_by_bars_from_beats(
        beats, bars=bars, hop_bars=hop_bars, beats_per_bar=bpb, phase_offset=phase
    )
    if not wins:
        # fall back using the detected beats-per-bar
        wins = _slice_by_bpm_fallback(bpm, duration_s, bars, hop_bars, bpb)

    # --- normalize wins to (s, e, b0, b1) ---
    if wins:
        first = wins[0]
        if len(first) == 4:
            # already (s, e, b0, b1) -> keep as-is
            pass
        elif len(first) == 2:
            # fallback produced (s, e) -> infer bar indices
            sec_per_bar = (60.0 / float(bpm)) * bpb if bpm else None
            wins = [
                (s, e,
                int(round(s / sec_per_bar)) if sec_per_bar else 0,
                int(round(e / sec_per_bar)) if sec_per_bar else 0)
                for (s, e) in wins
            ]
        else:
            raise ValueError(f"Unexpected window tuple length: {len(first)}")

    # collect all valid segments first
    segments = []
    q_meta = []
    for (s, e, b0, b1) in wins:
        s_i, e_i = int(round(s * sr)), int(round(e * sr))
        seg = y[s_i:e_i]
        if seg.shape[0] < sr:
            continue
        q_feats = _extra_features(seg, sr)
        segments.append(seg)
        q_meta.append({
            "q_start": float(s), "q_end": float(e),
            "q_bars": int(bars), "q_bpm": round(float(bpm), 1),
            "q_key": key, "q_scale": scale,
            "q_bar_start": int(b0), "q_bar_end": int(b1),
            **q_feats
        })

    if not segments:
        return pd.DataFrame()

    print(f"[query] windows to embed: {len(segments)}  (bars={bars}, hop={hop_bars})")

    # ---- query EQ profiles (if enabled) ----
    q_eq = None
    if eq_weight > 0.0:
        q_eq = [ _eq_profile(seg, sr, n_bands=eq_bands) for seg in segments ]


    # ---- batched embedding ----
    V = embed_audio_batch(segments, sr).astype(np.float32)  # shape: (nq, 512)

    # --- vectorized FAISS search on the SAME-BARS subindex ---
    if indices_by_bars is not None and rowids_by_bars is not None and int(bars) in indices_by_bars:
        sub_index = indices_by_bars[int(bars)]
        id_map    = rowids_by_bars[int(bars)]              # local->global row ids
    else:
        # Fallback: global (kept for safety), but this won’t be exact top-2 across same-bars
        sub_index = index
        id_map    = np.arange(len(df_corpus), dtype=np.int64)

    # we only need the true top-2 (+ small cushion for EQ failures); top-4 is fine
    k_fetch = min(max(int(k), 4), sub_index.ntotal)   # <<< was fixed to 4
    sims, ids_local = sub_index.search(V, k_fetch)

    # convert local ids (per-bars) to global df rows
    ids = np.where(ids_local >= 0, id_map[ids_local.clip(min=0)], -1)

    # ---- assemble results ----
    out_rows = []
    for qi, meta in enumerate(q_meta):
        cand = []
        ids_q   = ids[qi]
        sims_q  = sims[qi]
        valid   = ids_q >= 0
        ids_q   = ids_q[valid]
        sims_q  = sims_q[valid]

        for sim, cid in zip(sims_q, ids_q):
            row = df_corpus.iloc[int(cid)]
            # --- HARD GATE: skip if hat energy (HF percussive ratio) too different ---
            if abs(row.get("hf_perc_ratio", 0) - meta.get("hf_perc_ratio", 0)) > 0.08:
                continue
            # same-bars is guaranteed by the subindex; no need to check
            eq_sim = 0.0
            if eq_weight > 0.0 and corpus_dir is not None:
                try:
                    import soundfile as sf
                    c_path = Path(corpus_dir) / str(row["file"])
                    m_start, m_end = float(row["start_s"]), float(row["end_s"])
                    with sf.SoundFile(str(c_path)) as f:
                        native_sr = f.samplerate
                        nframes   = len(f)
                        start = max(0, min(int(round(m_start * native_sr)), nframes))
                        stop  = max(0, min(int(round(m_end   * native_sr)), nframes))
                        frames = stop - start
                        if frames > 0:
                            f.seek(start)
                            seg = f.read(frames, dtype="float32", always_2d=True)
                            seg = seg.mean(axis=1) if seg.shape[1] > 1 else seg[:, 0]
                            y_m, sr_m = seg, native_sr
                            if sr_m != sr:
                                try:
                                    import torch, torchaudio
                                    t = torch.from_numpy(y_m).unsqueeze(0)
                                    t = torchaudio.functional.resample(t, sr_m, sr).squeeze(0)
                                    y_m = t.detach().cpu().numpy().astype("float32")
                                except Exception:
                                    y_m = librosa.resample(y=y_m, orig_sr=sr_m, target_sr=sr).astype("float32")
                            if y_m.size >= sr * 0.5:
                                m_eq = _eq_profile(y_m, sr, n_bands=eq_bands)
                                eq_sim = _cos(q_eq[qi], m_eq)
                except Exception:
                    eq_sim = 0.0  # keep the candidate; just no EQ

            blended = (1.0 - eq_weight) * float(sim) + eq_weight * float(eq_sim)
            cand.append({
                **meta,
                "match_file": row["file"],
                "m_start": float(row["start_s"]),
                "m_end": float(row["end_s"]),
                "m_bars": int(row["bars"]),
                "m_bar_start": int(row.get("b0", 0)),       # ← add
                "m_bar_end":   int(row.get("b1", 0)),         # ← add
                "m_bpm": round(float(row["bpm"]), 1) if pd.notna(row["bpm"]) else None,
                "m_key": row.get("key"),
                "m_scale": row.get("scale"),
                "sim": float(sim),
                "eq_sim": float(eq_sim),
                "blend": float(blended),
            })

        # Select EXACTLY top-2 for this window (by blend if enabled, else sim)
        if cand:
            df_cand = pd.DataFrame(cand)
            sort_key = "blend" if (eq_weight > 0.0 and "blend" in df_cand.columns) else "sim"
            top2 = df_cand.sort_values(sort_key, ascending=False).head(2)
            out_rows.extend(top2.to_dict("records"))
        else:
            # extremely small corpus for this bars length (ntotal == 0) → nothing to return
            # You could synthesize a placeholder here if you truly want 2 rows regardless.
            pass

        out = pd.DataFrame(out_rows)
    
    if out.empty:
        return out
    if eq_weight > 0.0:
        return out.sort_values(["q_start","blend","sim"], ascending=[True, False, False]).reset_index(drop=True)
    else:
        return out.sort_values(["q_start","sim"], ascending=[True, False]).reset_index(drop=True)


def _cli():
    ap = argparse.ArgumentParser(description="Section-level similarity (CLAP + beat-synced bars)")
    ap.add_argument("--cache-dir", default="data/cache", type=str)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--corpus", default="data/corpus", type=str)
    # positional (optional) query path, e.g. `python -m app.pipeline data/aud.wav`
    ap.add_argument("query_pos", nargs="?", help="Path to the query audio file")
    # flag variant, e.g. `--query data/aud.wav` or `-q data/aud.wav`
    ap.add_argument("--query", "-q", dest="query_opt", type=str, help="Path to the query audio file")
    ap.add_argument("--bars", default="4", type=str,
                    help="Bars to use. Comma-separated (e.g., '2,4,8,16') or single int.")
    ap.add_argument("--hop",    default=2, type=int)
    ap.add_argument("--beats-per-bar", default=4, type=int)
    ap.add_argument("-k",       default=30, type=int)
    ap.add_argument("--eq-weight", type=float, default=0.0,
                help="Weight (0..1) to mix EQ similarity into ranking. 0 disables.")
    ap.add_argument("--eq-bands", type=int, default=64,
                    help="Number of mel bands for EQ profile (e.g., 16, 24, 32, 64).")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    query_path = args.query_opt or args.query_pos or "data/query.mp3"
    query_fp   = Path(query_path)
    if not corpus_dir.exists():
        sys.exit(f"Corpus dir not found: {corpus_dir}")
    if not query_fp.exists():
        sys.exit(f"Query file not found: {query_fp}")

    # No device/thread changes here — let your shell exports control it
    _ = _get_clap()   # warm-up model

    print("Ingesting corpus...")
    t0 = time.time()
    
    index, df, X, indices_by_bars, rowids_by_bars = ingest_corpus(
        Path(args.corpus),
        bars_list=(2, 4, 8, 16),
        hop_bars=args.hop,
        beats_per_bar=args.beats_per_bar,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        rebuild_index=args.rebuild_index,
    )
    print(f"Corpus sections indexed: {len(df)}  [{time.time()-t0:.1f}s]")

    print("Querying...")
    bars_arg = args.bars
    bars_choices = [int(b.strip()) for b in str(bars_arg).split(",") if b.strip()]
    all_out = []
    for b in bars_choices:
        out_b = query_file(
            query_fp, index, df, X,
            bars=b, hop_bars=args.hop, k=args.k, beats_per_bar=args.beats_per_bar,
            eq_weight=args.eq_weight, eq_bands=args.eq_bands, corpus_dir=corpus_dir,
            indices_by_bars=indices_by_bars, rowids_by_bars=rowids_by_bars,
        )
        if not out_b.empty:
            out_b["q_bars"] = b
            all_out.append(out_b)


    out = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    if out.empty:
        print("No results. Try a different bars/hop or add more corpus tracks.")
    else:
        # Convert seconds to mm:ss format for readability
        for col in ["q_start", "q_end", "m_start", "m_end"]:
            out[col] = out[col].apply(lambda x: _sec_to_mmss(x) if pd.notna(x) else "")

        base_cols = [
            "q_bars","q_bar_start","q_bar_end","q_start","q_end","q_bpm",
            "match_file","m_bar_start","m_bar_end","m_start","m_end","m_bpm",
            "sim","hf_perc_ratio","crest_db","flatness"
            ]

        if args.eq_weight > 0.0:
            cols_show = base_cols + ["eq_sim","blend"]
            sort_cols = ["q_start", "blend", "sim"]
            ascending = [True, False, False]
        else:
            cols_show = base_cols
            sort_cols = ["q_start", "sim"]
            ascending = [True, False]

        print(
            out[cols_show]
            .sort_values(sort_cols, ascending=ascending)
            .to_string(index=False)
        )

if __name__ == "__main__":
    _cli()

