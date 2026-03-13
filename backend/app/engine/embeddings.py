
# app/embeddings.py
from __future__ import annotations
import os
import numpy as np

# Prefer stability on Apple: allow CPU fallback for unsupported MPS ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from laion_clap import CLAP_Module
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*MPS backend.*",
    category=UserWarning,
)


try:
    # Keep it modest; bump if CPU isn’t saturated.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
except Exception:
    pass

# --------- Optional resamplers (no librosa here) ----------
_HAVE_TORCHAUDIO = False
try:
    import torchaudio
    _HAVE_TORCHAUDIO = True
except Exception:
    pass

_HAVE_SCIPY = False
try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    pass

# --------- Global model handle ----------
_CLAP = None

def _select_device() -> str:
    """
    Gold-standard device policy:
      1) Honor CLAP_DEVICE env if set to one of {cuda, mps, cpu}
      2) Otherwise prefer CUDA, then MPS, else CPU
    """
    dev_env = os.environ.get("CLAP_DEVICE", "").lower()
    if dev_env in {"cuda", "mps", "cpu"}:
        return dev_env
    if torch.cuda.is_available():
        # enable higher precision matmul on CUDA if supported
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_clap() -> CLAP_Module:
    """
    Returns a singleton CLAP model, loaded with the specified checkpoint.
    Accuracy-first: HTSAT-base with official music_audioset_epoch_15_esc_90.14.pt.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*MPS backend.*will fall back to run on the CPU.*",
        category=UserWarning,
    )

    global _CLAP
    if _CLAP is None:
        # Device selection: env override > CUDA > CPU (skip MPS by default on macOS)
        dev = _select_device() 
        ckpt_path = os.environ.get(
            "CLAP_CKPT",
            "/Users/lbonazelli/models/clap/music_audioset_epoch_15_esc_90.14.pt"
        )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CLAP checkpoint not found at: {ckpt_path}")
        amodel = os.environ.get("CLAP_AMODEL", "HTSAT-base")

        m = CLAP_Module(enable_fusion=False, amodel=amodel)
        m.load_ckpt(ckpt=ckpt_path, verbose=True)
        m = m.to(dev)
        m.eval()
        print(f"[embeddings] CLAP device: {dev} | amodel={amodel} | ckpt={ckpt_path}")
        _CLAP = m
    return _CLAP



# ----------------- Resampling (to 48k) -----------------
def _resample_to_48k(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Resample mono float32 to 48 kHz using:
      - torchaudio (best general choice; GPU-accel on CUDA),
      - else exact polyphase via scipy.signal.resample_poly,
      - else a last-resort ZOH (accuracy not ideal, but avoids crashes).
    """
    if sr == 48000:
        return y.astype(np.float32, copy=False)

    y = y.astype(np.float32, copy=False)

    if _HAVE_TORCHAUDIO:
        # torchaudio handles high-quality resampling; uses CUDA if available
        wav = torch.from_numpy(y)[None, :]  # (1, T)
        wav48 = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
        return wav48.squeeze(0).contiguous().cpu().numpy().astype(np.float32)

    if _HAVE_SCIPY:
        from math import gcd
        g = gcd(sr, 48000)
        up, down = 48000 // g, sr // g
        y48 = resample_poly(y, up, down)
        return y48.astype(np.float32)

    # FINAL fallback
    ratio = 48000 / float(sr)
    idx = (np.arange(int(len(y) * ratio)) / ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx].astype(np.float32)


# ----------------- Public embedding APIs -----------------
def embed_audio_batch(segments: list[np.ndarray], sr: int, batch_size: int = 64) -> np.ndarray:
    """
    Embed many mono segments. Returns (N, 512) float32 L2-normalized.
    - Resamples all to 48k.
    - Batches to prevent OOM on long lists.
    """
    if not segments:
        return np.zeros((0, 512), dtype=np.float32)

    segs = []
    for s in segments:
        s = np.asarray(s)
        if s.ndim > 1:
            s = s.mean(axis=0)
        if s.dtype != np.float32:
            s = s.astype(np.float32, copy=False)
        segs.append(s)

    m = _get_clap()
    device = next(m.parameters()).device

    # Pre-resample in Python (predictable memory), then feed tensors on-device.
    seg48 = [_resample_to_48k(s, sr) for s in segs]

    out = []
    with torch.inference_mode():
        for i in range(0, len(seg48), batch_size):
            chunk = seg48[i:i + batch_size]
            tensors = [torch.from_numpy(w).contiguous() for w in chunk]
            if str(device) != "cpu":
                tensors = [t.to(device, non_blocking=True) for t in tensors]
            embs = m.get_audio_embedding_from_data(x=tensors, use_tensor=True)
            M = torch.stack([e.detach().float() for e in embs], dim=0)  # (B, 512)
            # L2-normalize in torch (stable + fast), then move to CPU
            M = torch.nn.functional.normalize(M, p=2, dim=1)
            out.append(M.cpu().numpy())

    V = np.vstack(out).astype(np.float32)
    return V


def build_hybrid_embedding(
    clap_vec: np.ndarray,
    hf_perc_ratio: float = 0.0,
    rms_dbfs: float = -20.0,
    peak_dbfs: float = -6.0,
    crest_db: float = 10.0,
    flatness: float = 0.1,
    band_energies: dict | None = None,
) -> np.ndarray:
    """
    Build a Diverse Audio Embedding (DAE) by concatenating the L2-normalized
    CLAP vector with scaled engineered features. Returns (512 + N) float32.
    """
    eng_feats = [
        hf_perc_ratio * 10.0,
        (rms_dbfs + 40.0) / 40.0,
        (peak_dbfs + 40.0) / 40.0,
        crest_db / 20.0,
        flatness * 10.0,
    ]
    if band_energies:
        from app.engine.spectral import BAND_NAMES
        for name in BAND_NAMES:
            eng_feats.append((band_energies.get(name, -96.0) + 96.0) / 96.0)

    eng = np.array(eng_feats, dtype=np.float32)
    hybrid = np.concatenate([clap_vec.flatten(), eng])
    hybrid = hybrid / (np.linalg.norm(hybrid) + 1e-12)
    return hybrid.astype(np.float32)


def embed_text(queries: list[str]) -> np.ndarray:
    """
    Embed text queries using CLAP's text encoder.
    Returns (N, 512) float32 L2-normalized vectors in the same space as audio embeddings.
    """
    if not queries:
        return np.zeros((0, 512), dtype=np.float32)

    m = _get_clap()

    with torch.inference_mode():
        embs = m.get_text_embedding(queries, use_tensor=True)
        M = embs.detach().float()
        M = torch.nn.functional.normalize(M, p=2, dim=1)

    return M.cpu().numpy().astype(np.float32)