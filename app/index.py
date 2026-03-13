# app/index.py
import os, time
import numpy as np
import faiss

def build_index(X: np.ndarray, *, normalize: bool = False) -> faiss.IndexFlatIP:
    # macOS thread clamp (harmless elsewhere)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        bad_rows = np.where(~np.isfinite(X).all(axis=1))[0][:5]
        raise ValueError(f"NaN/Inf in embeddings at rows {bad_rows.tolist()} (first 5)")

    if normalize:
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(X.shape[1])  # exact cosine because vectors are L2-normalized
    index.add(X)
    return index