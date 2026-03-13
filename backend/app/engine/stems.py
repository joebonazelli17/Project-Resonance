from __future__ import annotations

import os
import numpy as np
import torch

STEM_NAMES = ("drums", "bass", "vocals", "other")
_MODEL = None


def _get_demucs():
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model
        model_name = os.environ.get("DEMUCS_MODEL", "htdemucs")
        _MODEL = get_model(model_name)
        device = os.environ.get("DEMUCS_DEVICE", "cpu")
        _MODEL = _MODEL.to(device)
        _MODEL.eval()
        print(f"[stems] Demucs model={model_name} device={device}")
    return _MODEL


def separate_stems(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Separate a mono/stereo track into stems using Demucs.
    Returns dict mapping stem name -> mono float32 array at original sr.
    """
    from demucs.apply import apply_model

    model = _get_demucs()
    device = next(model.parameters()).device

    if y.ndim == 1:
        wav = np.stack([y, y])
    else:
        wav = y

    tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)

    if model.samplerate != sr:
        import torchaudio
        tensor = torchaudio.functional.resample(tensor, sr, model.samplerate)

    with torch.inference_mode():
        sources = apply_model(model, tensor, device=str(device))

    if model.samplerate != sr:
        import torchaudio
        sources = torchaudio.functional.resample(sources, model.samplerate, sr)

    sources = sources.squeeze(0).cpu().numpy()

    stems = {}
    for i, name in enumerate(model.sources):
        stem = sources[i]
        mono = stem.mean(axis=0) if stem.ndim > 1 else stem
        stems[name] = mono.astype(np.float32)

    return stems