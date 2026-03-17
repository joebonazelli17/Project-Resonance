"""
Early-init patches: block HuggingFace network access and redirect model loading
to pre-downloaded local files. Import this module BEFORE any ML imports.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_LOCAL_MODEL_PATHS = {
    "bert-base-uncased": "/models/bert-base-uncased",
    "roberta-base": "/models/roberta-base",
    "facebook/bart-base": "/models/bart-base",
}

import transformers


def _make_patched_from_pretrained(orig_fn):
    def _patched(cls, name, *args, **kwargs):
        for key, path in _LOCAL_MODEL_PATHS.items():
            if isinstance(name, str) and key in name:
                kwargs["local_files_only"] = True
                return orig_fn(cls, path, *args, **kwargs)
        return orig_fn(cls, name, *args, **kwargs)
    return _patched


for _cls in [
    transformers.BertTokenizer,
    transformers.RobertaTokenizer,
    transformers.RobertaTokenizerFast,
    transformers.BartTokenizer,
    transformers.PreTrainedModel,
    transformers.AutoTokenizer,
    transformers.AutoModel,
]:
    try:
        _orig = _cls.from_pretrained.__func__
        _cls.from_pretrained = classmethod(_make_patched_from_pretrained(_orig))
    except (AttributeError, TypeError):
        pass
