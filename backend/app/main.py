import os

# Block ALL HuggingFace network access -- Zscaler intercepts these requests.
# All models must be pre-downloaded in /models/ at Docker build time.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Redirect HuggingFace model loading to pre-downloaded local files.
_LOCAL_MODEL_PATHS = {
    "bert-base-uncased": "/models/bert-base-uncased",
    "roberta-base": "/models/roberta-base",
    "facebook/bart-base": "/models/bart-base",
}

import transformers

# Patch all tokenizer/model from_pretrained calls to use local paths
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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.core.storage import ensure_bucket
from app.api.routes import health, tracks, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    ensure_bucket()
    yield


app = FastAPI(
    title="Resonance",
    description="AI Music Intelligence Platform -- Reference Track Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(tracks.router, prefix="/api")
app.include_router(search.router, prefix="/api")