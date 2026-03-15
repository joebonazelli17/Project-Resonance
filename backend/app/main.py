import os

# Redirect HuggingFace model loading to pre-downloaded local files.
# CLAP hardcodes from_pretrained("bert-base-uncased") and from_pretrained("roberta-base")
# at module import time. We intercept those calls to use bundled models in /models/.
_LOCAL_MODEL_PATHS = {
    "bert-base-uncased": "/models/bert-base-uncased",
    "roberta-base": "/models/roberta-base",
}

import transformers

_orig_bert_from_pretrained = transformers.BertTokenizer.from_pretrained.__func__
def _patched_bert_from_pretrained(cls, name, *args, **kwargs):
    for key, path in _LOCAL_MODEL_PATHS.items():
        if isinstance(name, str) and key in name:
            return _orig_bert_from_pretrained(cls, path, *args, **kwargs)
    return _orig_bert_from_pretrained(cls, name, *args, **kwargs)
transformers.BertTokenizer.from_pretrained = classmethod(_patched_bert_from_pretrained)

_orig_model_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__
def _patched_model_from_pretrained(cls, name, *args, **kwargs):
    for key, path in _LOCAL_MODEL_PATHS.items():
        if isinstance(name, str) and key in name:
            return _orig_model_from_pretrained(cls, path, *args, **kwargs)
    return _orig_model_from_pretrained(cls, name, *args, **kwargs)
transformers.PreTrainedModel.from_pretrained = classmethod(_patched_model_from_pretrained)

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