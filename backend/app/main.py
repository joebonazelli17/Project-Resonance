import app.core.patches  # noqa: F401 -- must be first to patch HF before any ML imports

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.core.storage import ensure_bucket
from app.api.routes import health, tracks, search


def _log_hardware():
    import torch
    import platform
    gpu = "CUDA" if torch.cuda.is_available() else "MPS" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "none"
    print(f"[resonance] platform={platform.machine()} torch={torch.__version__} gpu={gpu} threads={torch.get_num_threads()}")
    if gpu == "CUDA":
        print(f"[resonance] GPU: {torch.cuda.get_device_name(0)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_hardware()
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