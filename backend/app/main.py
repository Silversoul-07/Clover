from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil
from contextlib import asynccontextmanager

from .database import Base, engine
from .users.route import router as user_router
from .inference import manager
from .minio import minio_client
from .milvus import milvus_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    minio_client.create_bucket()
    milvus_service.create()

    yield

    await manager.unload()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(user_router)

@app.get("/", tags=["redirect"])
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["health"])
def get_health():
    return {"status": "ok"}

@app.get("/stats", tags=["stats"])
def memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info() 
    return {
        "rss": memory_info.rss / (1024 * 1024), 
        "vms": memory_info.vms / (1024 * 1024),
        "percent": psutil.virtual_memory().percent,
    }