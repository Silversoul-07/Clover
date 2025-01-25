from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil
from contextlib import asynccontextmanager

from .database import Base, engine, init_db
from .users.route import router as user_router
from .minioclient import minio_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(Base, engine)
    minio_client.create_bucket()

    yield


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