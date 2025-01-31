from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List
import numpy as np
import yaml
import os
import shutil
import asyncio
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from .utils import ModelWrapper

class InputData(BaseModel):
    data: Any

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    load_models()
    
    # Background task for model cleanup
    cleanup_task = asyncio.create_task(model_cleanup())
    
    yield
    
    # Cleanup tasks on shutdown
    cleanup_task.cancel()

def load_models():
    global models
    models = {}
    
    # Load from config
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    for model in config.get('models', []):
        if model['priority'] == None:
            continue

        repo_id = model['repo_id']
        filename = model['filename']
        name = model['name']
        
        try:
            target_dir = f"models/{name}"
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, "model.onnx")
            if not os.path.exists(target_path):
                file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir_use_symlinks=False)
                shutil.copy(file_path, target_path)
                if os.path.islink(file_path):
                    os.unlink(file_path)
                else:
                    os.remove(file_path)

            models[name] = ModelWrapper(target_path)
            print(models[name].get_input_output_names())
            print(f"Model {name} prepared for loading")
        except Exception as e:
            print(f"Could not prepare model {name}: {e}")

async def model_cleanup():
    while True:
        await asyncio.sleep(60)
        for model in models.values():
            model.unload()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware for better API flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# list all available models
@app.get("/api/v2/models")
async def get_models():
    return list(models.keys())

@app.post("/api/v2/{model_name}")
async def infer(model_name: str, input_data: InputData):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # data = np.array(input_data.data, dtype=np.float32)
        result = models[model_name].infer(input_data.data)
        return {"result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/{model_name}")
async def health(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[model_name]
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "memory_available": model.can_load_model()
    }