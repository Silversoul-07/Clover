from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder
import yaml
import os
import shutil
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from typing import Dict
from .utils import CLIPTextTokenizer, CLIPVisualEncoder, TaggerUtils
from pathlib import Path
from fastapi.responses import JSONResponse

# Global dictionary to store model sessions
model_sessions: Dict[str, ort.InferenceSession] = {}
processor = None
encoder = None

# Load and initialize models
def load_models():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    for model_name, model_info in config.get('models', {}).items():
        repo_id = model_info['repo_id']
        files = model_info['files']
        
        try:
            target_dir = f"models/{model_name}"
            os.makedirs(target_dir, exist_ok=True)
            
            for filename in files:
                target_path = os.path.join(target_dir, filename.split("/")[-1])
                if not os.path.exists(target_path):
                    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir_use_symlinks=False)
                    shutil.copy(file_path, target_path)
                    if os.path.islink(file_path):
                        os.unlink(file_path)
                    else:
                        os.remove(file_path)
                
                # Load the ONNX model if the file is a model
                if target_path.endswith(".onnx"):
                    session = ort.InferenceSession(target_path)
                    model_sessions[model_name] = session
                    print(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise e
        
def load_essential():
    global processor, encoder, taggerUtils  
    encoder = CLIPVisualEncoder(Path("models/image-embedder/preprocess_cfg.json"))

    processor = CLIPTextTokenizer(
        tokenizer_path=Path("models/text-embedder/tokenizer.json"),
        tokenizer_cfg_path=Path("models/text-embedder/tokenizer_config.json"),
    )

    taggerUtils = TaggerUtils()
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models
    print("Loading models...")
    load_models()
    load_essential()
    yield
    # Shutdown: Clean up resources
    print("Cleaning up...")
    model_sessions.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

@app.post("/api/v2/text2vec")
async def text2vec(req: TextRequest):
    try:
        text = req.text
        text_embbeder = model_sessions.get("text-embedder")
        if text_embbeder is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        text_tokens = processor.tokenize(text)
        embedding = text_embbeder.run(None, text_tokens)
        return {"embedding": embedding[0].tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/img2vec")
async def img2vec(file: UploadFile = File(...)):
    try:
        image_embedder = model_sessions.get("image-embedder")
        if image_embedder is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        img = Image.open(io.BytesIO(await file.read()))
        img = encoder.preprocess(img)
        embedding = image_embedder.run(None, img)
        return {"embedding": embedding[0].tolist()[0]}
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v2/tags") 
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        tagger = model_sessions.get("tagger")

        if tagger is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        input_name = tagger.get_inputs()[0].name
        label_name = tagger.get_outputs()[0].name
        _, height, width, _ = tagger.get_inputs()[0].shape

        image = TaggerUtils.preprocess(image, height, width)
        preds = tagger.run([label_name], {input_name: image})[0][0]
        tag_names = TaggerUtils.get_tags()
        
        tags = {tag_names[i]: float(preds[i]) for i in range(len(tag_names)) if preds[i] > 0.35}
        return JSONResponse(jsonable_encoder(sorted(tags.items(), key=lambda x: x[1], reverse=True)))

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
 