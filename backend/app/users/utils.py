

from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends
import dotenv
dotenv.load_dotenv()
import os
from PIL import Image, ImageSequence
from imagehash import phash
import asyncio
# import requests
from ..config import get_env

env = get_env()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token", auto_error=False)
SECRET_KEY = os.getenv("SECRET_KEY")
async def validate_user(token:str = Depends(oauth2_scheme)) -> str:
    if token is None:
        print("Token is missing")
        return None
        # raise HTTPException(status_code=401, detail="Token is missing")
    payload:dict = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    user_id:str =  payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Token is invalid")
    return user_id



async def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

async def encrypt(password) -> str:
    return pwd_context.hash(password)


    
async def hash_image(image: Image):
    if image.format == 'GIF':
        frames = list(ImageSequence.Iterator(image))
        if len(frames) > 1:
            first_frame_hash = await asyncio.to_thread(phash, frames[0].convert("RGB"))
            last_frame_hash = await asyncio.to_thread(phash, frames[-1].convert("RGB"))
            combined_hash = f"{first_frame_hash}{last_frame_hash}"
        else:
            frame_hash = await asyncio.to_thread(phash, frames[0].convert("RGB"))
            combined_hash = f"{frame_hash}{frame_hash}"
        return str(combined_hash)
    else:
        image_hash = await asyncio.to_thread(phash, image)
        return str(image_hash)
    
async def hash(image: Image):
    value = await asyncio.to_thread(phash, image)
    return str(value)

async def text2vec(text: str):
    payload = {"text": text}
    response = requests.post(f"{env.ml_url}/api/v2/clip", json=payload)
    if response.status_code != 200:
        return None
    return response.json()["embedding"]

async def img2vec(image: Image):
    # Preprocess image to rank 4 list of floats
    image = image.resize((384, 384))  # Adjust size as needed
    image = image.convert('RGB')
    image_array = np.array(image).astype(float)
    image_array = np.transpose(image_array, (2, 0, 1))  # Channels first
    image_array = np.expand_dims(image_array, axis=0)  # Batch size of 1
    payload = {"data": image_array.tolist()}
    print(type(payload["data"][0][0]))
    response = requests.post(f"http://localhost:8000/api/v2/image-embedder", json=payload)
    if response.status_code != 200:
        print(response.json()["detail"])
        return None
    return response.json()['result'][0]
    


async def process_embeddings(image_id:str, image: Image.Image, text: str):
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image type. Must be PIL Image.")
    if not image_id or not text:
        raise ValueError("Image ID and text are required.")
    async with manager.get_model("clip", ClipEmbedder) as model:
        image_embed = await model.img2vec(image)
        text_embed = await model.text2vec(text)
    if image_embed is None:
        raise ValueError("Failed to generate embeddings.")
    merged_embed = (image_embed + text_embed) / 2
    milvus_service.insert(
        image_id,
        image_embed.tolist(),
        merged_embed.tolist(),
    )

import numpy as np
from PIL import Image
import torchvision.transforms as T
import requests

def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                   (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).numpy()
    return np.expand_dims(image, 0).flatten().tolist()