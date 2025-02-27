

from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import Depends
import dotenv

dotenv.load_dotenv()
from PIL import Image, ImageSequence
from imagehash import phash
import asyncio
# import requests
from ..config import get_env
import io
import httpx


env = get_env()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token", auto_error=False)
async def validate_user(token:str = Depends(oauth2_scheme)) -> str:
    if token is None:
        print("Token is missing")
        return None
        # raise HTTPException(status_code=401, detail="Token is missing")
    payload:dict = jwt.decode(token, env.jwt_secret_key, algorithms=["HS256"])
    user_id:str =  payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Token is invalid")
    return user_id

async def process_avatar(avatar, size=(128, 128)):
    image = Image.open(io.BytesIO(await avatar.read()))
    image.thumbnail(size)
    output = io.BytesIO()
    image.save(output, format=image.format)
    output.seek(0)
    return output.getvalue()

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, 
                             env.jwt_secret_key,
                             algorithm="HS256")
    return encoded_jwt

    
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

async def url_to_binary(url: str):
    # use httpx to get the image
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.content


