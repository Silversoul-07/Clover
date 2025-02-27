import asyncio
import httpx
from typing import Tuple, List

Tag = Tuple[str, float]

class APIClient:
    BASE_URL = "http://localhost:9876/api/v2"

    @staticmethod
    async def text2vec(text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{APIClient.BASE_URL}/text2vec", json={"text": text})
            response.raise_for_status()
            return response.json()["embedding"]

    @staticmethod
    async def img2vec(file: bytes) -> List[float]:
        async with httpx.AsyncClient() as client:
            files = {"file": file}
            response = await client.post(f"{APIClient.BASE_URL}/img2vec", files=files)
            response.raise_for_status()
            return response.json()["embedding"]

    @staticmethod
    async def predict_tags(file: bytes) -> List[Tag]:
        async with httpx.AsyncClient() as client:
            files = {"file": file}
            response = await client.post(f"{APIClient.BASE_URL}/tags", files=files)
            response.raise_for_status()
            return response.json()
            