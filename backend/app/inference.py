from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import asyncio
from contextlib import asynccontextmanager
import time
import logging
import gc
from PIL import Image
from typing import Optional
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
import psutil


class ModelManager:
    def __init__(self, idle_timeout: int = 300):
        self._models = {}
        self._lock = asyncio.Lock()
        self._last_access = {}

    @asynccontextmanager
    async def get_model(self, model_name: str, model_class):
        """Context manager for safely accessing models with automatic cleanup."""
        try:
            model = await self.load(model_name, model_class)
            yield model
        finally:
            self._last_access[model_name] = time.time()

    async def load(self, model_name: str, model_class):
        """Load a model with proper error handling and logging."""
        async with self._lock:
            if model_name not in self._models:
                try:
                    logging.info(f"Loading model {model_name}...")
                    self._models[model_name] = await model_class.loader()
                    self._last_access[model_name] = time.time()
                    logging.info(f"Successfully loaded model {model_name}")
                except Exception as e:
                    logging.error(f"Failed to load model {model_name}: {str(e)}")
                    raise
            else:
                self._last_access[model_name] = time.time()
            
            return self._models[model_name]

    async def unload(self, force: bool = True):
        async with self._lock:
            model_names = list(self._models.keys())
            for model_name in model_names:
               del self._models[model_name]
            
            # Additional cleanup
            gc.collect()
            # torch.cuda.empty_cache()

class ClipEmbedder:
    def __init__(self, model: CLIPModel, processor: CLIPProcessor, tokenizer: CLIPTokenizerFast):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    @classmethod
    async def loader(cls):
        try:
            model = await asyncio.to_thread(CLIPModel.from_pretrained, "openai/clip-vit-large-patch14")
            processor = await asyncio.to_thread(CLIPProcessor.from_pretrained, "openai/clip-vit-large-patch14")
            tokenizer = await asyncio.to_thread(CLIPTokenizerFast.from_pretrained, "openai/clip-vit-large-patch14")
            return cls(model, processor, tokenizer)
        except Exception as e:
            logging.error(f"Error loading CLIP model: {e}")
            raise

    async def text2vec(self, text: str):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            text_features = self.model.get_text_features(**inputs)
            return text_features / text_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logging.error(f"Error in text2vec: {e}")
            return None

    async def img2vec(self, image: Image.Image):
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Invalid image type. Must be PIL Image.")
            if image.mode != "RGB":
                image = image.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logging.error(f"Error in img2vec: {e}")
            return None


# Example usage
async def main():
    # Initialize manager
    manager = ModelManager(idle_timeout=300)
    await manager.start()
    
    try:
        # Use context manager for safe model access
        async with manager.get_model("bert-base", BertModel) as model:
            # Use your model here
            result = await model.predict("some text")
            
    finally:
        # Cleanup when done
        await manager.stop()

manager = ModelManager(idle_timeout=300)