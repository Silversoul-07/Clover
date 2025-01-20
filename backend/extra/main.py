import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import asyncio
import torch
import torch.nn as nn
from typing import Any, Optional
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
from PIL import Image
import logging
import time
import pytorch_lightning as pl
import torchvision.transforms.functional as TVF
import onnxruntime as ort


import asyncio
import time
import logging
import gc
import torch
from contextlib import asynccontextmanager

class ModelManager:
    def __init__(self, idle_timeout: int = 300):
        self._models = {}
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()
        self._last_access = {}
        self._cleanup_task = None
        self._is_running = True

    async def start(self):
        """Start the cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_models())

    async def stop(self):
        """Stop the cleanup task and cleanup all models."""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.clean()

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

    async def unload(self, model_name: str):
        """Unload a model with proper cleanup of CUDA memory."""
        async with self._lock:
            if model_name in self._models:
                try:
                    model = self._models[model_name]
                    
                    # Proper cleanup for HuggingFace models
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    
                    # Clear CUDA cache for the model
                    if torch.cuda.is_available():
                        with torch.cuda.device('cuda'):
                            torch.cuda.empty_cache()
                    
                    # Delete model and clear from memory
                    del self._models[model_name]
                    del self._last_access[model_name]
                    
                    # Force garbage collection
                    gc.collect()
                    
                    logging.info(f"Successfully unloaded model {model_name}")
                except Exception as e:
                    logging.error(f"Error unloading model {model_name}: {str(e)}")
                    raise

    async def clean(self):
        """Clean up all models."""
        async with self._lock:
            model_names = list(self._models.keys())
            for model_name in model_names:
                await self.unload(model_name)

    async def _cleanup_idle_models(self):
        """Periodic cleanup of idle models."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Check every minute instead of waiting for full timeout
                current_time = time.time()
                
                async with self._lock:
                    model_names = list(self._models.keys())
                    for model_name in model_names:
                        if current_time - self._last_access.get(model_name, 0) > self._idle_timeout:
                            logging.info(f"Auto-unloading idle model {model_name}")
                            await self.unload(model_name)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in cleanup task: {str(e)}")

        

class MLP(pl.LightningModule):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

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

    async def text2vec(self, text: str) -> Optional[torch.Tensor]:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            return text_features / text_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logging.error(f"Error in text2vec: {e}")
            return None

    async def img2vec(self, image: Image.Image) -> Optional[torch.Tensor]:
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logging.error(f"Error in img2vec: {e}")
            return None


    
class AestheticScorer:
    def __init__(self):
        self.device = 'cpu'
        self.model_path = '/home/praveen/Desktop/newProject/backend/extra/state.pth'    
        self.model = MLP()
        try:
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logging.error(f"Error loading MLP model: {e}")
            raise

    @classmethod
    async def loader(cls):
        try:
            instance = cls()
            return instance
        except Exception as e:
            logging.error(f"Error in AestheticScorer.loader: {e}")
            raise

    async def predict(self, img, model_manager: ModelManager):
        try:
            # Load ClipEmbedder via ModelManager
            clip_embedder: ClipEmbedder = await model_manager.load("clip", ClipEmbedder)

            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            image_embedding = await clip_embedder.img2vec(img)
            if image_embedding is None:
                logging.error("Failed to generate image embedding.")
                return None

            # Ensure image_embedding is on the correct device and has the right datatype
            if isinstance(image_embedding, torch.Tensor):
                embedding = image_embedding.to(self.device)
            else:
                embedding = torch.tensor(image_embedding).to(self.device)

            with torch.no_grad():
                prediction = self.model(embedding)

            if self.device == "cuda":
                prediction = prediction.cpu()

            return prediction.item()
        except Exception as e:
            logging.error(f"Error in AestheticScorer.predict: {e}")
            return None
        

# onnx only
class ImageTagger:
    def __init__(self, model, top_tags: list[str]):
        self.image_size = 448
        self.model = model
        self.top_tags = top_tags
        self.threshold = 0.4

    @classmethod
    async def loader(cls) -> "ImageTagger":
        try:
            MODEL_PATH = "/home/praveen/joytag/files"            
            model = cls._load_model(MODEL_PATH)
            tags_file = f"{MODEL_PATH}/top_tags.txt"
            with open(tags_file, "r") as f:
                top_tags = [line.strip() for line in f.readlines() if line.strip()]
            return cls(model, top_tags)

        except Exception as e:
            logging.error(f"Error loading model or tags: {e}")
            raise

    @staticmethod
    def _load_model(path: str):
        onnx_path = f"{path}/model.onnx"
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        return {'session': session, 'input_name': input_name, 'image_size': 448}

    @staticmethod
    def _preprocess(image: Image.Image, target_size: int) -> torch.Tensor:
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize image
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to tensor
        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

        # Normalize
        image_tensor = TVF.normalize(
            image_tensor,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return image_tensor

    @torch.no_grad()
    async def predict(self, image: Image.Image, limit:int=20) -> tuple[str, dict, float]:
        try:
            image_tensor = self._preprocess(image, self.image_size)
            batch = image_tensor.unsqueeze(0)
            inputs = {self.model['input_name']: batch.numpy()}
            preds = self.model['session'].run(None, inputs)
            tag_preds = torch.from_numpy(preds[0]).sigmoid()
            scores = {self.top_tags[i]: tag_preds[0][i].item() for i in range(min(limit, len(self.top_tags)))}
            return scores

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {}

async def main():
    logging.basicConfig(level=logging.ERROR)
    manager = ModelManager()
    
    try:
        path = '/home/praveen/Desktop/2.jpeg'
        aesthetic_scorer: AestheticScorer = await manager.load('aesthetic', AestheticScorer)
        prediction = await aesthetic_scorer.predict(path, manager)
        print(f"Aesthetic Score: {prediction}")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

    finally:
        await manager.unload('aesthetic')
        await manager.unload('clip')


if __name__ == "__main__":
    asyncio.run(main())