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

from pymilvus import connections, Collection, DataType, CollectionSchema, FieldSchema
import logging
import numpy as np
from typing import Optional, Dict
from deepface import DeepFace

class FaceRecognizer:
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        model_name: str = "Facenet512",
        detector_backend: str = "retinaface",
        similarity_threshold: float = 0.9
    ):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embedding_dim = 512
        self.similarity_threshold = similarity_threshold
        self._initialize_connection(uri)

    def _initialize_connection(self, uri: str) -> None:
        connections.connect("default", uri=uri)
        self.collection = Collection("FaceRecognition")
        self.collection.load()

    def _get_embedding(self, image: str) -> list[float]:
        result = DeepFace.represent(
            img_path=image,
            model_name=self.model_name,
            detector_backend=self.detector_backend
        )
        embedding = np.array(result[0]["embedding"]).astype("float32")
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def add_face(self, image: str, identity: str) -> bool:
        try:
            embedding = self._get_embedding(image)
            data = {
                "id": [identity],
                "face_embed": [embedding]
            }
            self.collection.insert(data)
            self.collection.flush()
            return True
        except Exception as e:
            logging.error(f"Face addition failed - {identity}: {str(e)}")
            return False

    def predict(self, image: str) -> Optional[str]:
        try:
            embedding = self._get_embedding(image)
            
            results = self.collection.search(
                data=[embedding],
                anns_field="face_embed",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=1
            )

            if not results[0]:
                return None

            match = results[0][0]
            if match.score < self.similarity_threshold:
                return None

            return match.id

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return None

    def __del__(self):
        try:
            connections.disconnect("default")
        except:
            pass

import argparse
import os

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

# Model and label file details
MODEL = "model.onnx"
CSV = "selected_tags.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()

def load_labels(dataframe) -> tuple[list[str], list[int], list[int], list[int]]:
    name_series = dataframe["name"].map(lambda x: x.replace("_", " "))
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])

    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs: np.ndarray, args: argparse.Namespace) -> float:
    if len(probs) < 2:
        return args.score_general_threshold
    
    sorted_probs = np.sort(probs)[::-1]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2

    # Optionally adjust threshold based on arguments
    if thresh < args.score_general_threshold:
        thresh = args.score_general_threshold
    elif thresh > args.score_character_threshold:
        thresh = args.score_character_threshold

    return thresh

class Predictor:
    def __init__(self):
        self.model_target_size = None

    def load_model(self):
        tags_df = pd.read_csv(CSV)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(MODEL)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.width) // 2
        pad_top = (max_dim - image.height) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # Convert RGB to BGR

        return np.expand_dims(image_array, axis=0)

    def predict(self, image, general_thresh, character_thresh):
        self.load_model()

        image = self.prepare_image(image)
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

    
        # General tags
        general_tags = {
            self.tag_names[i]: preds[0][i]
            for i in self.general_indexes
            if preds[0][i] > general_thresh
        }

        # Character tags
        character_tags = {
            self.tag_names[i]: preds[0][i]
            for i in self.character_indexes
            if preds[0][i] > character_thresh
        }

        return general_tags, character_tags

def main():
    args = parse_args()
    predictor = Predictor()

    image_path = input("Enter the path to the image: ")
    image = Image.open(image_path).convert("RGBA")

    general_tags, character_tags = predictor.predict(
        image, args.score_general_threshold, args.score_character_threshold
    )

    print("General Tags:", general_tags)
    print("Character Tags:", character_tags)

if __name__ == "__main__":
    main()

import torch
from transformers import CLIPModel, CLIPProcessor
from backend.app.uselater import MLP

class CombinedModel(torch.nn.Module):
    def __init__(self, clip_model, scorer):
        super(CombinedModel, self).__init__()
        self.clip = clip_model
        self.scorer = scorer
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def forward(self, image):
        # Handle both preprocessed tensors and raw images
        if not isinstance(image, torch.Tensor):
            # Preprocess image using CLIP processor
            inputs = self.processor(images=image, return_tensors="pt")
            image = inputs.pixel_values
            
        with torch.no_grad():
            embeddings = self.clip.get_image_features(image)
        
        # Normalize embeddings as CLIP does
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        score = self.scorer(embeddings)
        return score

def export_model(save_path="aesthetic_scorer.onnx"):
    # Initialize models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    scorer = MLP()  # You would load your trained weights here
    scorer.load_state_dict(
        torch.load("/home/praveen/Desktop/newProject/backend/extra/state.pth", map_location="cpu", weights_only=True)
    )
    # Combine models
    combined = CombinedModel(clip_model, scorer)
    combined.eval()  # Set to evaluation mode
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # Example image tensor
    
    # Export to ONNX
    torch.onnx.export(
        combined,
        dummy_input,
        save_path,
        opset_version=14,
        input_names=['input'],
        output_names=['score'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'score': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {save_path}")

if __name__ == "__main__":
    export_model()
