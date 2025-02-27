import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from PIL import Image
import json
import string
from tokenizers import Encoding, Tokenizer
import pandas as pd

class CLIPTextTokenizer:
    def __init__(self, tokenizer_path: Path, tokenizer_cfg_path: Path, context_length:int = 64):
        self._PUNCTUATION_TRANS = str.maketrans("", "", string.punctuation)
        self.tokenizer:Tokenizer = Tokenizer.from_file(tokenizer_path.as_posix())
        self.context_length = context_length
        self._configure_tokenizer(tokenizer_cfg_path)

    def _configure_tokenizer(self, tokenizer_cfg_path: Path):
        tokenizer_cfg:dict = json.load(tokenizer_cfg_path.open())
        pad_token = tokenizer_cfg.get("pad_token", "<pad>")
        pad_id = self.tokenizer.token_to_id(pad_token)
        self.tokenizer.enable_padding(length=self.context_length, pad_token=pad_token, pad_id=pad_id)
        self.tokenizer.enable_truncation(max_length=self.context_length)


    def _clean_text(self, text: str, canonicalize: bool = False) -> str:
        text = " ".join(text.split())
        if canonicalize:
            text = text.translate(self._PUNCTUATION_TRANS).lower()
        return text
    
    def tokenize(self, text: str, canonicalize=True) -> dict[str, NDArray[np.int32]]:
        text = self._clean_text(text, canonicalize)
        tokens: Encoding = self.tokenizer.encode(text)
        return {"text": np.array([tokens.ids], dtype=np.int32)}


class CLIPVisualEncoder:
    def __init__(self, preprocess_cfg_path: Path):
        self.preprocess_cfg:dict = json.load(preprocess_cfg_path.open())
        self._configure_preprocessing()
        
    def _crop_pil(self, img: Image.Image, size: int) -> Image.Image:
        left = int((img.size[0] / 2) - (size / 2))
        upper = int((img.size[1] / 2) - (size / 2))
        right = left + size
        lower = upper + size

        return img.crop((left, upper, right, lower))

    def _normalize(
        self,
        img: NDArray[np.float32], mean: float | NDArray[np.float32], std: float | NDArray[np.float32]
    ) -> NDArray[np.float32]:
        return np.divide(img - mean, std, dtype=np.float32)

    def _resize_pil(self,img: Image.Image, size: int) -> Image.Image:
        if img.width < img.height:
            return img.resize((size, int((img.height / img.width) * size)), resample=Image.Resampling.BICUBIC)
        else:
            return img.resize((int((img.width / img.height) * size), size), resample=Image.Resampling.BICUBIC)

    def _to_numpy(self,img: Image.Image) -> NDArray[np.float32]:
        return np.asarray(img if img.mode == "RGB" else img.convert("RGB"), dtype=np.float32) / 255.0

    def _configure_preprocessing(self):
        size = self.preprocess_cfg.get("size", 224)
        self.size = size[0] if isinstance(size, list) else size
        self.mean = np.array(self.preprocess_cfg.get("mean", [0.481, 0.457, 0.408]), dtype=np.float32)
        self.std = np.array(self.preprocess_cfg.get("std", [0.268, 0.261, 0.275]), dtype=np.float32)

    def preprocess(self, image: Image.Image) -> dict[str, NDArray[np.float32]]:
        image = self._resize_pil(image, self.size)
        image = self._crop_pil(image, self.size)
        image_np = self._to_numpy(image)
        image_np = self._normalize(image_np, self.mean, self.std)
        return {"image": np.expand_dims(image_np.transpose(2, 0, 1), 0)}

class TaggerUtils:
    @staticmethod
    def preprocess(image: Image.Image, height=224, width=224):
        image = image.convert("RGB")
        image = image.resize((height, width), Image.BICUBIC)
        image_array = np.asarray(image, dtype=np.float32)[:, :, ::-1]  # Convert RGB to BGR
        return np.expand_dims(image_array, axis=0)
    
    @staticmethod
    def get_tags(csv_path: str = "models/tagger/selected_tags.csv") -> list[str]:
        tags_df = pd.read_csv(csv_path)
        return tags_df["name"].tolist()
