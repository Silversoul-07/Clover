import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as T

def load_and_preprocess_image(image_path):
    # CLIP's exact preprocessing pipeline
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                   (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).numpy()
    return np.expand_dims(image, 0)

# Load model and run inference
session = onnxruntime.InferenceSession("model.onnx")
image = load_and_preprocess_image("/home/praveen/Desktop/2.jpeg")
score = session.run(None, {"input": image})[0]
print(score)