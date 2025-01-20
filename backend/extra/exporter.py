import torch
from transformers import CLIPModel, CLIPProcessor
from main import MLP


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