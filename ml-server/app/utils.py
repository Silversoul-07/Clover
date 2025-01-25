import onnxruntime as ort
import numpy as np
from datetime import datetime
import psutil

class ModelWrapper:
    def __init__(self, model_path: str, memory_threshold: float = 0.8):
        self.model_path = model_path
        self.model = None
        self.last_used = None
        self.memory_threshold = memory_threshold

    def can_load_model(self) -> bool:
        """Check if there's enough memory to load the model."""
        memory_usage = psutil.virtual_memory()
        available_memory_percent = memory_usage.available / memory_usage.total
        return available_memory_percent < self.memory_threshold

    def load(self):
        """Load the model if enough memory is available."""
        if self.model is None and self.can_load_model():
            try:
                self.model = ort.InferenceSession(self.model_path)
                self.last_used = datetime.now()
            except Exception as e:
                print(f"Failed to load model: {e}")

    def unload(self, max_idle_time: int = 300):
        """Unload model if it has been idle for too long."""
        if (self.model and self.last_used and 
            (datetime.now() - self.last_used).seconds > max_idle_time):
            self.model = None

    def infer(self, data: np.ndarray) -> np.ndarray:
        """Run inference, loading model if necessary."""
        if self.model is None:
            self.load()
        
        if self.model is None:
            raise RuntimeError("Model could not be loaded due to memory constraints")
        
        return self.model.run(None, {"input": data})[0]
    
    # method to get input and output names
    def get_input_output_names(self):
        if self.model is None:
            self.load()
        
        if self.model is None:
            raise RuntimeError("Model could not be loaded due to memory constraints")
        
        return [input.name for input in self.model.get_inputs()], [output.name for output in self.model.get_outputs()]