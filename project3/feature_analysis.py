import torch
import numpy as np
from transformers import pipeline
from typing import List, Dict
from PIL import Image

def get_target_layers(model_name: str, model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Finds the encoder layers for the respective architectures so we can attach hooks.
    """
    if "siglip" in model_name:
        # SigLIP wraps the vision encoder
        layers = model.vision_model.encoder.layers
    elif "vit" in model_name:
        # Standard ViT (handle pipeline wrappers like ViTForImageClassification)
        base_model = getattr(model, 'vit', model)
        layers = base_model.encoder.layer
    elif "dinov2" in model_name:
        # DINOv2 (handle pipeline wrappers like Dinov2ForImageClassification)
        base_model = getattr(model, 'dinov2', model)
        layers = base_model.encoder.layer
    else:
        raise ValueError(f"Architecture for {model_name} not supported.")
    
    # Select 5 evenly distributed layers (first, 3 middle, last)
    total_layers = len(layers)
    target_indices = np.linspace(0, total_layers - 1, 5, dtype=int)
    print(f"[{model_name}] Hooking into layer indices: {target_indices}")
    
    return [layers[i] for i in target_indices]

class FeatureExtractor:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        print(f"Loading {model_name}...")
        
        # Determine task type identical to test_imagenet.py
        if "siglip" in model_name:
            task = "zero-shot-image-classification"
        else:
            task = "image-classification"
            
        # pipeline needs device argument as integer index or 'cpu'/'mps' string directly
        pipe_device = "mps" if device.type == "mps" else (device.index if device.type == "cuda" else "cpu")
        
        self.pipe = pipeline(
            task,
            model=model_name,
            device=pipe_device,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            use_fast=True  # As in test_imagenet.py
        )
        self.model = self.pipe.model
        self.model.eval()

        # Storage for our intercepted features
        self.features = []
        
        # Attach hooks
        self.hook_handles = []
        target_layers = get_target_layers(model_name, self.model)
        for layer in target_layers:
            self.hook_handles.append(layer.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, input, output):
        # depending on the model, output might be a tuple. The hidden states are always the first element.
        hidden_state = output[0] if isinstance(output, tuple) else output
        self.features.append(hidden_state.detach().cpu())

    def extract(self, image: np.ndarray) -> List[torch.Tensor]:
        self.features.clear() # Reset storage
        
        # Pipelines prefer PIL images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Run through the pipeline directly without accessing vision_model
        with torch.inference_mode():
            if self.pipe.task == "zero-shot-image-classification":
                # Requires candidate labels for passing through the text encoder
                self.pipe(image, candidate_labels=["dummy reference"])
            else:
                self.pipe(image)
                
        return self.features.copy()

    def cleanup(self):
        # Remove hooks when done to prevent memory leaks
        for handle in self.hook_handles:
            handle.remove()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    models = [
        "facebook/dinov2-small-imagenet1k-1-layer",
        "google/vit-base-patch16-224",
        "google/siglip-so400m-patch14-384"
    ]
    
    dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

    for m in models:
        extractor = FeatureExtractor(m, device)
        extracted_layers = extractor.extract(dummy_image)
        
        print(f"\n{m} Extraction Results:")
        print(f"Number of layers intercepted: {len(extracted_layers)}")
        for i, feat in enumerate(extracted_layers):
            print(f" Layer {i} feature shape: {feat.shape}")
        
        extractor.cleanup()
        print("-" * 50)
