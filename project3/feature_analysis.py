import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from typing import List, Dict

def get_target_layers(model_name: str, model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Finds the encoder layers for the respective architectures so we can attach hooks.
    """
    if "siglip" in model_name:
        # SigLIP wraps the vision encoder
        layers = model.vision_model.encoder.layers
    elif "vit" in model_name:
        # Standard ViT
        layers = model.encoder.layer
    elif "dinov2" in model_name:
        # DINOv2
        layers = model.encoder.layer
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
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # For SigLIP, load the AutoModel directly. ViT/DINO will load their base models.
        self.model = AutoModel.from_pretrained(model_name).to(device)
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
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, "vision_model"):
                # Pass directly to vision_model to avoid text-branch requirements
                self.model.vision_model(pixel_values=inputs["pixel_values"])
            else:
                self.model(**inputs)
                
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
