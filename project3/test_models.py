import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
models = [
    "facebook/dinov2-small-imagenet1k-1-layer",
    "google/vit-base-patch16-224",
    "google/siglip-so400m-patch14-384"
]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

for m in models:
    print(f"Loading {m}")
    processor = AutoImageProcessor.from_pretrained(m)
    model = AutoModel.from_pretrained(m, output_hidden_states=True).to(device)
    if hasattr(model, 'vision_model'):
        model = model.vision_model
    
    # Use random Image in right format
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    
    print(f"Number of hidden states: {len(out.hidden_states)}")
    print(f"Hidden state shape: {out.hidden_states[0].shape}\n")
