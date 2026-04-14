import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
m = "google/siglip-so400m-patch14-384"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(m)
model = AutoModel.from_pretrained(m).to(device)

# Fix for config
model.config.output_hidden_states = True
model.vision_model.config.output_hidden_states = True

dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
inputs = processor(images=dummy_image, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.vision_model(**inputs, output_hidden_states=True)

print("Vision Model output keys:", out.keys())
