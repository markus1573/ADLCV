import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
m = "google/siglip-so400m-patch14-384"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading {m} on {device}")
processor = AutoImageProcessor.from_pretrained(m)
model = AutoModel.from_pretrained(m, output_hidden_states=True).to(device)

if hasattr(model, 'vision_model'):
    print("Using vision_model")
    model = model.vision_model

dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
inputs = processor(images=dummy_image, return_tensors="pt").to(device)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

print("Output keys:", out.keys() if hasattr(out, 'keys') else type(out))
if hasattr(out, 'hidden_states') and out.hidden_states is not None:
    print(f"Number of hidden states: {len(out.hidden_states)}")
    print(f"Hidden state shape: {out.hidden_states[0].shape}\n")
else:
    print("hidden_states is None or missing")
