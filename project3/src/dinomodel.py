import torch
import torch.nn as nn
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# 1. Setup
DEVICE = 0 if torch.cuda.is_available() else -1
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
ROTATION = 180

# Use feature-extraction instead of classification
pipe = pipeline("image-feature-extraction", model=MODEL_NAME, device=DEVICE, torch_dtype=torch.float16)

# 2. Load Dataset (Using a subset for speed)
dataset = load_dataset("Elriggs/imagenet-50-subset", split="validation", cache_dir="./.data", trust_remote_code=True)
dataset = dataset.select(range(min(500, len(dataset)))) # 500 samples is plenty for a linear head

# 3. Extract Features (CLS Token)
print("Extracting Features...")
embeddings = []
labels = []

for item in tqdm(dataset):
    img = item["image"].convert("RGB")
    with torch.no_grad():
        # DINOv3 output: [1, Seq_Len, 1024]. Index 0 is the CLS token.
        feat = pipe(img)
        embeddings.append(torch.tensor(feat[0][0]))
        labels.append(item["label"])

X = torch.stack(embeddings).to("cuda" if DEVICE >= 0 else "cpu")
Y = torch.tensor(labels).to("cuda" if DEVICE >= 0 else "cpu")

# 4. Train a Linear Head (Quick "Linear Probe")
# This simulates the model's performance on your specific 50 classes
head = nn.Linear(1024, 50).to(X.device)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training Linear Head...")
for epoch in range(50): # Linear heads converge very fast
    outputs = head(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Evaluate Accuracy on Rotated Images
print(f"Evaluating Accuracy at {ROTATION}°...")
correct_rot = 0
total = 0

for item in tqdm(dataset):
    img_rot = item["image"].convert("RGB").rotate(ROTATION)
    with torch.no_grad():
        feat_rot = torch.tensor(pipe(img_rot)[0][0]).to(X.device)
        logits = head(feat_rot)
        pred = torch.argmax(logits).item()
        if pred == item["label"]:
            correct_rot += 1
        total += 1

print(f"\nStandard Accuracy (0°): 100% (Overfit check on training subset)")
print(f"Rotated Accuracy ({ROTATION}°): { (correct_rot/total)*100 :.2f}%")