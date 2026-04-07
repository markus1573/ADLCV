import torch
import numpy as np
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "facebook/dinov2-small-imagenet1k-1-layer"
N = 1000                 # how many samples to evaluate
BATCH_SIZE = 32          # tune this based on your GPU memory
NUM_WORKERS = 4          # tune based on CPU
DEVICE = 0 if torch.cuda.is_available() else -1
ROTATION = 180

# ----------------------------
# Pipeline
# ----------------------------
pipe = pipeline(
    task="image-classification",
    model=MODEL_NAME,
    dtype=torch.float16 if DEVICE >= 0 else torch.float32,
    device=DEVICE,
    use_fast=False
)

# ----------------------------
# Dataset
# ----------------------------
dataset = load_dataset("Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation")

# Keep only what you need
dataset = dataset.select(range(min(N, len(dataset))))

def collate_fn(batch):
    # Rotate images to showcase invariance to orientation
    images = [x["image"].rotate(ROTATION) for x in batch]
    labels = [x["class_name"] for x in batch]
    return images, labels

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE >= 0),
    collate_fn=collate_fn
)

# ----------------------------
# Inference
# ----------------------------
correct = 0
total = 0

with torch.inference_mode():
    for images, labels in loader:
        # Pass a whole batch at once
        outputs = pipe(images, batch_size=len(images))

        # outputs: list[ list[{"label": ..., "score": ...}, ...] ]
        pred_labels = [out[0]["label"] for out in outputs]

        correct += sum(pred == true or true in pred for pred, true in zip(pred_labels, labels))
        total += len(labels)

acc = correct / total
print(f"Accuracy: {acc:.4f}")