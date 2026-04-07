import torch
import torch.nn as nn
import numpy as np
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
# Note: Using DINOv3 as the default
DEFAULT_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_N = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_ROTATIONS = [0, 180] 
DEFAULT_SCALES = [1.0]
DEFAULT_MAX_WORKERS = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 accuracy across rotations and scales.")
    parser.add_argument("--model-names", nargs="+", default=[DEFAULT_MODEL_NAME])
    parser.add_argument("--rotations", nargs="+", type=int, default=DEFAULT_ROTATIONS)
    parser.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()

# ----------------------------
# Helper: Linear Head Training
# ----------------------------
def train_linear_head(model_name, dataset, device):
    """
    DINOv3 is just a backbone. We need to train a 1-layer head 
    on 'upright' images first so we have an accuracy metric to test.
    """
    print(f"--- Calibrating Linear Head for {model_name} ---")
    pipe = pipeline("image-feature-extraction", model=model_name, device=device, torch_dtype=torch.float16)
    
    feats, labels = [], []
    # Use a subset of the data to 'teach' the labels
    for i in range(min(len(dataset), 500)): 
        item = dataset[i]
        img = item["image"].convert("RGB")
        with torch.no_grad():
            out = pipe(img)
            feats.append(torch.tensor(out[0][0])) # CLS token
            labels.append(item["label"])
    
    X = torch.stack(feats).to("cuda" if device >= 0 else "cpu")
    Y = torch.tensor(labels).to(X.device)
    
    # 1024 is the embedding dim for ViT-L/16
    head = nn.Linear(1024, 50).to(X.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
# 2. Load Dataset (Using a subset for speed)
dataset = load_dataset("Elriggs/imagenet-50-subset", split="validation", cache_dir="./.data", trust_remote_code=True)
dataset = dataset.select(range(min(500, len(dataset)))) # 500 samples is plenty for a linear head

    for _ in range(50):
        loss = crit(head(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return head.eval()

# ----------------------------
# Core Logic
# ----------------------------
def make_collate_fn(rotation, scale):
    def collate_fn(batch):
        # Apply transformations
        images = [x["image"].convert("RGB").rotate(rotation) for x in batch]
        if scale != 1.0:
            images = [img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale)))) for img in images]
        labels = [x["label"] for x in batch] # Using integer labels for accuracy
        return images, labels
    return collate_fn

def evaluate_accuracy(model_name, dataset, rotation, scale, batch_size, num_workers, device, trained_head):
    # Use feature-extraction since DINOv3 doesn't have a native class head
    pipe = pipeline("image-feature-extraction", model=model_name, device=device, torch_dtype=torch.float16)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate_fn(rotation, scale),
    )

    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            # 1. Get DINOv3 Features
            outputs = pipe(images, batch_size=len(images))
            # 2. Extract CLS tokens [Batch, 1024]
            cls_tokens = torch.stack([torch.tensor(out[0][0]) for out in outputs]).to("cuda" if device >= 0 else "cpu")
            # 3. Pass through our calibrated head
            logits = trained_head(cls_tokens)
            preds = torch.argmax(logits, dim=1)
            
            targets = torch.tensor(labels).to(preds.device)
            correct += (preds == targets).sum().item()
            total += len(labels)

    return correct / total

def main():
    args = parse_args()
    dataset = load_dataset("Elriggs/imagenet-50-subset", split="validation", trust_remote_code=True)
    dataset = dataset.select(range(min(args.n, len(dataset))))

    # Map model names to their respective trained heads
    model_heads = {}
    for m_name in args.model_names:
        model_heads[m_name] = train_linear_head(m_name, dataset, args.device)

    combos = list(product(args.model_names, args.rotations, args.scales))
    print("\nmodel_name\trotation\tscale\taccuracy")

    for m_name, rot, scale in combos:
        acc = evaluate_accuracy(m_name, dataset, rot, scale, args.batch_size, args.num_workers, args.device, model_heads[m_name])
        print(f"{m_name}\t{rot}\t{scale}\t{acc:.4f}")

if __name__ == "__main__":
    main()