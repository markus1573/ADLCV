import torch
import torch.nn as nn
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import os
from pathlib import Path

# Setup
DEVICE = 0 if torch.cuda.is_available() else -1
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
FEATURE_DIM = 1024
NUM_CLASSES = 50
HEADS_DIR = Path(__file__).parent.parent / "heads"


def get_head_path(model_name):
    """Get the path where the head for a model should be saved."""
    HEADS_DIR.mkdir(exist_ok=True)
    # Create a safe filename from model name
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return HEADS_DIR / f"{safe_name}_head.pth"


def train_linear_head(model_name=MODEL_NAME, device=DEVICE, num_samples=500):
    """
    Train a linear head on DINOv3 features for 50-class classification.
    Returns the trained head and the feature extraction pipeline.
    """
    print(f"--- Training Linear Head for {model_name} ---")
    
    # Load pipeline and dataset
    pipe = pipeline(
        "image-feature-extraction",
        model=model_name,
        device=device,
        torch_dtype=torch.float16,
    )
    dataset = load_dataset(
        "Elriggs/imagenet-50-subset",
        split="validation",
        cache_dir="./.data",
        trust_remote_code=True,
    )
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Extract features
    print("Extracting Features...")
    embeddings = []
    labels = []
    
    for item in tqdm(dataset):
        img = item["image"].convert("RGB")
        with torch.no_grad():
            feat = pipe(img)
            embeddings.append(torch.tensor(feat[0][0]))
            labels.append(item["label"])
    
    X = torch.stack(embeddings).to("cuda" if device >= 0 else "cpu")
    Y = torch.tensor(labels).to(X.device)
    
    # Train linear head
    print("Training Linear Head...")
    head = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(X.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(50):
        outputs = head(X)
        loss = criterion(outputs, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Training Complete!")
    return head, pipe


def load_or_train_head(model_name=MODEL_NAME, device=DEVICE, num_samples=500):
    """
    Load a trained head from disk, or train it if it doesn't exist.
    Returns the head and feature extraction pipeline.
    """
    head_path = get_head_path(model_name)
    
    # Try to load existing head
    if head_path.exists():
        print(f"Loading pre-trained head from {head_path}")
        head = nn.Linear(FEATURE_DIM, NUM_CLASSES)
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        head = head.to("cuda" if device >= 0 else "cpu")
    else:
        # Train new head
        head, _ = train_linear_head(model_name, device, num_samples)
        # Save for future use
        torch.save(head.state_dict(), head_path)
        print(f"Head saved to {head_path}")
    
    # Load pipeline for feature extraction
    pipe = pipeline(
        "image-feature-extraction",
        model=model_name,
        device=device,
        torch_dtype=torch.float16,
    )
    
    return head, pipe


def predict_with_head(images, head, pipe, device=DEVICE, batch_size=32):
    """
    Extract features from images and predict using the trained head.
    Returns predicted class indices.
    """
    predictions = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        
        with torch.no_grad():
            # Extract features
            features = pipe(batch_images)  # List of [1, 1, FEATURE_DIM]
            features_tensor = torch.tensor(
                [f[0][0] for f in features]
            ).to("cuda" if device >= 0 else "cpu")
            
            # Get predictions
            logits = head(features_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    return predictions


if __name__ == "__main__":
    head, pipe = train_linear_head()
    head_path = get_head_path(MODEL_NAME)
    torch.save(head.state_dict(), head_path)
    print(f"Head saved to {head_path}")
