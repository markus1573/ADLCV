import torch
import numpy as np
import argparse
from itertools import product
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path to import dinomodel
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dinomodel import load_or_train_head, predict_with_head

# ----------------------------
# Config
# ----------------------------
DEFAULT_MODEL_NAME = "facebook/dinov2-small-imagenet1k-1-layer"
DEFAULT_N = 1000  # how many samples to evaluate
DEFAULT_BATCH_SIZE = 32  # tune this based on your GPU memory
DEFAULT_NUM_WORKERS = 4  # tune based on CPU
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_ROTATIONS = [180]  # 0-360
DEFAULT_SCALES = [1.0]  # 0.5-2.0
DEFAULT_MAX_WORKERS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image classification accuracy across rotations, scales, and model names."
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=[DEFAULT_MODEL_NAME],
        help="One or more Hugging Face model names.",
    )
    parser.add_argument(
        "--rotations",
        nargs="+",
        type=int,
        default=DEFAULT_ROTATIONS,
        help="One or more rotation angles in degrees.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=DEFAULT_SCALES,
        help="One or more image scale factors.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="How many validation samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for DataLoader and pipeline inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help="Device for pipeline (-1 for CPU, >=0 for CUDA device index).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of concurrent combo evaluations. Use 1 for sequential execution.",
    )
    return parser.parse_args()


def make_collate_fn(rotation, scale):
    def collate_fn(batch):
        images = [x["image"].rotate(rotation) for x in batch]
        images = [
            x.resize(
                (
                    max(1, int(x.width * scale)),
                    max(1, int(x.height * scale)),
                )
            )
            for x in images
        ]
        labels = [x["class_name"] for x in batch]
        return images, labels

    return collate_fn


def is_dinov3_model(model_name):
    """Check if model is a DINOv3 model that requires a linear head."""
    return "dinov3" in model_name.lower()


def evaluate_accuracy_dinov3(
    model_name, dataset, rotation, scale, batch_size, num_workers, device
):
    """Evaluate accuracy for DINOv3 models using feature extraction + linear head."""
    # Load or train the head
    head, pipe = load_or_train_head(model_name, device)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device >= 0),
        collate_fn=make_collate_fn(rotation, scale),
    )

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            # Get feature predictions
            pred_indices = predict_with_head(images, head, pipe, device, batch_size)
            
            # Get class names from dataset
            class_names = dataset.features["label"].names
            pred_labels = [class_names[idx] for idx in pred_indices]

            correct += sum(pred == true for pred, true in zip(pred_labels, labels))
            total += len(labels)

    return correct / total if total > 0 else np.nan


def evaluate_accuracy(
    model_name, dataset, rotation, scale, batch_size, num_workers, device
):
    """Evaluate accuracy for model. Routes to DINOv3 handler if needed."""
    if is_dinov3_model(model_name):
        return evaluate_accuracy_dinov3(
            model_name, dataset, rotation, scale, batch_size, num_workers, device
        )
    
    # Standard pipeline-based models
    if "siglip" in model_name:
        task = "zero-shot-image-classification"
        labels = dataset.features["label"].names
        pipe_func = lambda images: pipe(images, batch_size=len(images), candidate_labels=labels)
    else:
        task = "image-classification"
        labels = None
        pipe_func = lambda images: pipe(images, batch_size=len(images))

    pipe = pipeline(
        task=task,
        model=model_name,
        dtype=torch.float16 if device >= 0 else torch.float32,
        device=device,
        use_fast=True,
        batch_size=batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device >= 0),
        collate_fn=make_collate_fn(rotation, scale),
    )

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            outputs = pipe_func(images)
            pred_labels = [out[0]["label"] for out in outputs]

            correct += sum(
                pred == true or true in pred for pred, true in zip(pred_labels, labels)
            )
            total += len(labels)

    return correct / total if total > 0 else np.nan


def main():
    args = parse_args()

    dataset = load_dataset(
        "Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation"
    )
    dataset = dataset.select(range(min(args.n, len(dataset))))

    combos = list(product(args.model_names, args.rotations, args.scales))

    print("model_name\trotation\tscale\taccuracy")

    def run_combo(combo):
        model_name, rotation, scale = combo
        acc = evaluate_accuracy(
            model_name=model_name,
            dataset=dataset,
            rotation=rotation,
            scale=scale,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        return model_name, rotation, scale, acc

    if args.max_workers <= 1:
        for combo in combos:
            model_name, rotation, scale, acc = run_combo(combo)
            print(f"{model_name}\t{rotation}\t{scale}\t{acc:.4f}")
    else:
        for combo in combos:
            future = run_combo(combo) 
            model_name, rotation, scale, acc = future
            print(f"{model_name}\t{rotation}\t{scale}\t{acc:.4f}")


if __name__ == "__main__":
    main()
