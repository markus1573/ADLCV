import torch
import numpy as np
import argparse
import time
from itertools import product
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
from pathlib import Path

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
DEFAULT_TTA_STEPS = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image classification accuracy with TTA across rotations, scales, and model names.")
    parser.add_argument("--model-names", nargs="+", default=[DEFAULT_MODEL_NAME], help="One or more Hugging Face model names.")
    parser.add_argument("--rotations", nargs="+", type=int, default=DEFAULT_ROTATIONS, help="One or more rotation angles in degrees.")
    parser.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES, help="One or more image scale factors.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="How many validation samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for DataLoader and pipeline inference.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="Device for pipeline (-1 for CPU, >=0 for CUDA device index).")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent combo evaluations.")
    parser.add_argument("--tta-steps", type=int, default=DEFAULT_TTA_STEPS, help="Number of TTA rotations")
    return parser.parse_args()

def make_collate_fn(rotation, scale):
    def collate_fn(batch):
        images = [x["image"].rotate(rotation) for x in batch]
        images = [x.resize((max(1, int(x.width * scale)), max(1, int(x.height * scale)))) for x in images]
        labels = [x["class_name"] for x in batch]
        return images, labels
    return collate_fn

class Pipelines:
    pipes = {}
    def get_pipe(self, model_name, task, device, batch_size):
        if model_name not in self.pipes:
            self.pipes[model_name] = pipeline(
                task=task,
                model=model_name,
                dtype=torch.float16 if device >= 0 else torch.float32,
                device=device,
                use_fast=True,
                batch_size=batch_size,
            )
        return self.pipes[model_name]
    
pipelines = Pipelines()

def evaluate_accuracy(model_name, dataset, rotation, scale, batch_size, num_workers, device, tta_steps):
    task = "image-classification"
    pipe = pipelines.get_pipe(model_name, task, device, batch_size)
    pipe_func = lambda images: pipe(images, batch_size=len(images), top_k=None)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device >= 0),
        collate_fn=make_collate_fn(rotation, scale),
    )

    correct_base = 0
    correct_tta = 0
    total = 0
    
    start_time = time.time()

    with torch.inference_mode():
        for images, labels in loader:
            # Base Evaluation
            outputs_base = pipe_func(images)
            pred_labels_base = [out[0]["label"] for out in outputs_base]
            correct_base += sum(p == t or t in p for p, t in zip(pred_labels_base, labels))

            if tta_steps > 1:
                tta_angles = np.linspace(0, 360, tta_steps, endpoint=False)
                tta_scores = []
                for angle in tta_angles:
                    tta_imgs = [img.rotate(angle) for img in images]
                    out_tta = pipe_func(tta_imgs)
                    tta_scores.append(out_tta)
                
                # aggregate TTA scores per image
                tta_pred_labels = []
                for i in range(len(images)):
                    label_scores = {}
                    for step_idx in range(tta_steps):
                        for item in tta_scores[step_idx][i]:
                            l, s = item["label"], item["score"]
                            label_scores[l] = label_scores.get(l, 0) + s
                    best_label = max(label_scores, key=label_scores.get)
                    tta_pred_labels.append(best_label)
                    
                correct_tta += sum(p == t or t in p for p, t in zip(tta_pred_labels, labels))
            
            total += len(labels)

    elapsed = time.time() - start_time
    time_per_image = elapsed / total if total > 0 else 0
    
    acc_base = correct_base / total if total > 0 else np.nan
    acc_tta = correct_tta / total if (total > 0 and tta_steps > 1) else np.nan
    
    return acc_base, acc_tta, time_per_image

def main():
    args = parse_args()
    dataset = load_dataset("Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation")
    dataset = dataset.select(range(min(args.n, len(dataset))))

    combos = list(product(args.model_names, args.rotations, args.scales))
    print(f"Evaluating {len(dataset)} samples. TTA steps: {args.tta_steps}")
    print("model_name\trotation\tscale\tbase_acc\ttta_acc\ttime/img(s)")

    for combo in combos:
        model_name, rotation, scale = combo
        base_acc, tta_acc, time_per_img = evaluate_accuracy(
            model_name, dataset, rotation, scale, args.batch_size, 
            args.num_workers, args.device, args.tta_steps
        )
        print(f"{model_name[:20]}\t{rotation}\t{scale}\t{base_acc:.4f}\t{tta_acc:.4f}\t{time_per_img:.4f}")

if __name__ == "__main__":
    main()
