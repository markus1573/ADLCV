import torch
import numpy as np
import argparse
import time
from itertools import product
import csv
from collections import Counter
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader

# ----------------------------
# Config
# ----------------------------
DEFAULT_MODEL_NAME = "facebook/dinov2-small-imagenet1k-1-layer"
DEFAULT_N = 1000  
DEFAULT_BATCH_SIZE = 32  
DEFAULT_NUM_WORKERS = 4  
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_ROTATIONS = [180]  
DEFAULT_SCALES = [1.0]  
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
    parser.add_argument("--tta-method", type=str, choices=['entropy', 'confidence', 'majority'], default='entropy', help="Method to aggregate TTA scores.")
    parser.add_argument("--output-csv", type=str, default="results_tta.csv", help="Path to save the results in CSV format.")
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

def evaluate_accuracy(model_name, dataset, rotation, scale, batch_size, num_workers, device, tta_steps, tta_method):
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
    
    # Isolate timers
    time_base_total = 0.0
    time_tta_total = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            
            # --- BASE EVALUATION ---
            t0 = time.perf_counter()
            outputs_base = pipe_func(images)
            t1 = time.perf_counter()
            
            time_base_total += (t1 - t0)
            
            pred_labels_base = [out[0]["label"] for out in outputs_base]
            correct_base += sum(p == t or t in p for p, t in zip(pred_labels_base, labels))

            # --- TTA EVALUATION ---
            if tta_steps > 1:
                t2 = time.perf_counter()
                
                tta_angles = np.linspace(0, 360, tta_steps, endpoint=False)
                tta_scores = []
                
                for angle in tta_angles:
                    # Optimization: If angle is 0, we already did the forward pass!
                    if angle == 0.0:
                        tta_scores.append(outputs_base)
                        continue
                        
                    tta_imgs = [img.rotate(angle) for img in images]
                    out_tta = pipe_func(tta_imgs)
                    tta_scores.append(out_tta)
                
                # Aggregate TTA scores per image
                tta_pred_labels = []
                for i in range(len(images)):
                    if tta_method == 'entropy':
                        best_label = None
                        min_entropy = float('inf')
                        for step_idx in range(tta_steps):
                            entropy = 0.0
                            top_label = None
                            top_score = -1.0
                            for item in tta_scores[step_idx][i]:
                                l, s = item["label"], item["score"]
                                if s > 0:
                                    entropy -= s * np.log(s + 1e-12)
                                if s > top_score:
                                    top_score = s
                                    top_label = l
                            if entropy < min_entropy:
                                min_entropy = entropy
                                best_label = top_label
                        tta_pred_labels.append(best_label)
                        
                    elif tta_method == 'confidence':
                        best_label = None
                        max_confidence = -1.0
                        for step_idx in range(tta_steps):
                            for item in tta_scores[step_idx][i]:
                                if item["score"] > max_confidence:
                                    max_confidence = item["score"]
                                    best_label = item["label"]
                        tta_pred_labels.append(best_label)
                        
                    elif tta_method == 'majority':
                        step_top_labels = []
                        for step_idx in range(tta_steps):
                            top_label = None
                            top_score = -1.0
                            for item in tta_scores[step_idx][i]:
                                if item["score"] > top_score:
                                    top_score = item["score"]
                                    top_label = item["label"]
                            step_top_labels.append(top_label)
                        
                        # Majority vote
                        vote_counts = Counter(step_top_labels)
                        best_label = vote_counts.most_common(1)[0][0]
                        tta_pred_labels.append(best_label)
                    
                correct_tta += sum(p == t or t in p for p, t in zip(tta_pred_labels, labels))
                
                t3 = time.perf_counter()
                time_tta_total += (t3 - t2)
            
            total += len(labels)

    # Calculate metrics
    acc_base = correct_base / total if total > 0 else np.nan
    acc_tta = correct_tta / total if (total > 0 and tta_steps > 1) else np.nan
    
    time_per_img_base = time_base_total / total if total > 0 else 0
    time_per_img_tta = time_tta_total / total if (total > 0 and tta_steps > 1) else 0
    
    # Calculate how much longer total evaluation takes compared to base
    total_time_per_img = time_per_img_base + time_per_img_tta
    time_multiplier = total_time_per_img / time_per_img_base if time_per_img_base > 0 else 0
    
    return acc_base, acc_tta, time_per_img_base, total_time_per_img, time_multiplier

def main():
    args = parse_args()
    dataset = load_dataset("Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation")
    dataset = dataset.select(range(min(args.n, len(dataset))))

    combos = list(product(args.model_names, args.rotations, args.scales))
    print(f"Evaluating {len(dataset)} samples. TTA steps: {args.tta_steps}")
    
    # Updated headers to show isolated base time, total time, and the multiplier
    print(f"{'model_name':<22}\t{'rot'}\t{'scale'}\t{'base_acc'}\t{'tta_acc'}\t{'base_t/img'}\t{'total_t/img'}\t{'multiplier'}")
    print("-" * 105)

    csv_data = []
    
    for combo in combos:
        model_name, rotation, scale = combo
        base_acc, tta_acc, base_time, total_time, multiplier = evaluate_accuracy(
            model_name, dataset, rotation, scale, args.batch_size, 
            args.num_workers, args.device, args.tta_steps, args.tta_method
        )
        
        csv_data.append([model_name, rotation, scale, base_acc, tta_acc, base_time, total_time, multiplier])
        
        m_name = model_name.split("/")[-1][:22] # truncate model name nicely for console
        print(f"{m_name:<22}\t{rotation}\t{scale}\t{base_acc:.4f}\t\t{tta_acc:.4f}\t\t{base_time:.4f}s\t\t{total_time:.4f}s\t\t{multiplier:.1f}x")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "rotation", "scale", "base_acc", "tta_acc", "base_t_img", "total_t_img", "multiplier"])
        writer.writerows(csv_data)
        
    print(f"\nResults saved to {args.output_csv}\n")
if __name__ == "__main__":
    main()