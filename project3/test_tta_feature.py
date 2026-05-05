import torch
import numpy as np
import argparse
import time
from itertools import product
import csv
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

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
DEFAULT_TTA_STEPS = 6

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image classification accuracy with Feature Mean TTA.")
    parser.add_argument("--model-names", nargs="+", default=[DEFAULT_MODEL_NAME], help="One or more Hugging Face model names.")
    parser.add_argument("--rotations", nargs="+", type=int, default=DEFAULT_ROTATIONS, help="One or more rotation angles in degrees.")
    parser.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES, help="One or more image scale factors.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="How many validation samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for DataLoader.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="Device (-1 for CPU, >=0 for CUDA device index).")
    parser.add_argument("--tta-steps", type=int, default=DEFAULT_TTA_STEPS, help="Number of TTA rotations")
    parser.add_argument("--output-csv", type=str, default="results_tta_feature.csv", help="Path to save the results in CSV format.")
    return parser.parse_args()

def make_collate_fn(rotation, scale):
    def collate_fn(batch):
        images = [x["image"].convert("RGB").rotate(rotation) for x in batch]
        images = [x.resize((max(1, int(x.width * scale)), max(1, int(x.height * scale)))) for x in images]
        labels = [x["class_name"] for x in batch]
        return images, labels
    return collate_fn

class Pipelines:
    pipes = {}
    def get_pipe(self, model_name, task, device_idx):
        if model_name not in self.pipes:
            self.pipes[model_name] = pipeline(
                task=task,
                model=model_name,
                device=device_idx,
                use_fast=True
            )
        return self.pipes[model_name]

pipelines = Pipelines()

def evaluate_accuracy(model_name, dataset, rotation, scale, batch_size, num_workers, device_idx, tta_steps):
    device = torch.device(f"cuda:{device_idx}" if device_idx >= 0 else "cpu")
    
    is_siglip = "siglip" in model_name
    task = "zero-shot-image-classification" if is_siglip else "image-classification"
    
    pipe = pipelines.get_pipe(model_name, task, device_idx)
    processor = pipe.image_processor
    model = pipe.model
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(rotation, scale),
    )

    correct_base = 0
    correct_tta = 0
    total = 0
    
    time_base_total = 0.0
    time_tta_total = 0.0

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"Eval {model_name}"):
            
            inputs = processor(images=images, return_tensors="pt").to(device)
            if inputs.pixel_values.dtype == torch.float32 and model.dtype == torch.float16:
                 inputs.pixel_values = inputs.pixel_values.half()
            
            # --- BASE EVALUATION ---
            t0 = time.perf_counter()
            outputs_base = model(**inputs)
            logits_base = outputs_base.logits
            preds_base = logits_base.argmax(dim=-1).cpu().numpy()
            t1 = time.perf_counter()
            
            time_base_total += (t1 - t0)

            pred_labels_base = [model.config.id2label[p] for p in preds_base]
            correct_base += sum(p == t or t in p for p, t in zip(pred_labels_base, labels))

            # --- TTA EVALUATION ---
            if tta_steps > 1:
                t2 = time.perf_counter()
                
                tta_angles = np.linspace(0, 360, tta_steps, endpoint=False)
                all_features = []
                
                # We can use a hook to get the exact input to the classifier
                def hook_fn(module, args):
                    all_features.append(args[0].detach())
                    
                hook_handle = model.classifier.register_forward_pre_hook(hook_fn)
                
                for angle in tta_angles:
                    tta_imgs = [img.rotate(angle) for img in images]
                    tta_inputs = processor(images=tta_imgs, return_tensors="pt").to(device)
                    if tta_inputs.pixel_values.dtype == torch.float32 and model.dtype == torch.float16:
                        tta_inputs.pixel_values = tta_inputs.pixel_values.half()
                    
                    # Forward pass to trigger the hook and collect features
                    model(**tta_inputs)
                    
                hook_handle.remove()
                
                # Mean features
                mean_features = torch.stack(all_features, dim=0).mean(dim=0)
                
                # Classify
                logits_tta = model.classifier(mean_features)
                preds_tta = logits_tta.argmax(dim=-1).cpu().numpy()
                
                pred_labels_tta = [model.config.id2label[p] for p in preds_tta]
                correct_tta += sum(p == t or t in p for p, t in zip(pred_labels_tta, labels))
                
                t3 = time.perf_counter()
                time_tta_total += (t3 - t2)
            
            total += len(labels)

    acc_base = correct_base / total if total > 0 else np.nan
    acc_tta = correct_tta / total if (total > 0 and tta_steps > 1) else np.nan
    
    time_per_img_base = time_base_total / total if total > 0 else 0
    time_per_img_tta = time_tta_total / total if (total > 0 and tta_steps > 1) else 0
    total_time_per_img = time_per_img_base + time_per_img_tta
    time_multiplier = total_time_per_img / time_per_img_base if time_per_img_base > 0 else 0
    
    return acc_base, acc_tta, time_per_img_base, total_time_per_img, time_multiplier

def main():
    args = parse_args()
    dataset = load_dataset("Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42).select(range(min(args.n, len(dataset))))

    combos = list(product(args.model_names, args.rotations, args.scales))
    print(f"Evaluating {len(dataset)} samples. TTA steps: {args.tta_steps}")
    
    print(f"{'model_name':<22}\t{'rot'}\t{'scale'}\t{'base_acc'}\t{'tta_acc'}\t{'base_t/img'}\t{'total_t/img'}\t{'multiplier'}")
    print("-" * 105)

    csv_data = []
    
    for combo in combos:
        model_name, rotation, scale = combo
        base_acc, tta_acc, base_time, total_time, multiplier = evaluate_accuracy(
            model_name, dataset, rotation, scale, args.batch_size, 
            args.num_workers, args.device, args.tta_steps
        )
        
        csv_data.append([model_name, rotation, scale, base_acc, tta_acc, base_time, total_time, multiplier])
        
        m_name = model_name.split("/")[-1][:22]
        print(f"{m_name:<22}\t{rotation}\t{scale}\t{base_acc:.4f}\t\t{tta_acc:.4f}\t\t{base_time:.4f}s\t\t{total_time:.4f}s\t\t{multiplier:.1f}x")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "rotation", "scale", "base_acc", "feature_tta_acc", "base_t_img", "total_t_img", "multiplier"])
        writer.writerows(csv_data)
        
    print(f"\nResults saved to {args.output_csv}\n")

if __name__ == "__main__":
    main()
