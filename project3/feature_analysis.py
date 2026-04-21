import torch
import numpy as np
from transformers import pipeline, AutoImageProcessor
from typing import List, Dict
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

import metrics

def get_target_layers(model_name: str, model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Finds the encoder layers for the respective architectures so we can attach hooks.
    """
    if "siglip" in model_name:
        # SigLIP wraps the vision encoder
        layers = model.vision_model.encoder.layers
    elif "vit" in model_name:
        # Standard ViT (handle pipeline wrappers like ViTForImageClassification)
        base_model = getattr(model, 'vit', model)
        layers = base_model.encoder.layer
    elif "dinov2" in model_name:
        # DINOv2 (handle pipeline wrappers like Dinov2ForImageClassification)
        base_model = getattr(model, 'dinov2', model)
        layers = base_model.encoder.layer
    else:
        raise ValueError(f"Architecture for {model_name} not supported.")
    
    # Select 5 evenly distributed layers (first, 3 middle, last)
    total_layers = len(layers)
    target_indices = np.linspace(0, total_layers - 1, 5, dtype=int)
    print(f"[{model_name}] Hooking into layer indices: {target_indices}")
    
    return [layers[i] for i in target_indices]

class FeatureExtractor:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        print(f"Loading {model_name}...")
        
        # Use image-feature-extraction to bypass the need for a tokenizer
        task = "image-feature-extraction"
        
        pipe_device = "mps" if device.type == "mps" else (device.index if device.type == "cuda" else "cpu")
        
        # Load image processor explicitly
        img_processor = AutoImageProcessor.from_pretrained(model_name)

        self.pipe = pipeline(
            task,
            model=model_name,
            image_processor=img_processor,
            device=pipe_device,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        )
        
        # The underlying model is still accessible for hooking
        self.model = self.pipe.model
        self.model.eval()

        self.features = []
        self.hook_handles = []
        target_layers = get_target_layers(model_name, self.model)
        for layer in target_layers:
            self.hook_handles.append(layer.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, input, output):
        # Hidden states are usually the first element in the output tuple
        hidden_state = output[0] if isinstance(output, tuple) else output
        # Pooled representation [Batch, Dim]
        pooled_state = hidden_state.mean(dim=1).detach().cpu()
        self.features.append(pooled_state)

    def extract(self, images: List[Image.Image]) -> List[torch.Tensor]:
            self.features.clear()
            
            # Preprocess images using the pipeline's image_processor
            # We ensure they are moved to the correct device and dtype
            inputs = self.pipe.image_processor(images, return_tensors="pt").to(
                self.device, 
                dtype=torch.float16 if self.device.type != "cpu" else torch.float32
            )
            
            with torch.inference_mode():
                if hasattr(self.model, "vision_model"):
                    # For SigLIP/CLIP: call the vision tower directly
                    self.model.vision_model(**inputs)
                else:
                    # For ViT/DINOv2: call the model directly
                    self.model(**inputs)
                    
            return self.features.copy()

    def cleanup(self):
        for handle in self.hook_handles:
            handle.remove()

def make_collate_fn(rotation: int):
    def collate_fn(batch):
        # We apply the target rotation and ensure they are all in RGB format as PIL Images
        images = [x["image"].convert("RGB").rotate(rotation) for x in batch]
        return images
    return collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of images to process.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for pipeline.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    models = [
        "facebook/dinov2-small-imagenet1k-1-layer",
        "google/vit-base-patch16-224",
        "google/siglip-so400m-patch14-384"
    ]
    
    angles = [0, 90, 180, 270]

    # Load dataset
    dataset = load_dataset(
        "Elriggs/imagenet-50-subset", cache_dir="./.data", split="validation", trust_remote_code=True
    )
    dataset = dataset.select(range(min(args.n, len(dataset))))

    print("--- Running Feature-Level Analysis ---")

    results = {}

    for m in models:
        extractor = FeatureExtractor(m, device)
        results[m] = {}

        for angle in angles:
            print(f"\nProcessing angle: {angle}°")
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=make_collate_fn(angle),
            )
            
            # Since pipeline can yield chunked callbacks naturally, we collect accumulated hooked features.
            all_batches_features = {i: [] for i in range(5)}
            
            for batch_images in tqdm(loader, desc=f"Evaluating {m} @ {angle}°"):
                extracted_layers = extractor.extract(batch_images)
                
                for layer_idx, feats in enumerate(extracted_layers):
                    all_batches_features[layer_idx].append(feats)
            
            # Concatenate collected batches across the whole dataset (N=samples, D=hidden_dim)
            results[m][angle] = [
                torch.cat(all_batches_features[i], dim=0) for i in range(5)
            ]
        
        extractor.cleanup()
        print("-" * 50)
        
    print("\nExtraction complete! Data arrays ready for similarity evaluation.")

    import matplotlib.pyplot as plt
    import os
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # eval_results[metric][model][angle] = list of layer scores
    eval_results = {
        "CKA": {m: {a: [] for a in angles[1:]} for m in models},
        "Cosine": {m: {a: [] for a in angles[1:]} for m in models},
        "RSA": {m: {a: [] for a in angles[1:]} for m in models}
    }
    
    print("Computing metrics...")
    for m in models:
        for angle in angles[1:]:
            for layer_idx in range(5):
                # Bring back to device for metric computation
                baseline_feats = results[m][0][layer_idx].to(device)
                trans_feats = results[m][angle][layer_idx].to(device)
                
                # CKA
                cka_score = metrics.linear_cka(baseline_feats, trans_feats).item()
                # Cosine (mean over batch)
                cos_score = metrics.centered_cosine_similarity(baseline_feats, trans_feats).mean().item()
                # RSA
                rsa_score = metrics.rsa(baseline_feats, trans_feats).item()
                
                eval_results["CKA"][m][angle].append(cka_score)
                eval_results["Cosine"][m][angle].append(cos_score)
                eval_results["RSA"][m][angle].append(rsa_score)
                
    # Plotting
    print("Generating plots...")
    layer_ticks = ["First", "Early-Mid", "Mid", "Late-Mid", "Last"]
    
    for metric_name, model_data in eval_results.items():
        plt.figure(figsize=(15, 5))
        for i, m_name in enumerate(models):
            plt.subplot(1, 3, i+1)
            plt.title(m_name.split('/')[-1])
            for angle in angles[1:]:
                plt.plot(layer_ticks, model_data[m_name][angle], marker='o', label=f"{angle}°")
            plt.xlabel("Layers")
            plt.ylabel(f"{metric_name} Similarity")
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            if i == 2:
                plt.legend()
                
        plt.tight_layout()
        plt_path = os.path.join(out_dir, f"{metric_name.lower()}_similarity.png")
        plt.savefig(plt_path)
        plt.close()
        print(f"Saved {plt_path}")
        
    print("Done!")
