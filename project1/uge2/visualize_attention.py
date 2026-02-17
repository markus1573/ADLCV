"""
Vision Transformer Attention Visualization and Model Analysis
Exercise 1.2 - Tasks 3 & 4

This script provides:
1. Attention map visualization (Task 4 - Figure 6 style from ViT paper)
2. Model configuration experiments (Task 3 - analyzing different choices)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torchvision
from einops import rearrange

from vit import ViT
from imageclassification import prepare_dataloaders, set_seed


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_or_create_model(model_path='model.pth', image_size=(32, 32), patch_size=(4, 4), 
                         channels=3, embed_dim=128, num_heads=4, num_layers=4, 
                         num_classes=2, pos_enc='learnable', pool='cls', dropout=0.3):
    """Load a trained model or create a new one."""
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pos_enc=pos_enc,
        pool=pool,
        dropout=dropout,
        num_classes=num_classes
    )
    
    device = get_device()
    model = model.to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No saved model found at {model_path}. Using randomly initialized model.")
        print("Note: For meaningful attention visualizations, train the model first using imageclassification.py")
    
    return model


def visualize_attention_maps(model, images, class_names=['cat', 'horse'], patch_size=(4, 4)):
    """
    Visualize attention maps similar to Figure 6 in the ViT paper.
    Shows the attention from the [CLS] token to all image patches.
    
    Args:
        model: Trained ViT model
        images: Batch of images (B x C x H x W)
        class_names: Names of the classes for labeling
        patch_size: Size of image patches
    """
    device = get_device()
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        # Get predictions
        logits = model(images)
        predictions = logits.argmax(dim=1)
        
        # Get attention maps from all layers
        attention_maps = model.get_attention_maps(images)
    
    num_images = min(images.size(0), 4)  # Show at most 4 images
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].size(1)
    
    # Calculate number of patches per dimension
    H, W = images.size(2), images.size(3)
    patch_h, patch_w = patch_size
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    
    # Create figure for CLS token attention visualization
    fig, axes = plt.subplots(num_images, num_layers + 1, figsize=(3 * (num_layers + 1), 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Attention from [CLS] token to image patches (averaged over heads)", fontsize=14, y=1.02)
    
    for img_idx in range(num_images):
        # Show original image
        img = images[img_idx].cpu()
        img = (img * 0.5 + 0.5).clamp(0, 1)  # Denormalize
        img = img.permute(1, 2, 0).numpy()
        
        axes[img_idx, 0].imshow(img)
        pred_class = class_names[predictions[img_idx].item()]
        axes[img_idx, 0].set_title(f'Original\nPred: {pred_class}')
        axes[img_idx, 0].axis('off')
        
        # Show attention from CLS token for each layer
        for layer_idx in range(num_layers):
            attn = attention_maps[layer_idx][img_idx]  # num_heads x seq_len x seq_len
            
            # Get attention from CLS token (position 0) to all patches
            # Average over all heads
            cls_attention = attn[:, 0, 1:].mean(dim=0)  # Average over heads, skip CLS token
            cls_attention = cls_attention.cpu().numpy()
            
            # Reshape to 2D grid
            attn_map = cls_attention.reshape(num_patches_h, num_patches_w)
            
            # Upsample to image size for overlay
            attn_map_resized = np.kron(attn_map, np.ones((patch_h, patch_w)))
            
            # Show attention map
            axes[img_idx, layer_idx + 1].imshow(img)
            axes[img_idx, layer_idx + 1].imshow(attn_map_resized, alpha=0.6, cmap='hot')
            axes[img_idx, layer_idx + 1].set_title(f'Layer {layer_idx + 1}')
            axes[img_idx, layer_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_maps_cls_token.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: attention_maps_cls_token.png")


def visualize_attention_heads(model, images, layer_idx=0, patch_size=(4, 4)):
    """
    Visualize attention from different heads in a specific layer.
    
    Args:
        model: Trained ViT model
        images: Batch of images (B x C x H x W)
        layer_idx: Which layer to visualize
        patch_size: Size of image patches
    """
    device = get_device()
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        attention_maps = model.get_attention_maps(images)
    
    # Get attention from specified layer
    attn = attention_maps[layer_idx]  # batch x num_heads x seq_len x seq_len
    num_heads = attn.size(1)
    
    H, W = images.size(2), images.size(3)
    patch_h, patch_w = patch_size
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    
    # Visualize for first image only
    img = images[0].cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(2, num_heads // 2 + 1, figsize=(3 * (num_heads // 2 + 1), 6))
    axes = axes.flatten()
    
    fig.suptitle(f"Attention heads in Layer {layer_idx + 1} (CLS → patches)", fontsize=14)
    
    # Show original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show each head's attention
    for head_idx in range(num_heads):
        cls_attention = attn[0, head_idx, 0, 1:].cpu().numpy()
        attn_map = cls_attention.reshape(num_patches_h, num_patches_w)
        attn_map_resized = np.kron(attn_map, np.ones((patch_h, patch_w)))
        
        axes[head_idx + 1].imshow(img)
        axes[head_idx + 1].imshow(attn_map_resized, alpha=0.6, cmap='hot')
        axes[head_idx + 1].set_title(f'Head {head_idx + 1}')
        axes[head_idx + 1].axis('off')
    
    # Hide unused axes
    for i in range(num_heads + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'attention_heads_layer{layer_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: attention_heads_layer{layer_idx + 1}.png")


def visualize_full_attention_matrix(model, images, layer_idx=-1):
    """
    Visualize the full attention matrix for a given layer.
    
    Args:
        model: Trained ViT model
        images: Single image or batch
        layer_idx: Which layer to visualize (-1 for last)
    """
    device = get_device()
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        attention_maps = model.get_attention_maps(images)
    
    attn = attention_maps[layer_idx][0]  # First image, shape: num_heads x seq_len x seq_len
    num_heads = attn.size(0)
    
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))
    
    fig.suptitle(f"Full Attention Matrix - Layer {layer_idx if layer_idx >= 0 else 'Last'}", fontsize=14)
    
    # Average attention across heads
    avg_attn = attn.mean(dim=0).cpu().numpy()
    sns.heatmap(avg_attn, ax=axes[0], cmap='viridis', square=True, cbar=True)
    axes[0].set_title('Averaged')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # Individual heads
    for head_idx in range(num_heads):
        head_attn = attn[head_idx].cpu().numpy()
        sns.heatmap(head_attn, ax=axes[head_idx + 1], cmap='viridis', square=True, cbar=True)
        axes[head_idx + 1].set_title(f'Head {head_idx + 1}')
        axes[head_idx + 1].set_xlabel('Key Position')
        if head_idx == 0:
            axes[head_idx + 1].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('full_attention_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: full_attention_matrix.png")


# =============================================================================
# Task 3: Model Configuration Analysis
# =============================================================================

def analyze_model_configurations():
    """
    Document how different model choices affect performance.
    This function provides a summary of key hyperparameters and their effects.
    """
    print("=" * 80)
    print("TASK 3: Analysis of Model Configuration Choices")
    print("=" * 80)
    
    configs_info = """
    Key Hyperparameters and Their Effects on ViT Performance:
    
    1. PATCH SIZE (patch_size)
       - Smaller patches (e.g., 4x4): More patches → longer sequences → more computation
         but captures finer details. Better for small images like CIFAR.
       - Larger patches (e.g., 8x8, 16x16): Fewer patches → faster training but may
         lose fine-grained information.
       - For CIFAR 32x32 images:
         * 4x4 patches → 64 patches (recommended for small images)
         * 8x8 patches → 16 patches (faster but less detail)
    
    2. EMBEDDING DIMENSION (embed_dim)
       - Larger dim: More representational capacity but more parameters.
       - Common values: 64, 128, 256, 512
       - For CIFAR: 128 provides good balance between capacity and efficiency.
    
    3. NUMBER OF ATTENTION HEADS (num_heads)
       - More heads: Can learn diverse attention patterns.
       - Must divide embed_dim evenly.
       - Common: 4, 8, 12 heads
       - For small models: 4-8 heads usually sufficient.
    
    4. NUMBER OF TRANSFORMER LAYERS (num_layers)
       - Deeper models: More abstraction but harder to train.
       - For CIFAR with limited data: 4-6 layers work well.
       - Deeper models (8-12) may need more data and regularization.
    
    5. POSITIONAL ENCODING (pos_enc)
       - 'fixed': Sinusoidal encoding (no learnable parameters).
         Good for generalization, works well without much data.
       - 'learnable': Learned position embeddings.
         Can overfit on small datasets but more flexible.
    
    6. POOLING STRATEGY (pool)
       - 'cls': Use [CLS] token for classification (standard ViT approach).
       - 'mean': Average all patch embeddings.
       - 'max': Max pooling over patches.
       - 'cls' is standard; 'mean' sometimes works better for smaller models.
    
    7. DROPOUT
       - Higher dropout (0.3-0.5): Better regularization for small datasets.
       - Lower dropout (0.0-0.1): For larger datasets or when using other regularization.
    
    8. DATA AUGMENTATION (in prepare_dataloaders)
       - Crucial for ViT on small datasets!
       - Recommended augmentations:
         * RandomHorizontalFlip
         * RandomCrop with padding
         * ColorJitter
         * RandAugment or AutoAugment
    """
    print(configs_info)
    
    return configs_info


def run_config_comparison(num_epochs=5):
    """
    Run a quick comparison of different configurations.
    Note: This is a simplified comparison; full training takes longer.
    """
    print("\n" + "=" * 80)
    print("Quick Configuration Comparison (reduced training for demonstration)")
    print("=" * 80)
    
    configs = [
        {"name": "Baseline", "embed_dim": 128, "num_heads": 4, "num_layers": 4, "patch_size": (4, 4)},
        {"name": "Larger embed", "embed_dim": 256, "num_heads": 8, "num_layers": 4, "patch_size": (4, 4)},
        {"name": "More layers", "embed_dim": 128, "num_heads": 4, "num_layers": 6, "patch_size": (4, 4)},
        {"name": "Larger patches", "embed_dim": 128, "num_heads": 4, "num_layers": 4, "patch_size": (8, 8)},
    ]
    
    device = get_device()
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        # Count parameters
        model = ViT(
            image_size=(32, 32),
            patch_size=config['patch_size'],
            channels=3,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            pos_enc='learnable',
            pool='cls',
            dropout=0.3,
            num_classes=2
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        results.append({
            "config": config['name'],
            "params": num_params,
            "embed_dim": config['embed_dim'],
            "num_heads": config['num_heads'],
            "num_layers": config['num_layers'],
            "patch_size": config['patch_size']
        })
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("Configuration Comparison Summary")
    print("=" * 80)
    print(f"{'Config':<20} {'Params':>12} {'Embed':>8} {'Heads':>8} {'Layers':>8} {'Patch':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['config']:<20} {r['params']:>12,} {r['embed_dim']:>8} {r['num_heads']:>8} {r['num_layers']:>8} {str(r['patch_size']):>10}")
    
    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Prepare data
    classes = [3, 7]  # cat, horse
    class_names = ['cat', 'horse']
    _, test_loader, _, test_dataset = prepare_dataloaders(batch_size=8, classes=classes)
    
    # Get sample images
    sample_images = torch.stack([test_dataset[i][0] for i in range(8)])
    
    # Load or create model
    model = load_or_create_model()
    
    print("\n" + "=" * 80)
    print("TASK 4: Attention Visualization")
    print("=" * 80)
    
    # Visualize attention maps (Task 4)
    print("\n1. CLS Token Attention Across Layers:")
    visualize_attention_maps(model, sample_images, class_names=class_names)
    
    print("\n2. Attention Heads in First Layer:")
    visualize_attention_heads(model, sample_images[:1], layer_idx=0)
    
    print("\n3. Attention Heads in Last Layer:")
    visualize_attention_heads(model, sample_images[:1], layer_idx=-1)
    
    print("\n4. Full Attention Matrix:")
    visualize_full_attention_matrix(model, sample_images[:1], layer_idx=-1)
    
    # Analyze configurations (Task 3)
    # analyze_model_configurations()
    # run_config_comparison()
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - attention_maps_cls_token.png")
    print("  - attention_heads_layer1.png")
    print("  - attention_heads_layer4.png (or last layer)")
    print("  - full_attention_matrix.png")
