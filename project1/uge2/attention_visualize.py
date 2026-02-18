import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from imageclassification import prepare_dataloaders, set_seed
from vit import ViT


def build_model(
    checkpoint_path,
    image_size=(32, 32),
    patch_size=(4, 4),
    channels=3,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    num_classes=2,
    pos_enc="learnable",
    pool="cls",
    dropout=0.3,
    fc_dim=None,
    device="cpu",
):
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
        fc_dim=fc_dim,
        num_classes=num_classes,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def register_attention_hooks(model):
    attention_store = {}
    handles = []

    for layer_idx, block in enumerate(model.transformer_blocks):

        def hook_factory(idx):
            def hook(module, inputs, output):
                x = inputs[0]
                with torch.no_grad():
                    keys = module.k_projection(x)
                    queries = module.q_projection(x)

                    keys = rearrange(
                        keys,
                        "b s (h d) -> b h s d",
                        h=module.num_heads,
                        d=module.head_dim,
                    )
                    queries = rearrange(
                        queries,
                        "b s (h d) -> b h s d",
                        h=module.num_heads,
                        d=module.head_dim,
                    )

                    attention_logits = torch.matmul(queries, keys.transpose(-2, -1))
                    attention_logits = attention_logits * module.scale
                    attention = F.softmax(attention_logits, dim=-1)
                    attention_store[idx] = attention.detach().cpu()

            return hook

        handles.append(block.attention.register_forward_hook(hook_factory(layer_idx)))

    return attention_store, handles


def denormalize_image(image):
    image = image * 0.5 + 0.5
    return image.clamp(0.0, 1.0)


def attention_to_map(attention, image_size, patch_size, pool="cls"):
    height, width = image_size
    patch_h, patch_w = patch_size
    nph, npw = height // patch_h, width // patch_w
    num_patches = nph * npw

    attention = attention.mean(dim=1)
    seq_len = attention.size(-1)

    if pool == "cls" and seq_len == num_patches + 1:
        patch_attention = attention[:, 0, 1:]
    else:
        patch_attention = attention.mean(dim=1)

    patch_attention = patch_attention.reshape(-1, nph, npw)
    patch_attention = patch_attention.unsqueeze(1)
    upsampled = F.interpolate(
        patch_attention, size=(height, width), mode="bilinear", align_corners=False
    )
    upsampled = upsampled.squeeze(1)

    attn_min = upsampled.amin(dim=(-2, -1), keepdim=True)
    attn_max = upsampled.amax(dim=(-2, -1), keepdim=True)
    normalized = (upsampled - attn_min) / (attn_max - attn_min + 1e-6)
    return normalized


def plot_attention_grid(images, attention_maps, output_dir, prefix="attn"):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize_image(images).cpu()

    for idx, (image, attn) in enumerate(zip(images, attention_maps)):
        image_np = image.permute(1, 2, 0).numpy()
        attn_np = attn.numpy()

        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].imshow(image_np)
        ax[0].axis("off")
        ax[0].set_title("Image")

        ax[1].imshow(image_np)
        alpha_map = attn_np
        ax[1].imshow(np.zeros_like(image_np), cmap="gray", alpha=alpha_map)
        ax[1].axis("off")
        ax[1].set_title("Attention")

        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{idx:02d}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ViT attention on training images"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="attention_viz")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint is not None else root_dir / "model.pth"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    train_loader, _, _, _ = prepare_dataloaders(batch_size=args.batch_size)
    images, _ = next(iter(train_loader))
    images = images.to(device)

    attention_store, handles = register_attention_hooks(model)
    with torch.no_grad():
        _ = model(images)

    for handle in handles:
        handle.remove()

    num_layers = len(model.transformer_blocks)
    layer_idx = args.layer if args.layer >= 0 else num_layers - 1
    if layer_idx not in attention_store:
        raise ValueError("Requested layer index not captured")

    attention = attention_store[layer_idx]
    attention_map = attention_to_map(
        attention=attention,
        image_size=(32, 32),
        patch_size=(4, 4),
        pool=model.pool,
    )

    output_dir = root_dir / args.output_dir
    plot_attention_grid(images, attention_map, output_dir)
    print(f"Saved attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
