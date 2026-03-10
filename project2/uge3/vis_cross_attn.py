"""
Task: Cross-attention visualization.

In Stable Diffusion's UNet, each cross-attention layer computes attention between
spatial feature tokens (pixels) and text tokens (words). By capturing these
attention maps and averaging them across layers and attention heads, we can
visualize *which parts of the image* each word in the prompt attends to.

Attention map shape inside the UNet:
    (batch_size * num_heads, H*W, num_text_tokens)
    where H*W is the spatial resolution of that layer (e.g. 8×8, 16×16, 32×32, 64×64).

Your task: implement the aggregation of attention maps across heads (TODO below).

References:
  - Prompt-to-Prompt Image Editing with Cross Attention Control (Hertz et al., 2022): https://arxiv.org/abs/2208.01626
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

from pipeline_setup import pipe, device, vae_scale_factor


# ── Hook storage ──────────────────────────────────────────────────────────────
# We store attention weights from all cross-attention layers here.
_attention_maps = []   # each entry: Tensor of shape (batch*heads, H*W, seq_len)


def _make_attn_hook(num_heads):
    """
    Returns a forward hook that captures the cross-attention softmax weights.

    The hook is registered on an `nn.Module` whose forward() receives:
        query   : (batch*heads, H*W,       head_dim)
        key     : (batch*heads, seq_len,   head_dim)
    and internally computes attention weights:
        attn_w  : (batch*heads, H*W,       seq_len)  ← what we want to capture

    We register the hook on the `attn_processor` for each attention layer.
    A simpler approach: register a hook on the `to_out` projection and capture
    the attention weights from the processor's last call.

    Here we use a monkey-patch approach: wrap the processor's __call__ method
    so we can intercept the attention weights before they are consumed.
    """
    def hook(module, input, output):
        # `output` from an attention module is the projected value — not the weights.
        # We therefore register the hook on the *softmax* by wrapping the processor
        # (see `register_attention_hooks` below for the actual implementation).
        pass
    return hook


class AttentionCapture:
    """
    Replaces an attention processor to capture cross-attention weights.
    Compatible with diffusers >=0.16.

    Usage: set pipe.unet.attn_processors to a dict of these.
    """
    def __init__(self, num_heads):
        self.num_heads = num_heads

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        """
        Mirrors the default AttnProcessor but saves cross-attention weights.
        Only captures when encoder_hidden_states is provided (cross-attention).
        """
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        context  = encoder_hidden_states if is_cross else hidden_states

        # Project Q, K, V
        query = attn.to_q(hidden_states)
        key   = attn.to_k(context)
        value = attn.to_v(context)

        # Reshape to multi-head form: (batch * heads, seq, head_dim)
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute scaled dot-product attention scores + softmax
        attn_weights = attn.get_attention_scores(query, key, attention_mask)
        # attn_weights: (batch * heads, spatial_tokens, text_tokens)

        # ── Capture cross-attention maps ──────────────────────────────────────
        if is_cross:
            _attention_maps.append(attn_weights.detach().cpu())

        # Compute attended values and project back
        hidden_states = torch.bmm(attn_weights, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_hooks():
    """
    Installs AttentionCapture processors on all UNet attention layers.
    Call this once before running a forward pass.
    """
    new_processors = {}
    for name, module in pipe.unet.attn_processors.items():
        # Determine head count from the parent attention module
        # The processor name ends in '.processor'; strip that to get the module name
        parent_name = name.replace('.processor', '')
        try:
            parent = pipe.unet.get_submodule(parent_name)
            num_heads = parent.heads
        except Exception:
            num_heads = 8  # fallback

        new_processors[name] = AttentionCapture(num_heads)

    pipe.unet.set_attn_processor(new_processors)


def remove_attention_hooks():
    """Restore the default attention processors."""
    from diffusers.models.attention_processor import AttnProcessor
    pipe.unet.set_attn_processor(AttnProcessor())


def aggregate_attention_maps(attention_maps, num_heads, res=16):
    """
    Aggregate a list of raw cross-attention tensors into a single map per token.

    Each element in `attention_maps` has shape:
        (batch * num_heads,  H * W,  num_text_tokens)

    We want to return a single Tensor of shape:
        (num_text_tokens,  res,  res)

    where each channel is the spatial attention map for one text token,
    averaged across all layers and all attention heads, and resized to `res×res`.

    Parameters
    ----------
    attention_maps : list of Tensor
        Raw attention weight tensors captured from all cross-attention layers.
    num_heads : int
        Number of attention heads per layer.
    res : int
        Target spatial resolution for visualization (typically 16 for 16×16 maps).

    Returns
    -------
    Tensor, shape (num_text_tokens, res, res)
    """




    # ── TODO ──────────────────────────────────────────────────────────────────
    # 1. Filter `attention_maps` to keep only layers whose spatial dimension
    #    matches res*res (i.e. attention_maps[i].shape[1] == res*res).
    #
    # 2. For each kept map:
    #    a. Reshape from (batch*heads, H*W, seq) → (batch, heads, H*W, seq).
    #       Hint: assume batch=1, so first dim = num_heads.
    #    b. Average over the heads dimension → (1, H*W, seq).
    #    c. Reshape H*W → (res, res) → (1, seq, res, res).
    #
    # 3. Stack all per-layer maps and average them → (1, seq, res, res).
    #
    # 4. Remove the batch dimension and return → (seq, res, res).
    #
    # avg_map = ...  # shape: (num_text_tokens, res, res)
    # return avg_map
    # ─────────────────────────────────────────────────────────────────────────

    filtered = [m for m in attention_maps if m.shape[1] == res * res]
    if not filtered:
        raise ValueError(f"No attention maps found with spatial dimension {res}x{res}.")

    per_layer = []
    for m in filtered:
        # (num_heads, H*W, seq) → (1, heads, H*W, seq)
        m = m.reshape(1, num_heads, res * res, -1)
        # average over heads → (1, H*W, seq)
        m = m.mean(dim=1)
        # (1, H*W, seq) → (1, seq, res, res)
        m = m.permute(0, 2, 1).reshape(1, -1, res, res)
        per_layer.append(m)

    # Stack and average over layers → (1, seq, res, res)
    avg_map = torch.stack(per_layer, dim=0).mean(dim=0)

    # Remove batch dim → (seq, res, res)
    return avg_map.squeeze(0)


@torch.no_grad()
def visualize_cross_attention(prompt, num_inference_steps=20, guidance_scale=7.5,
                               res=16, save_path="cross_attn_vis.png"):
    """
    Generate an image and visualize the cross-attention map for each token.

    Parameters
    ----------
    prompt : str
        Text prompt to generate and visualize.
    num_inference_steps : int
        Number of denoising steps (fewer = faster but lower quality).
    guidance_scale : float
        CFG scale.
    res : int
        Spatial resolution of attention maps to aggregate (16 → 16×16 maps).
    save_path : str
        Where to save the visualization figure.
    """
    global _attention_maps
    _attention_maps = []

    # Install attention capture processors
    register_attention_hooks()

    # Run the pipeline; attention maps are populated as a side effect
    output = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    image = output.images[0]

    # Restore default processors
    remove_attention_hooks()

    # Tokenize the prompt to get token strings for axis labels
    tokens = pipe.tokenizer.encode(prompt)
    token_strs = [pipe.tokenizer.decode([t]) for t in tokens]

    # Aggregate attention maps
    try:
        num_heads = 8  # SD v1.x uses 8 heads in cross-attention
        agg = aggregate_attention_maps(_attention_maps, num_heads=num_heads, res=res)
    except NotImplementedError:
        print("aggregate_attention_maps not yet implemented — implement the TODO first.")
        return image

    # ── Visualize ─────────────────────────────────────────────────────────────
    num_tokens = min(agg.shape[0], len(token_strs))
    cols = min(num_tokens, 8)
    rows = (num_tokens + cols - 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    # Row 0: generated image
    axes[0].imshow(image)
    axes[0].set_title("Generated", fontsize=8)
    axes[0].axis("off")
    for j in range(1, cols):
        axes[j].axis("off")

    # Remaining rows: per-token attention maps
    for idx in range(num_tokens):
        ax = axes[cols + idx]
        attn_map = agg[idx].numpy()
        # Normalise each map independently for visibility
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        ax.imshow(attn_map, cmap="hot", interpolation="bilinear")
        ax.set_title(token_strs[idx], fontsize=8)
        ax.axis("off")

    for idx in range(num_tokens, len(axes) - cols):
        axes[cols + idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-attention visualization to {save_path}")

    return image


if __name__ == "__main__":
    prompt = "a fluffy cat sitting on a wooden table"
    visualize_cross_attention(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        res=16,
        save_path="cross_attn_vis.png",
    )
