"""
Task (Optional): Cross-attention control for structure-preserving image editing.

This technique, introduced in Prompt-to-Prompt (Hertz et al., 2022), allows
precise editing while preserving the spatial structure of the original image.

Key idea:
  During denoising of the *edited* image, replace the cross-attention maps
  produced by the new prompt with those captured from the *source* image
  denoising.  This injects the layout and structure of the source into the edit.

The injection can be partial: only inject at timesteps t > injection_threshold,
so that early steps fix structure while late steps refine appearance with the
new prompt.

Two inversion modes are supported (controlled by `use_nti`):

  use_nti=True  (default): Null-Text Inversion.
    Optimizes a per-step null-text embedding so that CFG-guided sampling
    exactly reconstructs the source image.  The optimized null texts are used
    as the unconditional embedding for BOTH the source and target passes at
    each step, which further anchors the structure of the source image.

  use_nti=False: Plain DDIM inversion.
    Faster but less accurate; structure preservation relies entirely on the
    cross-attention injection.

Your task: implement the attention-map injection logic in the TODO below.

Reference: Prompt-to-Prompt Image Editing with Cross Attention Control (https://arxiv.org/abs/2208.01626)
"""

import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

from pipeline_setup import pipe, device, vae_scale_factor
from ddim_inversion import invert
from null_text_inversion import null_text_inversion


# ── Shared state for attention injection ──────────────────────────────────────
_current_step: int = 0                # which denoising step we are on
_injection_threshold: float = 0.5    # fraction of steps to inject


class AttentionStore:
    """
    Stores cross-attention maps keyed by layer name.

    Using layer names (instead of a positional index) ensures that each
    processor always stores/injects the map for its own specific layer,
    regardless of the order in which layers are executed.
    """
    def __init__(self):
        self.maps: Dict[str, torch.Tensor] = {}   # layer_name → attention_weights

    def reset(self):
        self.maps = {}


class CrossAttentionController:
    """
    Attention processor that either *stores* or *injects* cross-attention maps.

    Mode 'store' : runs normally and saves attention weights under self.layer_name.
    Mode 'inject': replaces its own attention weights with the stored source map.
    """

    def __init__(self, store: AttentionStore, mode: str, layer_name: str):
        """
        Parameters
        ----------
        store : AttentionStore
            Shared storage between the source (store) and target (inject) processors.
        mode : str
            'store'  — record attention weights (used for source image pass).
            'inject' — replace attention weights with stored ones (for edited pass).
        layer_name : str
            The UNet attention layer name (e.g. 'down_blocks.1.attentions.0…').
            Used as the key for storing and looking up maps.
        """
        assert mode in ("store", "inject"), "mode must be 'store' or 'inject'"
        self.store      = store
        self.mode       = mode
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        """
        Forward pass through an attention layer with optional store/inject.

        Both cross-attention (text→spatial) and self-attention (spatial→spatial)
        are stored and injected.  Self-attention carries the spatial layout and
        pose of the source image — injecting it is the dominant factor for
        structure preservation (equivalent to self_replace_steps in PTP).
        """
        is_cross = encoder_hidden_states is not None
        context  = encoder_hidden_states if is_cross else hidden_states

        # Project Q, K, V
        query = attn.to_q(hidden_states)
        key   = attn.to_k(context)
        value = attn.to_v(context)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention weights (softmax)
        attn_weights = attn.get_attention_scores(query, key, attention_mask)
        # attn_weights: (batch * heads, spatial_tokens, seq_tokens)

        # Store or inject attention maps for BOTH cross- and self-attention.
        # Cross-attention (is_cross=True) : controls what content appears where.
        # Self-attention  (is_cross=False): encodes spatial layout and structure.
        if self.mode == "store":
            # ── Store mode: save this layer's attention weights by name ───────
            self.store.maps[self.layer_name] = attn_weights.detach().cpu()

        elif self.mode == "inject":
            # ── TODO ──────────────────────────────────────────────────────────
            # Inject the source attention maps into the target pass.
            #
            # Context:
            #   self.store.maps      : dict mapping layer_name → stored tensor.
            #                         The source pass stored one entry per layer.
            #   self.layer_name      : name of the attention layer this processor
            #                         handles — use it to look up the right map.
            #   _current_step        : global int, which denoising step we are on.
            #   _injection_threshold : fraction of total steps to inject.
            #                         e.g. 0.5 → inject for the first 50% of steps.
            #   pipe.scheduler.num_inference_steps : total number of steps.
            #
            # Your task (~4 lines):
            # 1. Check whether we should inject at this step:
            #      _current_step < _injection_threshold * total_steps
            # 2. If yes, AND this layer's map is stored, replace attn_weights
            #    with self.store.maps[self.layer_name] (move it to the right device).
            #
            # --- write your ~4 lines here ---

            pass
            # ──────────────────────────────────────────────────────────────────

        # Compute attended values and project
        hidden_states = torch.bmm(attn_weights, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def _set_processors(store: AttentionStore, mode: str):
    """Install CrossAttentionController processors on all UNet attention layers."""
    processors = {}
    for name in pipe.unet.attn_processors.keys():
        processors[name] = CrossAttentionController(store, mode=mode, layer_name=name)
    pipe.unet.set_attn_processor(processors)


def _restore_default_processors():
    """Restore the default (no-op) attention processors."""
    from diffusers.models.attention_processor import AttnProcessor
    pipe.unet.set_attn_processor(AttnProcessor())


def edit_with_cross_attn_control(
    input_image,
    source_prompt: str,
    target_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    injection_threshold: float = 0.5,
    use_nti: bool = True,
    nti_num_opt_steps: int = 10,
    nti_lr: float = 1e-2,
):
    """
    Edit an image using Prompt-to-Prompt cross-attention control.

    Parameters
    ----------
    input_image : PIL.Image
        Source image (512×512).
    source_prompt : str
        Text description of the source image.
    target_prompt : str
        Text description of the desired edit.
    num_inference_steps : int
        Number of DDIM steps.
    guidance_scale : float
        CFG guidance scale.
    injection_threshold : float
        Fraction of denoising steps (from the beginning / noisy end) during
        which source attention maps are injected.  0 = no injection, 1 = always.
    use_nti : bool
        If True (default), use Null-Text Inversion for more accurate reconstruction.
        The optimized null-text embeddings are used as the unconditional embedding
        for both source and target passes, further anchoring the source structure.
        If False, use plain DDIM inversion (faster, less accurate).
    nti_num_opt_steps : int
        Gradient steps per timestep during NTI optimization (only used if use_nti=True).
    nti_lr : float
        Learning rate for NTI optimization (only used if use_nti=True).

    Returns
    -------
    Tuple[PIL.Image, PIL.Image]
        (source_reconstruction, edited_image)
    """
    global _current_step, _injection_threshold
    _injection_threshold = injection_threshold

    from torchvision import transforms as tfms

    # ── Step 1: Encode input image to latent space ─────────────────────────────
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
        l = vae_scale_factor * latent.latent_dist.sample()

    # ── Step 2: Inversion ──────────────────────────────────────────────────────
    # Encode source prompt — we always need the conditional embedding.
    # _encode_prompt returns [uncond; cond] when CFG is enabled.
    with torch.no_grad():
        both_src = pipe._encode_prompt(source_prompt, device, 1, True, "")
    uncond_emb   = both_src[:1]   # (1, 77, 768) — generic null-text
    src_cond_emb = both_src[1:]   # (1, 77, 768) — source conditional

    with torch.no_grad():
        both_tgt = pipe._encode_prompt(target_prompt, device, 1, True, "")
    tgt_cond_emb = both_tgt[1:]   # (1, 77, 768) — target conditional

    all_null_texts: Optional[List[torch.Tensor]] = None

    if use_nti:
        # Null-Text Inversion: optimizes a per-step uncond embedding so that
        # CFG-guided sampling exactly reconstructs the source image.
        # Returns z_T (noisy start latent) and the list of optimized null texts.
        print("Running Null-Text Inversion…")
        z_T, all_null_texts = null_text_inversion(
            l, source_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_opt_steps=nti_num_opt_steps,
            lr=nti_lr,
            device=device,
        )
        start_latent = z_T.to(device)
    else:
        # Plain DDIM inversion: deterministic but less accurate at high guidance.
        print("Running DDIM inversion…")
        inverted_latents = invert(
            l, source_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        start_latent = inverted_latents[-1][None]

    # ── Step 3: Interleaved source + target denoising with attention control ───
    # No gradients needed from here on — NTI is already done above.
    store = AttentionStore()
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    latents_src = start_latent.clone().to(device)
    latents_tgt = start_latent.clone().to(device)

    with torch.no_grad():
        for step_idx, t in enumerate(pipe.scheduler.timesteps):
            _current_step = step_idx
            store.reset()

            # Select the unconditional embedding for this step:
            #   NTI  → use the per-step optimized null text (anchors source structure)
            #   DDIM → use the generic empty-string uncond embedding
            if all_null_texts is not None and step_idx < len(all_null_texts):
                null_emb = all_null_texts[step_idx].to(device)
            else:
                null_emb = uncond_emb

            # Build full CFG embeddings [uncond; cond] for each prompt
            src_emb = torch.cat([null_emb, src_cond_emb])   # (2, 77, 768)
            tgt_emb = torch.cat([null_emb, tgt_cond_emb])   # (2, 77, 768)

            # Source forward pass → stores attention maps keyed by layer name
            # Duplicate latents along batch dim to match the CFG text embeddings (batch=2)
            _set_processors(store, mode="store")
            inp_src = pipe.scheduler.scale_model_input(torch.cat([latents_src] * 2), t)
            noise_both = pipe.unet(inp_src, t, encoder_hidden_states=src_emb).sample
            noise_pred_uncond_src, noise_pred_text_src = noise_both.chunk(2)
            noise_src = noise_pred_uncond_src + guidance_scale * (noise_pred_text_src - noise_pred_uncond_src)
            latents_src = pipe.scheduler.step(noise_src, t, latents_src).prev_sample

            # Target forward pass → injects stored attention maps (same layer name lookup)
            # Same duplication for the target pass
            _set_processors(store, mode="inject")
            inp_tgt = pipe.scheduler.scale_model_input(torch.cat([latents_tgt] * 2), t)
            noise_both = pipe.unet(inp_tgt, t, encoder_hidden_states=tgt_emb).sample
            noise_pred_uncond_tgt, noise_pred_text_tgt = noise_both.chunk(2)
            noise_tgt = noise_pred_uncond_tgt + guidance_scale * (noise_pred_text_tgt - noise_pred_uncond_tgt)
            latents_tgt = pipe.scheduler.step(noise_tgt, t, latents_tgt).prev_sample

    _restore_default_processors()

    # Decode both latents — no_grad needed since VAE params have requires_grad=True
    with torch.no_grad():
        src_img = pipe.numpy_to_pil(pipe.decode_latents(latents_src))[0]
        tgt_img = pipe.numpy_to_pil(pipe.decode_latents(latents_tgt))[0]

    return src_img, tgt_img


if __name__ == "__main__":
    from diffusers.utils import load_image

    input_image = load_image(
        "https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg"
    ).resize((512, 512))

    # ── NTI + cross-attention control (default, best quality) ─────────────────
    src_img, edited_img = edit_with_cross_attn_control(
        input_image,
        source_prompt="A photograph of a puppy on the grass",
        target_prompt="A photograph of a cat on the grass",
        num_inference_steps=50,
        guidance_scale=7.5,
        injection_threshold=0.5,
        use_nti=True,          # set to False to use plain DDIM inversion instead
        nti_num_opt_steps=10,
        nti_lr=1e-2,
    )

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_image);  axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(src_img);      axes[1].set_title("Reconstructed"); axes[1].axis("off")
    axes[2].imshow(edited_img);   axes[2].set_title("Edited");        axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("cross_attn_edit_result.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cross_attn_edit_result.png")
