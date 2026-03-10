"""
Task: Implement the DDIM inversion update step.

DDIM inversion reverses the sampling process: starting from a clean image latent
(t = 0) and moving toward high noise (t = T). At each step we *undo* one denoising
step by rearranging the DDIM equation:

Sampling (t → t-1):
    x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(1 - α_t) * ε_θ) / sqrt(α_t)
             + sqrt(1 - α_{t-1}) * ε_θ

Inversion (t → t+1), same formula but with α_{t+1} instead of α_{t-1}:
    x_{t+1} = sqrt(α_{t+1}) * (x_t - sqrt(1 - α_t) * ε_θ) / sqrt(α_t)
             + sqrt(1 - α_{t+1}) * ε_θ

Your job is to fill in the TODO below.
"""

import torch
from tqdm.auto import tqdm
from pipeline_setup import pipe, device, vae_scale_factor


@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
    """
    Invert a latent back through the diffusion process (t=0 → t=T).

    Parameters
    ----------
    start_latents : Tensor
        Clean image latents (output of VAE encoder * vae_scale_factor).
        Shape: (1, 4, 64, 64).
    prompt : str
        Text description of the input image (used for CFG during inversion).
    guidance_scale : float
        Classifier-free guidance scale. Use 1.0 for exact inversion (no CFG error).
    num_inference_steps : int
        Number of inversion steps (more steps → more accurate inversion).

    Returns
    -------
    Tensor
        Stacked intermediate latents, shape (num_inference_steps - 2, 1, 4, 64, 64).
        The last element [-1] is the most-noisy latent (≈ z_T).
    """

    # Encode the text prompt into embeddings
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    latents = start_latents.clone()
    intermediate_latents = []

    # Set the scheduler; inversion walks timesteps in *reverse* order (0 → T)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # Skip the final iteration to avoid going out of bounds
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Duplicate latents for CFG
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict noise with the UNet
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Apply classifier-free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # α for the *current* timestep (t-1 in standard notation, but we're going forward)
        current_t   = max(0, t.item() - (1000 // num_inference_steps))
        next_t      = t  # this is t+1 in the inversion direction
        alpha_t      = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # ── TODO 2 ────────────────────────────────────────────────────────────
        # Implement the inversion update step.
        # It is the same formula as DDIM sampling (see ddim_sampling.py TODO 1),
        # but replace alpha_t_prev with alpha_t_next — because we're moving
        # *forward* in noise (t → t+1) instead of backward (t → t-1).
        #
        # Useful variables:
        #   latents      : x_t,         shape (1, 4, 64, 64)
        #   noise_pred   : ε_θ(x_t),   shape (1, 4, 64, 64)
        #   alpha_t      : α_t,         scalar tensor
        #   alpha_t_next : α_{t+1},     scalar tensor
        #
        # latents = ...
        # ─────────────────────────────────────────────────────────────────────

        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms as tfms
    from diffusers.utils import load_image
    from ddim_sampling import sample

    NUM_STEPS  = 50
    START_STEP = 0   # 0 = full re-sampling from z_T (strictest reconstruction test)

    # Load a sample image and encode it to a latent
    input_image = load_image(
        "https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg"
    ).resize((512, 512))
    input_image_prompt = "Photograph of a puppy on the grass"

    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
    l = vae_scale_factor * latent.latent_dist.sample()

    # Run inversion to get the full noisy latent trajectory
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=NUM_STEPS)
    print(f"Inverted latents shape: {inverted_latents.shape}")

    # Decode the most-noisy latent to see what pure noise looks like
    with torch.no_grad():
        noisy_decoded = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
    pipe.numpy_to_pil(noisy_decoded)[0].save("ddim_inverted_noisy.png")
    print("Saved ddim_inverted_noisy.png")

    # Reconstruct by sampling from the most-noisy inverted latent
    reconstructed = sample(
        input_image_prompt,
        start_latents=inverted_latents[-(START_STEP + 1)][None],
        start_step=START_STEP,
        num_inference_steps=NUM_STEPS,
        guidance_scale=3.5,
    )

    # Save original and reconstruction side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_image);        axes[0].set_title("Original");        axes[0].axis("off")
    axes[1].imshow(reconstructed[0]);   axes[1].set_title("DDIM Reconstruction"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("ddim_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ddim_reconstruction.png")
