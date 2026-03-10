"""
Task: Implement the DDIM sampling update step.

DDIM uses a deterministic update rule (no extra noise, σ_t = 0):

    x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(1 - α_t) * ε_θ(x_t)) / sqrt(α_t)
              + sqrt(1 - α_{t-1}) * ε_θ(x_t)

Where:
  x_t           : noisy latent at step t
  ε_θ(x_t)     : noise predicted by the UNet (noise_pred)
  α_t           : cumulative product of (1 - β_t), accessed via alphas_cumprod
  α_{t-1}       : same for the *previous* (cleaner) timestep

Your job is to fill in the TODO below.

References:
    - Denoising Diffusion Implicit Models (DDIM) (Song et al., 2020): https://arxiv.org/abs/2010.02502
"""

import torch
from tqdm.auto import tqdm
from pipeline_setup import pipe, device, vae_scale_factor


@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           null_texts=None,
           negative_prompt="", device=device):
    """
    Run DDIM sampling (or continue sampling from start_step with given latents).

    Parameters
    ----------
    prompt : str
        Text prompt describing the target image.
    start_step : int
        Index into the timestep schedule to start from (0 = pure noise).
    start_latents : Tensor or None
        Pre-computed latents to start from. Random noise if None.
    guidance_scale : float
        Classifier-free guidance scale (higher → stronger prompt adherence).
    num_inference_steps : int
        Total number of DDIM denoising steps.
    null_texts : list of Tensor or None
        Per-step optimized null-text embeddings from Null-Text Inversion.
        When provided, replaces the generic unconditional embedding at each step.
    negative_prompt : str
        Text to suppress in the output (used as unconditional prompt).

    Returns
    -------
    list of PIL.Image
    """

    # Encode the text prompt (and negative prompt for CFG) into embeddings
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Tell the scheduler how many steps we want
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # If no starting latents are given, start from pure Gaussian noise
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Duplicate latents for CFG (unconditional + conditional pass)
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Null-Text Inversion: swap the unconditional embedding per step
        # null_texts[i] was optimized during null_text_inversion() at this step
        if do_classifier_free_guidance and null_texts is not None and i < len(null_texts):
            null_emb = null_texts[i].to(device)           # shape: (1, 77, 768)
            step_embeddings = torch.cat([null_emb, text_embeddings[1:]])
        else:
            step_embeddings = text_embeddings

        # Run the UNet to predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=step_embeddings).sample

        # Classifier-free guidance: combine unconditional and conditional predictions
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Look up the α values for the current and previous timestep
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t      = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

        # ── TODO 1 ────────────────────────────────────────────────────────────
        # Implement the DDIM update step using the formula at the top of this file.
        # Do NOT use pipe.scheduler.step() — compute it manually.
        #
        # Useful variables:
        #   latents      : x_t,        shape (1, 4, 64, 64)
        #   noise_pred   : ε_θ(x_t),  shape (1, 4, 64, 64)
        #   alpha_t      : α_t,        scalar tensor
        #   alpha_t_prev : α_{t-1},    scalar tensor
        #
        # Hint: first recover the predicted clean image x_0 from x_t and ε,
        #       then project it to x_{t-1}.
        #
        # latents = ...
        # ─────────────────────────────────────────────────────────────────────

            # Recover the predicted clean image x_0 from x_t and ε
        x0_pred = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        # Project x_0 to x_{t-1}
        latents = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred




    # Decode the final latent back to pixel space
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


if __name__ == "__main__":
    from PIL import Image

    prompt = "Watercolor painting of Nyhavn, Copenhagen"
    negative_prompt = "blurry, ugly, stock photo"

    images = sample(prompt, negative_prompt=negative_prompt, num_inference_steps=50)
    images[0].save("ddim_sample_output.png")
    print("Saved ddim_sample_output.png")
