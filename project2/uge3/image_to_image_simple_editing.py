"""
Task: Why not just use img2img instead of DDIM inversion?

A simpler alternative to inversion is to directly add noise to the image latent
and then denoise with a new prompt.  This is the classic img2img approach:

    noisy_latent = scheduler.add_noise(clean_latent, ε, t_start)
    output = sample(new_prompt, start_latents=noisy_latent, start_step=start_step)

Compare the results of this file with ddim_inversion_simple_editing.py to see the
difference in structure preservation.

Your task: implement the TODO below to corrupt the latent with the right amount of
noise, then pass it to the sample() function.
"""

import torch
from torchvision import transforms as tfms
from diffusers.utils import load_image

from pipeline_setup import pipe, device, vae_scale_factor
from ddim_sampling import sample


def img2img_edit(input_image, edit_prompt,
                 num_steps=50, start_step=10, guidance_scale=3.5):
    """
    Edit an image by adding noise directly then denoising with a new prompt.

    Parameters
    ----------
    input_image : PIL.Image
        The source image to edit (should be 512×512).
    edit_prompt : str
        The target description for the edited image.
    num_steps : int
        Total number of diffusion steps for both noise scheduling and sampling.
    start_step : int
        Which timestep index to start denoising from.
        Controls the noise level added to the latent:
          - Low  (e.g. 5)  → subtle edit, background mostly preserved
          - High (e.g. 40) → strong edit, but background changes significantly
    guidance_scale : float
        CFG scale for the sampling pass.

    Returns
    -------
    PIL.Image
        The edited image.
    """
    # Encode the input image to latent space with the VAE
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
    l = vae_scale_factor * latent.latent_dist.sample()

    # Set up the scheduler timestep schedule
    pipe.scheduler.set_timesteps(num_steps)

    # ── TODO ──────────────────────────────────────────────────────────────────
    # Add noise to the clean latent `l` to simulate a partially-noisy image at
    # timestep `pipe.scheduler.timesteps[start_step]`.
    #
    # Use pipe.scheduler.add_noise(), which takes:
    #   original_samples : the clean latent  (l)
    #   noise            : Gaussian noise with the same shape as l
    #   timesteps        : the target timestep as a tensor
    #
    # Hint: sample the noise with torch.randn_like(l).
    # Hint: the target timestep is pipe.scheduler.timesteps[start_step].
    #
    # noisy_l = ...
    # ─────────────────────────────────────────────────────────────────────────

    # Denoise from start_step onward with the edit prompt
    output = sample(
        edit_prompt,
        start_latents=noisy_l,  # noqa: F821  (defined in TODO above)
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    return output[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    input_image = load_image(
        "https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg"
    ).resize((512, 512))

    source_prompt = "Photograph of a puppy on the grass"
    edit_prompt   = source_prompt.replace("puppy", "cat")

    num_steps = 50

    # ── Compare different start_steps ─────────────────────────────────────────
    # Low start_step  → Effect? ...
    # High start_step → Effect? ...
    start_steps = [5, 10, 20, 30]

    fig, axes = plt.subplots(1, len(start_steps) + 1, figsize=((len(start_steps) + 1) * 3, 4))
    axes[0].imshow(input_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, ss in zip(axes[1:], start_steps):
        print(f"Running img2img edit with start_step={ss} ...")
        result = img2img_edit(
            input_image, edit_prompt,
            num_steps=num_steps,
            start_step=ss,
            guidance_scale=3.5,
        )
        ax.imshow(result)
        ax.set_title(f"start_step={ss}")
        ax.axis("off")

    plt.suptitle(f'img2img edit: "{edit_prompt}"', fontsize=10)
    plt.tight_layout()
    plt.savefig("img2img_edit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved img2img_edit_comparison.png")
