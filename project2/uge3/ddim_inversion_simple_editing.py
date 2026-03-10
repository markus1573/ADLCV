"""
Task: Image editing via DDIM inversion.

This script demonstrates the full DDIM-based editing pipeline:
  1. Encode the input image to latent space.
  2. Run DDIM inversion to obtain intermediate noisy latents.
  3. Start denoising from an intermediate step with a *new* prompt.

The key insight: by starting denoising from an intermediate inverted latent,
the model preserves the structure of the original image while applying the
new prompt.  The `start_step` parameter controls the trade-off:
  - Low start_step  → more faithful to original structure (less edit freedom)
  - High start_step → more aggressive editing (less structure preserved)

Try the script with your own group photo and edit prompt!
"""

import torch
from torchvision import transforms as tfms
from diffusers.utils import load_image
from PIL import Image

from pipeline_setup import pipe, device, vae_scale_factor
from ddim_sampling import sample
from ddim_inversion import invert


def edit(input_image, input_image_prompt, edit_prompt,
         num_steps=100, start_step=30, guidance_scale=3.5):
    """
    Edit an image by DDIM inversion + re-sampling with a new prompt.

    Parameters
    ----------
    input_image : PIL.Image
        The source image to edit (should be 512×512).
    input_image_prompt : str
        A text description of the *source* image (used during inversion).
    edit_prompt : str
        The target description for the edited image.
    num_steps : int
        Number of DDIM inversion/sampling steps. More steps → more accurate.
    start_step : int
        Which inverted latent to start sampling from.
        Larger values allow more structural change.
    guidance_scale : float
        CFG scale for the sampling pass.

    Returns
    -------
    PIL.Image
        The edited image.
    """
    # Encode the input image to latent space
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
    l = vae_scale_factor * latent.latent_dist.sample()

    # Run DDIM inversion to get the trajectory of noisy latents
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)

    # Sample (denoise) from the intermediate inverted latent with the new prompt
    final_im = sample(
        edit_prompt,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )[0]

    return final_im


if __name__ == "__main__":
    # ── Example 1: Puppy → cat ─────────────────────────────────────────────────
    input_image = load_image(
        "https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg"
    ).resize((512, 512))

    print("Running edit: puppy → cat")
    result = edit(
        input_image,
        input_image_prompt="A puppy on the grass",
        edit_prompt="A cat on the grass",
        num_steps=50,
        start_step=10,
        guidance_scale=3.5,
    )
    result.save("edit_puppy_to_cat.png")
    input_image.save("edit_original.png")
    print("Saved edit_puppy_to_cat.png and edit_original.png")

    # ── Example 2: Group photo with sunglasses ─────────────────────────────────
    # Replace the URL below with a path to your own group photo:
    #   group_image = Image.open("your_group_photo.jpg").resize((512, 512))
    group_image = load_image(
        "https://yourURL.com/group_photo.jpg"
    ).resize((512, 512))

    print("\nRunning edit: group → group with sunglasses")
    result_group = edit(
        group_image,
        input_image_prompt="A group of three men",
        edit_prompt="A group of three men with sunglasses",
        num_steps=350,
        start_step=25,
        guidance_scale=5.5,
    )
    result_group.save("edit_group_sunglasses.png")
    group_image.save("edit_group_original.png")
    print("Saved edit_group_sunglasses.png and edit_group_original.png")

    # ── Hyperparameter exploration ─────────────────────────────────────────────
    # Try varying these parameters and observe their effect:
    #
    # num_steps:       More steps → more accurate inversion but slower.
    #                  Range: 50–500
    #
    # start_step:      Higher → more of the image is re-generated → bigger edit.
    #                  Must be < num_steps.  Typical range: 5–50.
    #
    # guidance_scale:  Higher → output more strongly follows the edit prompt.
    #                  Too high can cause artifacts.  Typical range: 3–10.
