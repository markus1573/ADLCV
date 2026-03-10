"""
Shared pipeline initialization for all exercise scripts.
Imports from here to avoid code duplication across files.
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load Stable Diffusion v1-5 pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Replace the default scheduler with a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# VAE latent scaling factor (standard for SD v1.x)
vae_scale_factor = 0.18215
