"""
Task: Implement Null-Text Inversion (NTI).

Null-Text Inversion (Mokady et al., 2022) improves upon DDIM inversion by
optimizing a *per-timestep* unconditional (null-text) embedding so that the
classifier-free-guidance (CFG) backward trajectory exactly matches the DDIM
forward pivot trajectory.

Algorithm overview:
  1. Run DDIM inversion at guidance_scale=1 to obtain a pivot latent trajectory.
  2. For each timestep t (large → small), optimize the null-text embedding ∅_t
     so that the CFG-guided DDIM step matches the pivot:

         min_{∅_t}  || scheduler.step(ε_CFG, t, z_t) - z*_{t-1} ||²

     where z*_{t-1} is the pivot latent for the previous step.
  3. Return z_T and the list of optimized embeddings for use in sample().

Your job: fill in the three TODOs inside the optimization loop.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pipeline_setup import pipe, device, vae_scale_factor
import matplotlib.pyplot as plt

def null_text_inversion(start_latents, prompt, guidance_scale=7.5,
                        num_inference_steps=50, num_opt_steps=10, lr=1e-2,
                        device=device):
    """
    Optimize null-text embeddings so that NTI reconstructs the input exactly.

    Parameters
    ----------
    start_latents : Tensor
        Clean image latent, shape (1, 4, 64, 64).
    prompt : str
        Text description of the input image.
    guidance_scale : float
        CFG scale used during both optimization and later sampling.
    num_inference_steps : int
        Number of diffusion steps.
    num_opt_steps : int
        Gradient steps per timestep (more → better reconstruction, slower).
    lr : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    z_T : Tensor
        Most-noisy pivot latent to pass as start_latents to sample().
    all_null_texts : list of Tensor
        Per-timestep optimized null-text embeddings (CPU tensors).
        all_null_texts[i] aligns with sampling step i — no reversal needed.
    """

    # ── Step 1: Encode text prompt ─────────────────────────────────────────────
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    with torch.no_grad():
        # _encode_prompt returns [uncond_emb; cond_emb] when CFG is enabled
        both_emb   = pipe._encode_prompt(prompt, device, 1, True, '')  # (2, 77, 768)
    uncond_emb = both_emb[:1].detach()   # empty-string (null-text) embedding
    text_emb   = both_emb[1:].detach()  # conditional prompt embedding

    # ── Step 2: Build the pivot trajectory via DDIM inversion at scale=1 ───────
    # At guidance_scale=1, CFG collapses to a single prediction → no approximation.
    # We walk small t → large t, saving each intermediate latent.
    latents     = start_latents.clone()
    all_latents = [latents.detach().cpu()]   # index 0 = z_0 (clean image latent)

    timesteps_fwd = list(reversed(pipe.scheduler.timesteps))   # ascending noise

    for i in range(len(timesteps_fwd) - 1):
        t      = timesteps_fwd[i]
        t_next = timesteps_fwd[i + 1]   # next noisier timestep

        with torch.no_grad():
            inp        = pipe.scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=text_emb).sample

        # DDIM forward step: x_t → x_{t+1}
        alpha_t      = pipe.scheduler.alphas_cumprod[t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[t_next]
        pred_x0  = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        latents  = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * noise_pred

        all_latents.append(latents.detach().cpu())

    # all_latents = [z_0, z_1, …, z_T],  length = num_inference_steps

    # ── Step 3: Initialize the null-text embedding (warm-started across steps) ─
    # We optimize a single embedding and carry it forward across timesteps,
    # which is faster than re-initializing from the generic uncond_emb each time.
    null_text_emb = uncond_emb.clone().detach().requires_grad_(True)
    

    # ── Step 4: Online optimization ────────────────────────────────────────────
    # For each sampling step i (large t → small t):
    #   - The current running latent is z_{T-i}  (from the pivot trajectory)
    #   - The optimization target (pivot) is z_{T-i-1}
    all_null_texts = []
    latents = all_latents[-1].to(device)   # start from z_T (most noisy)

    print("Optimizing null-text embeddings (online)…")
    # zip pairs each scheduler timestep with its corresponding pivot target
    pairs = list(zip(pipe.scheduler.timesteps, reversed(all_latents[:-1])))

    for t, pivot in tqdm(pairs):
        pivot = pivot.to(device)
        optimizer = torch.optim.Adam([null_text_emb], lr=lr)
        # Conditional prediction is fixed at this timestep — compute once (no grad)
        latent_input = pipe.scheduler.scale_model_input(latents.detach(), t)
        with torch.no_grad():
            noise_pred_text = pipe.unet(
                latent_input, t, encoder_hidden_states=text_emb
            ).sample.detach()

        for _ in range(num_opt_steps):

            # Unconditional prediction uses the *optimizable* null_text_emb
            noise_pred_uncond = pipe.unet(
                latent_input, t, encoder_hidden_states=null_text_emb
            ).sample

            # ── TODO 3 ──────────────────────────────────────────────────────────
            # Compute the CFG-combined noise prediction.
            # Recall: ε_CFG = ε_uncond + guidance_scale * (ε_text - ε_uncond)
            # Only noise_pred_uncond carries gradients (noise_pred_text is detached).
            #
            # noise_pred = ...
            # ────────────────────────────────────────────────────────────────────
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # ── TODO 4 ──────────────────────────────────────────────────────────
            # Use the scheduler to predict the next (less noisy) latent.
            # This is a single DDIM backward step from `latents` using `noise_pred`.
            # Hint: pipe.scheduler.step() returns an object; use .prev_sample.
            #
            # z_pred = ...
            # ────────────────────────────────────────────────────────────────────
            z_pred = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            # ── TODO 5 ──────────────────────────────────────────────────────────
            # Compute the MSE loss between z_pred and the pivot target,
            # then run one optimization step.
            # Steps: compute loss → zero_grad → backward → optimizer step
            #
            # loss = ...
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # ────────────────────────────────────────────────────────────────────
            loss = F.mse_loss(z_pred, pivot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save this timestep's optimized null-text and advance the running latent
        all_null_texts.append(null_text_emb.detach().cpu())
        latents = z_pred.detach()  # noqa: F821  (defined in TODO 4)

    # z_T is all_latents[-1]; all_null_texts[i] aligns with sampling step i
    return all_latents[-1], all_null_texts


if __name__ == "__main__":
    from torchvision import transforms as tfms
    from diffusers.utils import load_image
    from ddim_sampling import sample
    from skimage.metrics import structural_similarity as ssim
    from lpips import LPIPS

    # Load and encode the input image
    # input_image = load_image(
    #     "https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg"
    # ).resize((512, 512))
    # input_image_prompt = "Photograph of a puppy on the grass"
    image_folder = "images"
    txt_file = image_folder + "/input.txt"
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            image_path, input_image_prompt = line.strip().split(",")
            input_image = load_image(image_folder + "/" + image_path).resize((512, 512))
            print(f"Loaded image: {image_path} with prompt: {input_image_prompt}")

                    
            with torch.no_grad():
                latent = pipe.vae.encode(
                    tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
                )
            l = vae_scale_factor * latent.latent_dist.sample()

            NUM_STEPS = 50

            # Run Null-Text Inversion
            z_T, all_null_texts = null_text_inversion(
                l, input_image_prompt,
                guidance_scale=7.5,
                num_inference_steps=NUM_STEPS,
                num_opt_steps=10,
                lr=1e-2,
            )
            print(f"z_T shape: {z_T.shape}, num null texts: {len(all_null_texts)}")

            # Reconstruct the image using the optimized null texts
            reconstructed = sample(
                input_image_prompt,
                start_latents=z_T.to(device),
                guidance_scale=7.5,
                num_inference_steps=NUM_STEPS,
                null_texts=all_null_texts,
            )
            # reconstructed[0].save("nti_reconstruction.png")
            # input_image.save("nti_original.png")
            # print("Saved nti_reconstruction.png and nti_original.png")

            # Compute PSNR between the original and reconstructed image
            original = tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)
            reconstructed_tensor = tfms.functional.to_tensor(reconstructed[0]).unsqueeze(0).to(device)
            mse = F.mse_loss(reconstructed_tensor, original)
            psnr = 10 * torch.log10(1 / mse)

            # Compute SSIM between the original and reconstructed image
            original_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
            reconstructed_np = reconstructed_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            ssim_score = ssim(original_np, reconstructed_np, channel_axis=-1, data_range=1)

            # Compute LPIPS between the original and reconstructed image
            lpips_metric = LPIPS(net='alex').to(device)
            lpips_score = lpips_metric(reconstructed_tensor, original).item()



                        # Save original and reconstruction side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(input_image);        axes[0].set_title("Original");        axes[0].axis("off")
            axes[1].imshow(reconstructed[0]);   axes[1].set_title("Null-Text Reconstruction"); axes[1].axis("off")
            # add metrics as text at the bottom
            plt.figtext(0.5, 0.01, f"PSNR: {psnr:.2f} dB | SSIM: {ssim_score:.4f} | LPIPS: {lpips_score:.4f}", ha="center", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"images_processed_nti_v2/reconstruction_{image_path.split('.')[0]}.png", dpi=150, bbox_inches="tight")
            plt.close()
            with open(f"images_processed_nti/metrics.txt", "a") as f:
                f.write(f"Image: {image_path.split('.')[0]}\n")
                f.write(f"PSNR: {psnr:.2f} dB | SSIM: {ssim_score:.4f} | LPIPS: {lpips_score:.4f}\n")
            print(f"Saved reconstruction_{image_path.split('.')[0]}.png")
