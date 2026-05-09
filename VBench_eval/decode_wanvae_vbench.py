import argparse
import os
import glob
import torch
import numpy as np
from diffusers import AutoencoderKLWan
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Decode saved latents with Wan's original VAE")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--latent_dir", type=str, required=True, help="Directory containing .pt latent files")
    parser.add_argument("--output_dir", type=str,
                        default="<PATH_TO_OUTPUT_DIR>")  # e.g. /path/to/RefDecoder/VBench_eval/wanvae_480p_videos
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load Wan VAE ---
    MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    print(f"Loading Wan VAE from {MODEL_ID}...")
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(args.device)
    vae.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Gather latent files ---
    all_latent_files = sorted(glob.glob(os.path.join(args.latent_dir, "*.pt")))
    my_latent_files = all_latent_files[args.rank::args.world_size]

    print(f"[GPU {args.rank}] Decoding {len(my_latent_files)} latent files (of {len(all_latent_files)} total)")

    # --- Decode Loop ---
    for latent_path in tqdm(my_latent_files, desc=f"GPU {args.rank}"):
        basename = os.path.splitext(os.path.basename(latent_path))[0]
        save_path = os.path.join(args.output_dir, f"{basename}.mp4")

        if os.path.exists(save_path):
            continue

        data = torch.load(latent_path, map_location="cpu")
        latents = data["latents"].to(args.device, dtype=torch.float32)

        # Denormalize latents
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(args.device, dtype=torch.float32)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(args.device, dtype=torch.float32)
        latents = latents * latents_std + latents_mean

        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]

        # Convert to video frames
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        video = (video * 255).astype(np.uint8)

        import imageio
        imageio.mimwrite(save_path, video, fps=16, quality=10)
        print(f"  -> Saved: {os.path.basename(save_path)}")

        torch.cuda.empty_cache()

    print(f"[GPU {args.rank}] Complete.")


if __name__ == "__main__":
    main()
