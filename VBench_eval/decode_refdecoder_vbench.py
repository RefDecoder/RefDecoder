import argparse
import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers.utils import load_image
from utils.common_utils import instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser(description="Decode saved latents with RefDecoder")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--config_path", type=str, required=True, help="Path to RefDecoder config YAML")
    parser.add_argument("--latent_dir", type=str, required=True, help="Directory containing .pt latent files")
    parser.add_argument("--image_folder", type=str,
                        default="<PATH_TO_VBENCH_IMAGES>",  # e.g. /path/to/VBench/vbench2_beta_i2v/data/crop/16-9
                        help="Path to VBench images (for reference frames)")
    parser.add_argument("--output_dir", type=str,
                        default="<PATH_TO_OUTPUT_DIR>")  # e.g. /path/to/RefDecoder/VBench_eval/refdecoder_480p_videos
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_refdecoder_model(config_path, device="cuda"):
    print(f"Loading RefDecoder model from config: {config_path}")
    config = OmegaConf.load(config_path)
    if hasattr(config.model, "params"):
        config.model.params.inference_w_dropout = False
    model = instantiate_from_config(config.model)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"RefDecoder model loaded.")
    return model


def prepare_reference_frame(image_path, height, width, device):
    image = load_image(image_path)
    resized = image.resize((width, height), Image.LANCZOS)
    ref_array = np.array(resized).astype(np.float32)
    ref_tensor = torch.from_numpy(ref_array).permute(2, 0, 1)  # [C, H, W]
    ref_tensor = (ref_tensor / 255.0 - 0.5) * 2.0
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.float32)  # [1, C, 1, H, W]
    return ref_tensor


def main():
    args = parse_args()

    # --- Load models ---
    refdecoder_model = load_refdecoder_model(args.config_path, args.device)

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
        height = data["height"]
        width = data["width"]
        image_name = data["image_name"]

        # Prepare reference frame from the original image
        image_path = os.path.join(args.image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Reference image not found {image_path}, skipping")
            continue

        reference_frame = prepare_reference_frame(image_path, height, width, args.device)

        # Denormalize latents (Wan uses latents_mean/std from VAE config)
        latents_mean = torch.tensor(refdecoder_model.ae.config.latents_mean).view(1, -1, 1, 1, 1).to(args.device, dtype=torch.float32)
        latents_std = torch.tensor(refdecoder_model.ae.config.latents_std).view(1, -1, 1, 1, 1).to(args.device, dtype=torch.float32)
        latents = latents * latents_std + latents_mean

        with torch.no_grad():
            video = refdecoder_model.ae.decode(
                latents,
                transformer=refdecoder_model.transformer,
                return_dict=True,
                reference_frame=reference_frame,
                skip=False,
                window_size=-1,
            ).sample

        if hasattr(refdecoder_model.ae, 'clear_cache'):
            refdecoder_model.ae.clear_cache()

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
