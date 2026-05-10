import argparse
import os
import json
import random
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate latents only (no decoding) for VBench")
    parser.add_argument("--rank", type=int, default=0, help="The GPU index for this process (0-7)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--vbench_info_path", type=str,
                        default="<PATH_TO_VBENCH_INFO_JSON>")  # e.g. /path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json
    parser.add_argument("--image_folder", type=str,
                        default="<PATH_TO_VBENCH_IMAGES>")  # e.g. /path/to/VBench/vbench2_beta_i2v/data/crop/16-9
    parser.add_argument("--output_dir", type=str,
                        default="<PATH_TO_OUTPUT_DIR>")  # e.g. /path/to/RefDecoder/VBench_example/wan2.1_480p_latents
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Use CPU offload for large models")
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    Target_Area = 480 * 832

    NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # --- Load Model ---
    print(f"Loading {MODEL_ID}...")
    image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    if args.use_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load VBench Metadata ---
    with open(args.vbench_info_path, 'r') as f:
        full_info_list = json.load(f)

    my_info_list = full_info_list[args.rank::args.world_size]

    print(f"[GPU {args.rank}] Processing {len(my_info_list)} items (subset of {len(full_info_list)})")

    GPU_SEED_LOG = os.path.join(args.output_dir, f"seeds_gpu_{args.rank}.json")
    if os.path.exists(GPU_SEED_LOG):
        with open(GPU_SEED_LOG, 'r') as f:
            gpu_seed_log = json.load(f)
    else:
        gpu_seed_log = {}

    # --- Generation Loop ---
    for i, info in enumerate(tqdm(my_info_list, desc=f"GPU {args.rank}")):
        prompt = info["prompt_en"]

        missing_indices = []
        for index in range(args.samples_per_prompt):
            save_filename = f"{prompt}-{index}.pt"
            save_path = os.path.join(args.output_dir, save_filename)
            if not os.path.exists(save_path):
                missing_indices.append(index)

        if not missing_indices:
            continue

        image_filename = info["image_name"]
        image_path = os.path.join(args.image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found {image_path}")
            continue

        # Prepare Wan2.1 Input (Resize logic)
        original_image = load_image(image_path)
        aspect_ratio = original_image.height / original_image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]  # 16

        height = round(np.sqrt(Target_Area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(Target_Area / aspect_ratio)) // mod_value * mod_value
        resized_image = original_image.resize((width, height))

        print(f"[GPU {args.rank}] {prompt[:30]}... (Missing: {missing_indices})")

        for index in missing_indices:
            save_filename = f"{prompt}-{index}.pt"
            save_path = os.path.join(args.output_dir, save_filename)

            # Log or reuse seed
            if save_filename in gpu_seed_log:
                seed = gpu_seed_log[save_filename]
                print(f"  -> Reusing seed {seed} for {save_filename}")
            else:
                seed = random.randint(0, 2**32 - 1)
                gpu_seed_log[save_filename] = seed
                with open(GPU_SEED_LOG, 'w') as f:
                    json.dump(gpu_seed_log, f, indent=4)

            generator = torch.Generator(device=args.device).manual_seed(seed)

            with torch.no_grad():
                output = pipe(
                    image=resized_image,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_type="latent",
                )
                latents = output.frames  # normalized latents

            # Save latents and metadata
            torch.save({
                "latents": latents.cpu(),
                "height": height,
                "width": width,
                "image_name": image_filename,
                "prompt": prompt,
                "seed": seed,
            }, save_path)

            print(f"  -> Saved latents: {save_filename}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[GPU {args.rank}] Complete.")


if __name__ == "__main__":
    main()
