import os
import torch
import argparse
import logging
from decord import VideoReader, cpu
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.common_utils import instantiate_from_config
import torchvision

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# Chunk-size constraints per backend.
# Wan accepts (4k + 1) frames up to 17. VideoVAE+ accepts multiples of 4 up to 16.
WAN_MAX_CHUNK = 17           # Valid: {1, 5, 9, 13, 17}  (chunk_size %% 4 == 1)
VIDEOVAEPLUS_MAX_CHUNK = 16  # Valid: {4, 8, 12, 16}     (chunk_size %% 4 == 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RefDecoder Reconstruction Inference (supports Wan and VideoVAE+ variants)"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing input .mp4 videos.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Folder to save reconstructed videos.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to inference config YAML.")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "wan", "videovaeplus"],
                        help="Backend variant. 'auto' detects from the config target.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Frames per chunk. Wan: %%4==1 and <=17. VideoVAE+: %%4==0 and <=16. "
                             "Defaults to 17 for wan, 16 for videovaeplus.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 832],
                        help="Target resolution (height, width).")
    parser.add_argument("--ref_frame_idx", type=int, default=0,
                        help="Index of the reference frame in the input video. "
                             "Used identically by both backbones.")
    return parser.parse_args()


def detect_model_type(model, config):
    """Detect whether the loaded model is the Wan or VideoVAE+ variant."""
    target = config.model.get("target", "")
    if "Wan" in target or "transformerwrapper" in target.lower():
        return "wan"
    if "VideoVaePlus" in target or "videovaeplus" in target.lower():
        return "videovaeplus"
    # Fallback: structural detection. Wan wrapper exposes both `ae` and `transformer`.
    if hasattr(model, "ae") and hasattr(model, "transformer"):
        return "wan"
    return "videovaeplus"


def validate_chunk_size(chunk_size, model_type):
    """Ensure chunk_size obeys the per-backend constraint."""
    if model_type == "wan":
        max_chunk, mod, mod_msg = WAN_MAX_CHUNK, 1, "chunk_size %% 4 == 1"
    else:
        max_chunk, mod, mod_msg = VIDEOVAEPLUS_MAX_CHUNK, 0, "chunk_size %% 4 == 0"
    if chunk_size % 4 != mod:
        raise ValueError(f"For {model_type}, {mod_msg}, got {chunk_size}")
    if chunk_size <= 0 or chunk_size > max_chunk:
        raise ValueError(
            f"chunk_size must be in (0, {max_chunk}] for {model_type}, got {chunk_size}"
        )


def data_processing(video_path, resolution):
    """Load and preprocess video. Returns frames in [C, T, H, W] in [-1, 1]."""
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_resolution = video_reader[0].shape

        resolution = [
            min(video_resolution[0], resolution[0]),
            min(video_resolution[1], resolution[1]),
        ]
        video_reader = VideoReader(
            video_path, ctx=cpu(0), width=resolution[1], height=resolution[0]
        )

        video_length = len(video_reader)
        vid_fps = video_reader.get_avg_fps()
        frame_indices = list(range(0, video_length))
        frames = video_reader.get_batch(frame_indices)

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [C, T, H, W]
        frames = (frames / 255 - 0.5) * 2
        return frames, vid_fps
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return None, None


def save_video(tensor, save_path, fps: float):
    try:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        arr = tensor.detach().cpu().squeeze().to(torch.uint8)
        torchvision.io.write_video(
            save_path, arr.permute(1, 2, 3, 0), fps=fps,
            options={'codec': 'libx264', 'crf': '15'}
        )
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


def reconstruct_chunk_wan(model, chunk, reference_frame, device):
    """Wan-variant: encode -> decode(z, reference_frame=...)."""
    chunk = chunk.to(device)
    reference_frame = reference_frame.to(device)
    posterior = model.encode(chunk)
    z = posterior.mode()
    recon = model.decode(z, reference_frame=reference_frame, window_size=-1)
    if hasattr(model.ae, 'clear_cache'):
        model.ae.clear_cache()
    return recon


def reconstruct_chunk_videovaeplus(model, chunk, reference_frame, device):
    """
    VideoVAE+ variant. We bypass `model.forward` (which samples a random reference
    index internally) and run encode -> ref_conv_in -> decode explicitly so the
    caller-supplied reference_frame is used.
    """
    chunk = chunk.to(device)
    reference_frame = reference_frame.to(device)
    z, _posterior = model.encode(chunk, sample_posterior=False)
    reference_token = model.ref_conv_in(reference_frame)
    recon = model.decode(
        z, transformer=model.transformer, reference_token=reference_token
    )
    return recon


def process_in_chunks(video_data, model, model_type, chunk_size, ref_frame_idx, device):
    """
    Reconstruct a video chunk-by-chunk.

    video_data: [1, C, T, H, W] in [-1, 1]
    Returns:    [1, C, T, H, W] reconstruction with the same T as input.
    """
    try:
        num_frames = video_data.size(2)

        # Reference frame: picked from the original (un-padded) video.
        # Same selection logic for both backbones.
        ref_idx = max(0, min(ref_frame_idx, num_frames - 1))
        reference_frame = video_data[:, :, ref_idx:ref_idx + 1, :, :].clone()

        # Pad to a multiple of chunk_size by repeating the last frame.
        padding_frames = 0
        if num_frames % chunk_size != 0:
            padding_frames = chunk_size - (num_frames % chunk_size)
            pad = video_data[:, :, -1:, :, :].repeat(1, 1, padding_frames, 1, 1)
            video_data = torch.cat([video_data, pad], dim=2)

        output_chunks = []
        total = video_data.size(2)
        for start in range(0, total, chunk_size):
            chunk = video_data[:, :, start:start + chunk_size, :, :]
            with torch.no_grad():
                if model_type == "wan":
                    recon_chunk = reconstruct_chunk_wan(model, chunk, reference_frame, device)
                else:
                    recon_chunk = reconstruct_chunk_videovaeplus(model, chunk, reference_frame, device)
            output_chunks.append(recon_chunk.cpu().float())

        ret = torch.cat(output_chunks, dim=2)
        if padding_frames > 0:
            ret = ret[:, :, :-padding_frames, :, :]
        return ret
    except Exception as e:
        logging.error(f"Error processing chunks: {e}")
        return None


def main():
    args = parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    config = OmegaConf.load(args.config_path)

    # Disable inference dropout if present in the config
    if hasattr(config.model, "params") and hasattr(config.model.params, "inference_w_dropout"):
        config.model.params.inference_w_dropout = False

    model = instantiate_from_config(config.model)
    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Resolve model type
    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_model_type(model, config)
    logging.info(f"Model type: {model_type}")

    # Resolve and validate chunk size
    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = WAN_MAX_CHUNK if model_type == "wan" else VIDEOVAEPLUS_MAX_CHUNK
    validate_chunk_size(chunk_size, model_type)
    logging.info(f"Chunk size: {chunk_size}")

    all_videos = sorted(glob(os.path.join(args.data_root, "*.mp4")))
    if not all_videos:
        logging.error(f"No .mp4 videos found in {args.data_root}")
        return

    for video_path in tqdm(all_videos, desc="Reconstructing", unit="video"):
        logging.info(f"Processing: {video_path}")
        frames, vid_fps = data_processing(video_path, args.resolution)
        if frames is None:
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = frames.unsqueeze(0)  # [1, C, T, H, W]

        recon = process_in_chunks(
            frames, model, model_type, chunk_size, args.ref_frame_idx, args.device
        )
        if recon is None:
            continue

        save_path = os.path.join(args.out_root, f"{video_name}_reconstructed.mp4")
        save_video(recon, save_path, vid_fps)


if __name__ == "__main__":
    main()
