import argparse
import gc
import html
import math
import os
import random
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import ftfy
import gradio as gr
import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan as DiffusersWanVAE
from diffusers import WanImageToVideoPipeline
from diffusers.pipelines.wan import pipeline_wan_i2v
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModel


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.Wan.autoencoder_wanT import AutoencoderKLWan
from src.models.Wan.transformer_wan import WanDecoderTransformer


MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
REFDECODER_REPO_ID = "Arrokothwhi/RefDecoder"
# Preferred local/download layout is ckpt/I2V_Wan2.1/model.pt. The second
# entry is a legacy Wan checkpoint filename kept as a fallback.
REFDECODER_CKPT_FILENAMES = (
    "I2V_Wan2.1/model.pt",
    "VAE/Wan2.1/wan2.1_ref.pt",
)
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

# Some diffusers Wan builds reference a module-level `ftfy` during prompt cleaning.
pipeline_wan_i2v.ftfy = ftfy


@dataclass
class DemoConfig:
    model_id: str
    refdecoder_repo_id: str
    checkpoint_path: Path
    output_root: Path
    device: str
    pipe_dtype: torch.dtype
    local_files_only: bool
    cpu_offload: bool
    target_area: int
    fps: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float


class ModelRuntime:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._generation_pipe = None
        self._wan_vae = None
        self._refdecoder = None

    @property
    def device(self):
        return self.config.device

    def get_generation_pipe(self):
        with self._lock:
            if self._generation_pipe is None:
                print(f"[init] Loading Wan I2V generation pipeline from {self.config.model_id}", flush=True)
                self._generation_pipe = load_generation_pipe(self.config)
            return self._generation_pipe

    def get_wan_vae(self):
        with self._lock:
            if self._wan_vae is None:
                print(f"[init] Loading Wan baseline VAE from {self.config.model_id}", flush=True)
                self._wan_vae = load_wan_vae(self.config)
            return self._wan_vae

    def get_refdecoder(self):
        with self._lock:
            if self._refdecoder is None:
                print(f"[init] Loading RefDecoder checkpoint from {self.config.checkpoint_path}", flush=True)
                self._refdecoder = load_refdecoder_module(self.config)
            return self._refdecoder


def default_port():
    for name in ("GRADIO_SERVER_PORT", "PORT"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return 7860


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RefDecoder Gradio I2V demo.")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=default_port())
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--device", default="auto", help="Device for inference, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--model-id", default=MODEL_ID, help="Wan diffusers model id or local directory.")
    parser.add_argument(
        "--refdecoder-repo-id",
        default=REFDECODER_REPO_ID,
        help="Hugging Face repo used only if the RefDecoder checkpoint is not local.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Explicit RefDecoder checkpoint path. Preferred local default: ckpt/I2V_Wan2.1/model.pt.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only use checkpoints that already exist locally or in the Hugging Face cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download Wan or RefDecoder files; require all files to be cached.",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Keep the Wan generation pipeline on the selected device instead of using diffusers CPU offload.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "gradio_outputs")
    parser.add_argument("--target-area", type=int, default=480 * 832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        cuda_device = torch.device(device)
        if cuda_device.index is not None:
            torch.cuda.set_device(cuda_device.index)
    return device


def path_has_file(path):
    return path is not None and path.is_file() and path.stat().st_size > 0


def candidate_checkpoint_paths(args):
    candidates = []
    env_path = os.environ.get("REFDECODER_CKPT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    ckpt_root = ROOT / "ckpt"
    for filename in REFDECODER_CKPT_FILENAMES:
        candidates.append(ckpt_root / filename)

    candidates.extend(
        [
            ckpt_root / "model.pt",
            ROOT.parent / "RefDecoder-HF" / "ckpt" / "model.pt",
        ]
    )
    return candidates


def try_hf_cache(repo_id, filename):
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        return None
    cached = Path(cached)
    return cached if path_has_file(cached) else None


def ensure_refdecoder_checkpoint(args):
    if args.ckpt_path is not None:
        explicit = args.ckpt_path.expanduser().resolve()
        if path_has_file(explicit):
            print(f"[init] RefDecoder checkpoint found at {explicit}", flush=True)
            return explicit
        raise FileNotFoundError(f"--ckpt-path does not exist or is empty: {explicit}")

    for path in candidate_checkpoint_paths(args):
        path = path.expanduser().resolve()
        if path_has_file(path):
            print(f"[init] RefDecoder checkpoint found at {path}", flush=True)
            return path

    for filename in REFDECODER_CKPT_FILENAMES:
        cached = try_hf_cache(args.refdecoder_repo_id, filename)
        if cached is not None:
            print(f"[init] RefDecoder checkpoint found in HF cache at {cached}", flush=True)
            return cached

    if args.no_download or args.local_files_only:
        searched = "\n  - ".join(str(p) for p in candidate_checkpoint_paths(args))
        filenames = ", ".join(REFDECODER_CKPT_FILENAMES)
        raise FileNotFoundError(
            "RefDecoder checkpoint is not available locally.\n"
            f"Searched:\n  - {searched}\n"
            f"Also checked HF cache filenames: {filenames}\n"
            "Pass --ckpt-path, set REFDECODER_CKPT_PATH, or run without --no-download."
        )

    local_dir = ROOT / "ckpt"
    local_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for filename in REFDECODER_CKPT_FILENAMES:
        try:
            print(
                f"[init] RefDecoder checkpoint not found locally; downloading "
                f"{args.refdecoder_repo_id}/{filename}",
                flush=True,
            )
            downloaded = hf_hub_download(
                repo_id=args.refdecoder_repo_id,
                filename=filename,
                local_dir=local_dir,
            )
            downloaded = Path(downloaded)
            if path_has_file(downloaded):
                print(f"[init] RefDecoder checkpoint ready at {downloaded}", flush=True)
                return downloaded
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not find or download a RefDecoder checkpoint from {args.refdecoder_repo_id}."
    ) from last_error


def find_available_port(host, preferred_port):
    if preferred_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    bind_host = "" if host in ("0.0.0.0", "::") else host
    for port in range(preferred_port, preferred_port + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available port found in [{preferred_port}, {preferred_port + 199}]")


def node_hostname():
    return socket.getfqdn() or socket.gethostname() or "localhost"


def print_connection_info(host, port):
    current_node = node_hostname()
    bind_url = f"http://{host}:{port}" if host not in ("0.0.0.0", "::") else f"http://0.0.0.0:{port}"
    print(f"[demo] Listening on: {bind_url}", flush=True)
    print(f"[demo] Running on node: {current_node}", flush=True)
    print(f"[demo] On this node, open: http://127.0.0.1:{port}", flush=True)
    print(f"[demo] From your laptop, tunnel to this node and open: http://localhost:{port}", flush=True)
    print(
        f"[demo] Example via login node: ssh -N -L {port}:{current_node}:{port} "
        f"{os.environ.get('USER', '<user>')}@<login-host>",
        flush=True,
    )


def build_config(args):
    device = resolve_device(args.device)
    pipe_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    checkpoint_path = ensure_refdecoder_checkpoint(args)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return DemoConfig(
        model_id=args.model_id,
        refdecoder_repo_id=args.refdecoder_repo_id,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        device=device,
        pipe_dtype=pipe_dtype,
        local_files_only=args.local_files_only,
        cpu_offload=device.startswith("cuda") and not args.no_cpu_offload,
        target_area=args.target_area,
        fps=args.fps,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


def log_cuda_mem(tag):
    if not torch.cuda.is_available():
        print(f"[mem] {tag}: CUDA not available", flush=True)
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    print(
        f"[mem] {tag}: free={free_bytes / 1024**3:.2f} GB, "
        f"total={total_bytes / 1024**3:.2f} GB, "
        f"allocated={allocated_bytes / 1024**3:.2f} GB, "
        f"reserved={reserved_bytes / 1024**3:.2f} GB",
        flush=True,
    )


def get_module_dtype(module):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def load_generation_pipe(config):
    image_encoder = CLIPVisionModel.from_pretrained(
        config.model_id,
        subfolder="image_encoder",
        torch_dtype=config.pipe_dtype,
        local_files_only=config.local_files_only,
    )
    vae = DiffusersWanVAE.from_pretrained(
        config.model_id,
        subfolder="vae",
        torch_dtype=config.pipe_dtype,
        local_files_only=config.local_files_only,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        config.model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=config.pipe_dtype,
        local_files_only=config.local_files_only,
    )
    if config.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(config.device)
    return pipe


def load_wan_vae(config):
    vae = DiffusersWanVAE.from_pretrained(
        config.model_id,
        subfolder="vae",
        torch_dtype=config.pipe_dtype,
        local_files_only=config.local_files_only,
    )
    vae.eval()
    return vae


def clean_checkpoint_key(key):
    return key.replace("_forward_module.", "").replace("_orig_mod.", "")


def split_refdecoder_state_dict(state_dict):
    vae_sd = {}
    transformer_sd = {}
    for key, value in state_dict.items():
        key = clean_checkpoint_key(key)
        if key.startswith("vae."):
            vae_sd[key[len("vae.") :]] = value
        elif key.startswith("ae."):
            vae_sd[key[len("ae.") :]] = value
        elif key.startswith("model.vae."):
            vae_sd[key[len("model.vae.") :]] = value
        elif key.startswith("model.ae."):
            vae_sd[key[len("model.ae.") :]] = value
        elif key.startswith("transformer."):
            transformer_sd[key[len("transformer.") :]] = value
        elif key.startswith("model.transformer."):
            transformer_sd[key[len("model.transformer.") :]] = value
    return vae_sd, transformer_sd


def print_load_summary(name, load_info):
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    print(
        f"[init] {name}: missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[init] {name} first missing keys: {missing[:5]}", flush=True)
    if unexpected:
        print(f"[init] {name} first unexpected keys: {unexpected[:5]}", flush=True)


def load_refdecoder_module(config):
    vae = AutoencoderKLWan.from_pretrained(
        config.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        gradient_checkpointing=False,
        dropout_p=0.0,
        inference_w_dropout=False,
        use_reference=True,
        skip_decoder_attention=False,
        local_files_only=config.local_files_only,
    ).eval()
    if hasattr(vae, "_init_ref_conv_in"):
        vae._init_ref_conv_in()

    transformer = WanDecoderTransformer(
        chunk=5,
        num_layers=10,
        num_heads=12,
        head_dim=128,
        reusing=True,
        pretrained=False,
        gradient_checkpointing=False,
    ).eval()

    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("module", checkpoint))
    vae_sd, transformer_sd = split_refdecoder_state_dict(state_dict)
    if not vae_sd or not transformer_sd:
        sample_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            "The checkpoint did not contain both RefDecoder VAE and transformer weights. "
            f"Found {len(vae_sd)} VAE keys and {len(transformer_sd)} transformer keys. "
            f"Sample keys: {sample_keys}"
        )

    print(f"[init] Loading {len(vae_sd)} VAE keys and {len(transformer_sd)} transformer keys", flush=True)
    print_load_summary("RefDecoder VAE", vae.load_state_dict(vae_sd, strict=False))
    print_load_summary(
        "RefDecoder transformer",
        transformer.load_state_dict(transformer_sd, strict=False),
    )

    for module in (vae, transformer):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return vae, transformer


def resize_image_for_wan(image, pipe, target_area):
    image = image.convert("RGB")
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    resized = image.resize((width, height))
    return resized, height, width


def build_reference_frame(image, device):
    ref_array = np.asarray(image).astype(np.float32)
    ref_tensor = torch.from_numpy(ref_array).permute(2, 0, 1)
    ref_tensor = (ref_tensor / 255.0 - 0.5) * 2.0
    return ref_tensor.unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.float32)


def normalize_latent_shape(latents):
    if isinstance(latents, list):
        latents = latents[0]
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"Expected latent shape [B,C,T,H,W], got {tuple(latents.shape)}")
    return latents


def gradio_file_url(path):
    return f"/gradio_api/file={quote(str(path), safe='/')}"


def build_compare_html(wan_video_path, ref_video_path, fps):
    compare_id = f"compare-{uuid.uuid4().hex}"
    wan_url = gradio_file_url(wan_video_path) if wan_video_path else ""
    ref_url = gradio_file_url(ref_video_path) if ref_video_path else ""
    base_source = (
        f'<video class="compare-video compare-base" src="{wan_url}" autoplay muted loop playsinline></video>'
        if wan_url
        else '<div class="compare-video compare-base compare-placeholder"></div>'
    )
    overlay_source = (
        f'<video class="compare-video compare-overlay" src="{ref_url}" autoplay muted loop playsinline></video>'
        if ref_url
        else '<div class="compare-video compare-overlay compare-placeholder"></div>'
    )
    inner_doc = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Manrope, Inter, system-ui, sans-serif;
        }}
        .compare-shell {{
          display: flex;
          flex-direction: column;
          gap: 12px;
        }}
        .compare-topbar {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
        }}
        .compare-chip {{
          padding: 12px 22px;
          border-radius: 999px;
          background: rgba(31, 106, 82, 0.14);
          color: #123a2d;
          font-size: 22px;
          font-weight: 800;
          letter-spacing: 0.03em;
          text-transform: uppercase;
          box-shadow: inset 0 0 0 1px rgba(31, 106, 82, 0.12);
          justify-self: start;
        }}
        .compare-chip-right {{
          background: rgba(201, 111, 66, 0.16);
          color: #6e3d23;
          box-shadow: inset 0 0 0 1px rgba(201, 111, 66, 0.16);
          justify-self: end;
        }}
        .compare-button {{
          border: 0;
          border-radius: 999px;
          padding: 10px 22px;
          background: #1f6a52;
          color: white;
          font-size: 16px;
          font-weight: 700;
          cursor: pointer;
          justify-self: center;
        }}
        .compare-stage {{
          position: relative;
          width: 100%;
          aspect-ratio: 16 / 9;
          overflow: hidden;
          border-radius: 22px;
          background: #16120f;
          border: 1px solid rgba(255,255,255,0.08);
        }}
        .compare-video {{
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: contain;
          background: #16120f;
        }}
        .compare-overlay {{
          clip-path: inset(0 0 0 50%);
        }}
        .compare-placeholder {{
          background:
            linear-gradient(135deg, rgba(255,255,255,0.055), transparent 35%),
            #16120f;
        }}
        .compare-divider {{
          position: absolute;
          top: 0;
          bottom: 0;
          left: 50%;
          width: 2px;
          background: rgba(255,255,255,0.96);
          box-shadow: 0 0 0 1px rgba(31, 26, 20, 0.15);
          transform: translateX(-1px);
          pointer-events: none;
        }}
        .compare-divider::after {{
          content: "";
          position: absolute;
          top: 50%;
          left: 50%;
          width: 18px;
          height: 18px;
          border-radius: 999px;
          background: #fff;
          border: 2px solid rgba(31, 26, 20, 0.18);
          transform: translate(-50%, -50%);
        }}
        .compare-range {{
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          opacity: 0.01;
          cursor: ew-resize;
          margin: 0;
          -webkit-appearance: none;
          appearance: none;
        }}
        .compare-caption {{
          color: #201a14;
          font-size: 14px;
          line-height: 1.5;
          text-align: center;
        }}
        .compare-controls {{
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }}
        .compare-controls .compare-button {{
          padding: 9px 16px;
          font-size: 14px;
        }}
        .compare-button-step {{
          background: #2f5746;
        }}
        .compare-button-reset {{
          background: #c96f42;
        }}
        .compare-button[disabled] {{
          opacity: 0.55;
          cursor: not-allowed;
        }}
      </style>
    </head>
    <body>
      <div class="compare-shell" id="{compare_id}">
        <div class="compare-topbar">
          <div class="compare-chip">Wan Baseline</div>
          <div class="compare-chip compare-chip-right">RefDecoder</div>
        </div>
        <div class="compare-stage">
          {base_source}
          {overlay_source}
          <div class="compare-divider"></div>
          <input class="compare-range" type="range" min="0" max="100" value="50" />
        </div>
        <div class="compare-controls">
          <button class="compare-button compare-button-step" type="button" data-action="prev">- 1 Frame</button>
          <button class="compare-button" type="button" data-action="toggle">Pause</button>
          <button class="compare-button compare-button-step" type="button" data-action="next">+ 1 Frame</button>
          <button class="compare-button compare-button-reset" type="button" data-action="reset">Reset Playback</button>
        </div>
        <div class="compare-caption">Drag the divider to compare the two decoders on the same latent video.</div>
      </div>
      <script>
      (() => {{
        const root = document.getElementById("{compare_id}");
        const base = root.querySelector(".compare-base");
        const overlay = root.querySelector(".compare-overlay");
        const divider = root.querySelector(".compare-divider");
        const slider = root.querySelector(".compare-range");
        const button = root.querySelector('[data-action="toggle"]');
        const prevBtn = root.querySelector('[data-action="prev"]');
        const nextBtn = root.querySelector('[data-action="next"]');
        const resetBtn = root.querySelector('[data-action="reset"]');
        const stepButtons = [prevBtn, nextBtn, resetBtn];
        const videos = Array.from(root.querySelectorAll("video"));
        const FRAME_DELTA = 1 / {fps};

        const applySplit = () => {{
          const value = Number(slider.value);
          overlay.style.clipPath = `inset(0 0 0 ${{value}}%)`;
          divider.style.left = `${{value}}%`;
        }};

        const syncVideo = (source, target) => {{
          if (Math.abs((target.currentTime || 0) - (source.currentTime || 0)) > 0.08) {{
            try {{ target.currentTime = source.currentTime; }} catch (e) {{}}
          }}
        }};

        const playBoth = () => {{
          videos.forEach((video) => video.play().catch(() => {{}}));
          button.textContent = "Pause";
        }};

        const pauseBoth = () => {{
          videos.forEach((video) => video.pause());
          button.textContent = "Play";
        }};

        const bindSync = (primary, secondary) => {{
          primary.addEventListener("play", () => secondary.play().catch(() => {{}}));
          primary.addEventListener("pause", () => secondary.pause());
          primary.addEventListener("seeking", () => syncVideo(primary, secondary));
          primary.addEventListener("timeupdate", () => syncVideo(primary, secondary));
          primary.addEventListener("ratechange", () => {{ secondary.playbackRate = primary.playbackRate; }});
        }};

        const stepFrame = (delta) => {{
          if (!videos.length) return;
          pauseBoth();
          videos.forEach((video) => {{
            const duration = isFinite(video.duration) ? video.duration : 0;
            let nextTime = (video.currentTime || 0) + delta;
            if (duration > 0) {{
              nextTime = ((nextTime % duration) + duration) % duration;
            }} else {{
              nextTime = Math.max(0, nextTime);
            }}
            try {{ video.currentTime = nextTime; }} catch (e) {{}}
          }});
        }};

        const resetPlayback = () => {{
          pauseBoth();
          videos.forEach((video) => {{
            try {{ video.currentTime = 0; }} catch (e) {{}}
          }});
        }};

        if (base.tagName === "VIDEO" && overlay.tagName === "VIDEO") {{
          bindSync(base, overlay);
          bindSync(overlay, base);
        }} else {{
          button.disabled = true;
          button.textContent = "Play";
          button.style.opacity = "0.55";
          stepButtons.forEach((btn) => {{ if (btn) btn.disabled = true; }});
        }}

        videos.forEach((video) => {{
          video.addEventListener("loadeddata", playBoth, {{ once: true }});
        }});

        button.addEventListener("click", () => {{
          if (!videos.length || videos[0].paused) {{
            playBoth();
          }} else {{
            pauseBoth();
          }}
        }});

        if (prevBtn) prevBtn.addEventListener("click", () => stepFrame(-FRAME_DELTA));
        if (nextBtn) nextBtn.addEventListener("click", () => stepFrame(FRAME_DELTA));
        if (resetBtn) resetBtn.addEventListener("click", resetPlayback);

        slider.addEventListener("input", applySplit);
        applySplit();

        const reportHeight = () => {{
          const h = Math.ceil(root.getBoundingClientRect().height + 2);
          parent.postMessage({{ type: "compare-iframe-height", id: "{compare_id}", height: h }}, "*");
        }};
        reportHeight();
        window.addEventListener("load", reportHeight);
        if (typeof ResizeObserver !== "undefined") {{
          new ResizeObserver(reportHeight).observe(root);
        }}
        videos.forEach((video) => {{
          video.addEventListener("loadedmetadata", reportHeight);
        }});
      }})();
      </script>
    </body>
    </html>
    """
    return (
        '<iframe class="compare-frame" '
        'sandbox="allow-scripts allow-same-origin" '
        'scrolling="no" '
        'srcdoc="' + html.escape(inner_doc, quote=True) + '"></iframe>'
    )


def save_video_tensor(video_tensor, output_path, fps):
    video = (video_tensor / 2 + 0.5).clamp(0, 1)
    video = video.squeeze(0).permute(1, 2, 3, 0).detach().cpu().float().numpy()
    video = (video * 255).astype(np.uint8)
    imageio.mimwrite(output_path, video, fps=fps, quality=10)
    return str(output_path)


def decode_with_wan_vae(latents, vae, device):
    vae_dtype = get_module_dtype(vae)
    latents = latents.to(device=device, dtype=vae_dtype)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae_dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=vae_dtype).view(1, -1, 1, 1, 1)
    latents = latents * latents_std + latents_mean
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    return video


def decode_with_refdecoder(latents, reference_frame, vae, transformer, device):
    decode_dtype = get_module_dtype(vae)
    latents = latents.to(device=device, dtype=decode_dtype)
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=device,
        dtype=decode_dtype,
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=device,
        dtype=decode_dtype,
    ).view(1, -1, 1, 1, 1)
    latents = latents * latents_std + latents_mean
    reference_frame = reference_frame.to(device=device, dtype=decode_dtype)
    with torch.no_grad():
        video = vae.decode(
            latents,
            transformer,
            return_dict=True,
            reference_frame=reference_frame,
            skip=False,
            window_size=-1,
        ).sample
    if hasattr(vae, "clear_cache"):
        vae.clear_cache()
    return video


def normalize_seed(seed):
    if seed is None:
        return random.randint(0, 2**32 - 1)
    try:
        if math.isnan(float(seed)):
            return random.randint(0, 2**32 - 1)
    except TypeError:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def generate_and_decode(runtime, image, prompt, seed, progress=gr.Progress()):
    with runtime.inference_lock:
        return _generate_and_decode(runtime, image, prompt, seed, progress)


def _generate_and_decode(runtime, image, prompt, seed, progress=gr.Progress()):
    config = runtime.config
    if image is None:
        raise gr.Error("Please upload an input image.")
    if not config.device.startswith("cuda"):
        raise gr.Error("This demo expects a CUDA GPU to run Wan I2V generation.")

    request_start = time.perf_counter()
    prompt = prompt.strip() if prompt else ""
    seed = normalize_seed(seed)
    run_dir = config.output_root / f"refdecoder_demo_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_cuda_mem("request start")
    progress(0.05, desc="Loading Wan generation pipeline")
    pipe = runtime.get_generation_pipe()

    progress(0.12, desc="Generating Wan latents")
    t0 = time.perf_counter()
    resized_image, height, width = resize_image_for_wan(image, pipe, config.target_area)
    generator = torch.Generator(device=config.device).manual_seed(seed)
    with torch.no_grad():
        output = pipe(
            image=resized_image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            output_type="latent",
        )
    latents = normalize_latent_shape(output.frames).detach().cpu()
    latent_secs = time.perf_counter() - t0
    print(f"[timing] latent generation: {latent_secs:.2f}s", flush=True)

    reference_frame = build_reference_frame(resized_image, "cpu")
    latent_path = run_dir / "wan_latents.pt"
    torch.save(
        {
            "latents": latents,
            "height": height,
            "width": width,
            "prompt": prompt,
            "seed": seed,
        },
        latent_path,
    )

    progress(0.78, desc="Decoding Wan baseline")
    t0 = time.perf_counter()
    wan_vae = runtime.get_wan_vae()
    wan_vae.to(config.device)
    try:
        wan_video = decode_with_wan_vae(latents, wan_vae, config.device).detach().cpu()
    finally:
        if config.device.startswith("cuda"):
            wan_vae.to("cpu")
            torch.cuda.empty_cache()
    wan_secs = time.perf_counter() - t0
    print(f"[timing] wan decode: {wan_secs:.2f}s", flush=True)
    wan_video_path = save_video_tensor(wan_video, run_dir / "wan_vae.mp4", config.fps)
    del wan_video
    gc.collect()

    progress(0.9, desc="Decoding RefDecoder")
    t0 = time.perf_counter()
    refdecoder_vae, refdecoder_transformer = runtime.get_refdecoder()
    refdecoder_vae.to(config.device)
    refdecoder_transformer.to(config.device)
    try:
        ref_video = decode_with_refdecoder(
            latents,
            reference_frame,
            refdecoder_vae,
            refdecoder_transformer,
            config.device,
        ).detach().cpu()
    finally:
        if config.device.startswith("cuda"):
            refdecoder_vae.to("cpu")
            refdecoder_transformer.to("cpu")
            torch.cuda.empty_cache()
    ref_secs = time.perf_counter() - t0
    print(f"[timing] refdecoder decode: {ref_secs:.2f}s", flush=True)
    ref_video_path = save_video_tensor(ref_video, run_dir / "refdecoder.mp4", config.fps)
    del ref_video
    gc.collect()

    compare_html = build_compare_html(wan_video_path, ref_video_path, config.fps)
    total_secs = time.perf_counter() - request_start
    print(
        f"[timing] request total: {total_secs:.2f}s "
        f"(latents={latent_secs:.2f}s, wan={wan_secs:.2f}s, ref={ref_secs:.2f}s)",
        flush=True,
    )

    return (
        gr.update(value=compare_html, visible=True),
        wan_video_path,
        ref_video_path,
        "",
        gr.update(value=wan_video_path, interactive=True),
        gr.update(value=ref_video_path, interactive=True),
    )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --page-bg: #f4f1e8;
    --card-bg: rgba(255, 252, 246, 0.92);
    --card-border: rgba(50, 43, 32, 0.12);
    --accent: #1f6a52;
    --accent-2: #c96f42;
    --text-main: #201a14;
    --text-soft: #201a14;
    --ui-font: "Manrope", "Inter", "Segoe UI", sans-serif;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(201, 111, 66, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(31, 106, 82, 0.16), transparent 28%),
        linear-gradient(180deg, #f8f4ec 0%, var(--page-bg) 100%);
    font-family: var(--ui-font);
}

.app-shell {
    max-width: 1320px;
    margin: 0 auto;
}

.hero-card,
.panel-card,
.output-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 24px;
    box-shadow: 0 18px 50px rgba(49, 39, 26, 0.08);
}

.hero-card {
    padding: 28px 30px 20px 30px;
    margin-bottom: 18px;
}

.hero-title {
    margin: 14px 0 8px 0;
    font-size: 42px;
    line-height: 1.05;
    font-weight: 800;
    color: var(--text-main);
}

.hero-copy {
    margin: 0;
    max-width: 840px;
    color: var(--text-soft);
    font-size: 17px;
    line-height: 1.6;
    font-family: var(--ui-font);
}

.panel-card,
.output-card {
    padding: 18px;
}

.panel-card {
    overflow: hidden;
}

.section-title {
    margin: 0 0 6px 0;
    color: var(--text-main);
    font-size: 22px;
    font-weight: 750;
}

.section-copy {
    margin: 0 0 14px 0;
    color: var(--text-soft);
    font-size: 14px;
    line-height: 1.55;
    font-family: var(--ui-font);
}

#generate-btn {
    min-height: 108px;
    height: 100%;
    width: 100%;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, #154f3d 100%);
    border: none;
}

#generate-btn:hover {
    filter: brightness(1.04);
}

.compare-frame {
    width: 100%;
    aspect-ratio: 16 / 11;
    border: 0;
    background: transparent;
    overflow: hidden;
    display: block;
    transition: height 120ms ease;
}

.compare-panel {
    padding-bottom: 34px;
}

.seed-action-row {
    align-items: stretch;
}

.seed-action-row > .gradio-column {
    min-width: 0;
}

.run-status {
    margin-top: 8px;
    color: var(--text-soft);
    font-size: 13px;
    line-height: 1.4;
    min-height: 1.4em;
}

.run-status p {
    margin: 0;
}

.download-row {
    margin-top: 12px;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
}

.download-row button {
    border: 0 !important;
    border-radius: 999px !important;
    padding: 10px 22px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    min-height: 0 !important;
}

button.download-baseline {
    background: var(--accent) !important;
    color: #fff !important;
}

button.download-ref {
    background: var(--accent-2) !important;
    color: #fff !important;
}

.download-row button:hover:not([disabled]):not(:disabled) {
    filter: brightness(1.05);
}

button.download-baseline[disabled],
button.download-baseline:disabled {
    background: rgba(31, 106, 82, 0.14) !important;
    color: #123a2d !important;
    box-shadow: inset 0 0 0 1px rgba(31, 106, 82, 0.12) !important;
    opacity: 1 !important;
    cursor: not-allowed;
}

button.download-ref[disabled],
button.download-ref:disabled {
    background: rgba(201, 111, 66, 0.16) !important;
    color: #6e3d23 !important;
    box-shadow: inset 0 0 0 1px rgba(201, 111, 66, 0.16) !important;
    opacity: 1 !important;
    cursor: not-allowed;
}
"""


def create_demo(runtime):
    config = runtime.config
    with gr.Blocks(title="RefDecoder I2V Demo", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                """
                <script>
                (() => {
                    if (window.__refdecoderResizeBound) return;
                    window.__refdecoderResizeBound = true;

                    const STAGE_RATIO = 9 / 16;
                    const CHROME = 160;
                    const observed = new WeakSet();

                    const estimateHeight = (iframe) => {
                        if (iframe.dataset.exactSized === "1") return;
                        const w = iframe.getBoundingClientRect().width;
                        if (w > 0) {
                            iframe.style.height = Math.round(w * STAGE_RATIO + CHROME) + "px";
                        }
                    };

                    const trackIframe = (iframe) => {
                        if (observed.has(iframe)) return;
                        observed.add(iframe);
                        estimateHeight(iframe);
                        new ResizeObserver(() => estimateHeight(iframe)).observe(iframe);
                    };

                    document.querySelectorAll("iframe.compare-frame").forEach(trackIframe);

                    new MutationObserver((mutations) => {
                        for (const m of mutations) {
                            for (const n of m.addedNodes) {
                                if (n.nodeType !== 1) continue;
                                if (n.matches && n.matches("iframe.compare-frame")) trackIframe(n);
                                const inner = n.querySelectorAll && n.querySelectorAll("iframe.compare-frame");
                                if (inner) inner.forEach(trackIframe);
                            }
                        }
                    }).observe(document.body, { childList: true, subtree: true });

                    window.addEventListener("message", (e) => {
                        if (!e.data || e.data.type !== "compare-iframe-height") return;
                        const h = Math.max(200, Number(e.data.height) || 0);
                        document.querySelectorAll("iframe.compare-frame").forEach((f) => {
                            if (f.contentWindow === e.source) {
                                f.style.height = h + "px";
                                f.dataset.exactSized = "1";
                            }
                        });
                    });
                })();
                </script>
                <div class="hero-card">
                    <div class="hero-title">RefDecoder I2V Demo</div>
                    <p class="hero-copy">
                        Upload one image, optionally add a prompt, and compare two decoders on the same Wan latent video.
                        The app generates latents once, then renders them with Wan's original VAE and with RefDecoder.
                    </p>
                </div>
                """
            )

            with gr.Column(elem_classes=["panel-card", "compare-panel"]):
                gr.HTML(
                    """
                    <div class="section-title">Inputs</div>
                    <div class="section-copy">
                        Upload a reference image, optionally add a prompt, and compare the decoders below.
                    </div>
                    """
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        image_input = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=180,
                        )
                    with gr.Column(scale=5):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            placeholder="A woman turns toward the camera as her hair moves in the wind...",
                        )
                        with gr.Row(equal_height=True, elem_classes="seed-action-row"):
                            with gr.Column(scale=1):
                                seed_input = gr.Number(
                                    label="Seed",
                                    value=None,
                                    precision=0,
                                    info="Optional",
                                )
                            with gr.Column(scale=1):
                                run_button = gr.Button(
                                    "Generate Comparison",
                                    variant="primary",
                                    elem_id="generate-btn",
                                )
                        status_md = gr.Markdown(value="", elem_classes="run-status")

            with gr.Column(elem_classes="panel-card"):
                gr.HTML(
                    """
                    <div class="section-title">Decoder Comparison</div>
                    <div class="section-copy">
                        Left side shows Wan Baseline. Right side shows RefDecoder. Drag the divider across the frame to compare them.
                    </div>
                    """
                )
                compare_output = gr.HTML(value=build_compare_html(None, None, config.fps))

                with gr.Row(elem_classes="download-row"):
                    wan_download_btn = gr.DownloadButton(
                        label="Download Baseline",
                        value=None,
                        interactive=False,
                        elem_classes="download-baseline",
                    )
                    ref_download_btn = gr.DownloadButton(
                        label="Download RefDecoder",
                        value=None,
                        interactive=False,
                        elem_classes="download-ref",
                    )

                wan_video_hidden = gr.Video(visible=False)
                ref_video_hidden = gr.Video(visible=False)

            def reset_for_new_run():
                return (
                    "",
                    gr.update(value=None, interactive=False),
                    gr.update(value=None, interactive=False),
                )

            def run_generation(image, prompt, seed, progress=gr.Progress()):
                return generate_and_decode(runtime, image, prompt, seed, progress)

            run_button.click(
                fn=reset_for_new_run,
                inputs=None,
                outputs=[status_md, wan_download_btn, ref_download_btn],
                queue=False,
                show_progress="hidden",
            ).then(
                fn=run_generation,
                inputs=[image_input, prompt_input, seed_input],
                outputs=[
                    compare_output,
                    wan_video_hidden,
                    ref_video_hidden,
                    status_md,
                    wan_download_btn,
                    ref_download_btn,
                ],
            )
    return demo


def main():
    args = parse_args()
    config = build_config(args)
    runtime = ModelRuntime(config)
    demo = create_demo(runtime)

    port = find_available_port(args.host, args.port)
    print(f"[demo] RefDecoder checkpoint: {config.checkpoint_path}", flush=True)
    print(f"[demo] Output directory: {config.output_root}", flush=True)
    print(f"[demo] Device: {config.device}", flush=True)
    print_connection_info(args.host, port)

    demo.queue(max_size=args.queue_size).launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        allowed_paths=[str(config.output_root)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
