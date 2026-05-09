import gc
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed._composable.fsdp import fully_shard

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from src.models.Wan.autoencoder_wanT import AutoencoderKLWan
from src.models.Wan.transformer_wan import WanDecoderTransformer
from utils.common_utils import instantiate_from_config
from utils.debug_decoderT import MinimalDecoderTracker
from utils.attnScheduler import attnScheduler

class AutoencoderWanTransformer(pl.LightningModule):
    def __init__(
        self,
        lossconfig,
        base_learning_rate=1.0e-4,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        logdir=None,
        input_dim=4,
        freeze_encoder=False, # Freeze encoder
        freeze_quant_conv=False, # Freeze quant_conv and post_quant_conv
        freeze_decoder=False,  # Freeze decoder except attention blocks
        use_discriminator=True,
        use_reference=False,
        skip_decoder_attention=False,
        # =====LR schedule=====
        warmup_steps=1000,
        T_max=100000,
        # =====Self Attention Layers=====
        chunk=2,
        num_layers=30,
        num_heads=12,
        head_dim=128,
        reusing=False,
        pretrained=True,
        # =====CHECKPOINT OPTIONS=====
        gradient_checkpointing=False,
        # ======LoRA OPTIONS=====
        use_lora=False,  # Enable LoRA fine-tuning
        lora_rank=4,  # Rank of LoRA matrices
        lora_alpha=32,  # LoRA scaling factor
        lora_dropout=0.1,  # Dropout for LoRA layers
        # ======CURRICULUM LEARNING OPTIONS=====
        curriculum_enabled=False,
        curriculum_start_window=0,
        curriculum_max_window=0,
        curriculum_warmup_steps=0,
        curriculum_transition_steps=0,
        curriculum_full_attention_at_end=False,
        # ======DEBUG OPTIONS=====
        track_layers=False,
        log_og_loss=True,
        inference_w_dropout=False,
        dropout_p=0.7,
        val_begining=False,
    ):
        super().__init__()
        self.base_learning_rate = base_learning_rate
        self.image_key = image_key
        self.use_discriminator = use_discriminator
        self.use_reference = use_reference and not skip_decoder_attention
        self.log_og_loss = log_og_loss
        self.inference_w_dropout = inference_w_dropout
        self.dropout_p = dropout_p
        self.skip_decoder_attention = skip_decoder_attention
        self.val_begining = val_begining
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk = chunk
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.reusing = reusing
        self.pretrained = pretrained
        self.warmup_steps = warmup_steps
        self.T_max = T_max

        self.layer_tracker = None
        self.track_layers = track_layers

        if curriculum_enabled and use_reference and not skip_decoder_attention:
            self.attn_scheduler = attnScheduler(
                start_window=curriculum_start_window,
                max_window=curriculum_max_window,
                warmup_steps=curriculum_warmup_steps,
                transition_steps=curriculum_transition_steps,
                full_attention_at_end=curriculum_full_attention_at_end,
                enabled=curriculum_enabled,
            )
        else:
            self.attn_scheduler = None
            if curriculum_enabled:
                print("Note: Curriculum learning disabled because use_reference=False or skip_decoder_attention=True")

        # LoRA settings
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Track checkpoint loading status
        self._checkpoint_loaded = False  # Will be set to True if checkpoint loads
        self._loading_from_ckpt_path = ckpt_path is not None  # True if loading via ckpt_path

        # Initialize the Wan autoencoder
        self.ae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
            ignore_mismatched_sizes=True,
            gradient_checkpointing=self.gradient_checkpointing,
            dropout_p=self.dropout_p,
            inference_w_dropout=self.inference_w_dropout,
            use_reference=True,
            skip_decoder_attention=self.skip_decoder_attention
        )

        self.ae._init_ref_conv_in()

        self.transformer = WanDecoderTransformer(
            chunk=self.chunk,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            reusing=self.reusing,
            pretrained=self.pretrained,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        if freeze_encoder:
            self._freeze_encoder()

        if freeze_quant_conv:
            self._freeze_quant_conv()

        if freeze_decoder:
            self._freeze_decoder()

        # Setup loss
        self.loss = instantiate_from_config(lossconfig)     
        self.loss.eval()

        for p in self.loss.parameters():
            p.requires_grad = False

        self.loss_fn = lpips.LPIPS(net='alex')
        for p in self.loss_fn.parameters():
            p.requires_grad = False

        self.input_dim = input_dim
        self.logdir = logdir

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        
        # Extract the actual state dict
        if "state_dict" in sd:
            self._cur_epoch = sd.get("epoch", "null")
            sd = sd["state_dict"]
        elif "module" in sd:
            self._cur_epoch = sd.get("global_step", "null")
            sd = sd["module"]
        else:
            self._cur_epoch = "null"

        # Clean up keys
        cleaned_sd = {}
        for k, v in sd.items():
            # Skip ignored keys
            if any(k.startswith(ik) for ik in ignore_keys):
                continue
            # Remove DeepSpeed and torch.compile prefixes
            new_key = k.replace("_forward_module.", "").replace("_orig_mod.", "")
            cleaned_sd[new_key] = v

        load_info = self.load_state_dict(cleaned_sd, strict=False)

        print(f"\n====== Loaded checkpoint from {path} ======")
        print("Loaded keys:", len(sd))
        
        print("\n Missing keys (model has, ckpt doesn't):")
        for k in load_info.missing_keys:
            print("  -", k)

        print("\n Unexpected keys (ckpt has, model doesn't):")
        for k in load_info.unexpected_keys:
            print("  -", k)

        print("==========================================\n")

        self._checkpoint_loaded = True

    
    def on_load_checkpoint(self, checkpoint):
        """
        Called by PyTorch Lightning when loading checkpoint via --resume_from_checkpoint or --auto_resume.
        This happens AFTER __init__ but BEFORE training starts.
        """
        self._checkpoint_loaded = True
        epoch = checkpoint.get('epoch', 'unknown')
        global_step = checkpoint.get('global_step', 'unknown')
        print(f"✓ PyTorch Lightning loaded checkpoint from epoch {epoch}, step {global_step}")

    def compile_model(self):
        self.ae = torch.compile(
            self.ae, 
            mode="reduce-overhead",
            fullgraph=False
        )
        self.transformer = torch.compile(
            self.transformer, 
            mode="reduce-overhead",
            fullgraph=False
        )

    def encode(self, x):
        posterior = self.ae.encode(x, return_dict=True).latent_dist
        return posterior

    def decode(self, z, reference_frame=None, skip=False, window_size=-1):
        dec = self.ae.decode(
            z,
            self.transformer,
            return_dict=True,
            reference_frame=reference_frame,
            skip=skip,
            window_size=window_size,
        ).sample
        return dec

    def forward(self, input,sample_posterior=True):
        reference_frame = None

        if self.use_reference:
            # Extract ***random*** frame as reference
            idx = 0 if input.size(2) == 1 else torch.randint(0, input.size(2), ()).item()
            reference_frame = input[:, :, idx:idx + 1, :, :].clone()

        posterior = self.encode(input)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        window_size = -1
        if self.attn_scheduler is not None:
            window_size = self.attn_scheduler.get_window_size(self.global_step)
            
        dec = self.decode(z, reference_frame=reference_frame, window_size=window_size)
        
        return dec, posterior, idx, z
    

    def get_input(self, batch, k):
        x = batch[k]
        if x.dim() == 5 and self.input_dim == 4:
            b, c, t, h, w = x.shape
            self.b = b
            self.t = t
            x = rearrange(x, "b c t h w -> (b t) c h w")

        # Wan expects 5D input: (b, c, t, h, w)
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Add time dimension

        return x

    def training_step(self, batch, batch_idx):
        # Log curriculum progress every 10 steps
        if self.global_step % 10 == 0 and self.attn_scheduler is not None:
            phase_info = self.attn_scheduler.get_phase_info(self.global_step)
            self.log("curriculum/window_size", float(phase_info["window_size"]), prog_bar=False, logger=True, on_step=True)

        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, idx, _ = self(inputs)

        # Handle discriminator being disabled
        if not self.use_discriminator:
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "train/aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True
            )
            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True
            )

            if self.global_step % 10 == 0:
                self._compute_and_log_psnr(inputs, reconstructions, prefix="train")
                self._compute_and_log_ssim(inputs, reconstructions, prefix="train")
                self._compute_and_log_lpips(inputs, reconstructions, prefix="train")

                # log reference frame reconsturctions metrics
                inputs_ref = inputs[:, :, idx:idx + 1, :, :].clone()
                rec_ref = reconstructions[:, :, idx:idx + 1, :, :].clone()
                self._compute_and_log_psnr(inputs_ref, rec_ref, prefix="train_ref")
                self._compute_and_log_ssim(inputs_ref, rec_ref, prefix="train_ref")
                self._compute_and_log_lpips(inputs_ref, rec_ref, prefix="train_ref")

                # log other frames reconsturctions metrics
                if inputs.size(2) != 1:
                    inputs_other = torch.cat([inputs[:, :, :idx, :, :], inputs[:, :, idx + 1:, :, :]], dim=2)
                    reconstructions_other = torch.cat([reconstructions[:, :, :idx, :, :], reconstructions[:, :, idx + 1:, :, :]], dim=2)
                    self._compute_and_log_psnr(inputs_other, reconstructions_other, prefix="train_other")
                    self._compute_and_log_ssim(inputs_other, reconstructions_other, prefix="train_other")
                    self._compute_and_log_lpips(inputs_other, reconstructions_other, prefix="train_other")

                if self.track_layers:
                    decoder_stats = self.layer_tracker.get_stats()
                    log_layer = {}
                    for name, value in sorted(decoder_stats.items()):
                        log_layer[f"layer_stats/{name}"] = value
                    self.log_dict(
                        log_layer,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True
                    )
                    self.layer_tracker.clear()


            return aeloss

        # Original behavior when discriminator is enabled
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "train/aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True
            )
            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True
            )

            if self.global_step % 10 == 0:
                self._compute_and_log_psnr(inputs, reconstructions, prefix="train")
                self._compute_and_log_ssim(inputs, reconstructions, prefix="train")
                self._compute_and_log_lpips(inputs, reconstructions, prefix="train")

                # log reference frame reconsturctions metrics
                inputs_ref = inputs[:, :, idx:idx + 1, :, :].clone()
                rec_ref = reconstructions[:, :, idx:idx + 1, :, :].clone()
                self._compute_and_log_psnr(inputs_ref, rec_ref, prefix="train_ref")
                self._compute_and_log_ssim(inputs_ref, rec_ref, prefix="train_ref")
                self._compute_and_log_lpips(inputs_ref, rec_ref, prefix="train_ref")

                # log other frames reconsturctions metrics
                if inputs.size(2) != 1:
                    inputs_other = torch.cat([inputs[:, :, :idx, :, :], inputs[:, :, idx + 1:, :, :]], dim=2)
                    reconstructions_other = torch.cat([reconstructions[:, :, :idx, :, :], reconstructions[:, :, idx + 1:, :, :]], dim=2)
                    self._compute_and_log_psnr(inputs_other, reconstructions_other, prefix="train_other")
                    self._compute_and_log_ssim(inputs_other, reconstructions_other, prefix="train_other")
                    self._compute_and_log_lpips(inputs_other, reconstructions_other, prefix="train_other")

                if self.track_layers:
                    decoder_stats = self.layer_tracker.get_stats()
                    log_layer = {}
                    for name, value in sorted(decoder_stats.items()):
                        log_layer[f"layer_stats/{name}"] = value
                    self.log_dict(
                        log_layer,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True
                    )
                    self.layer_tracker.clear()
            return aeloss

        # Discriminator training requires manual optimization in Lightning 2.0+
        # If you need to re-enable discriminator, set self.automatic_optimization = False
        # and implement manual optimization with self.optimizers()
        raise NotImplementedError(
            "Discriminator training (use_discriminator=True) requires manual optimization in Lightning 2.0+. "
            "Set use_discriminator: False in your config, or implement manual optimization."
        )

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, idx, z = self(inputs)

        inputs_ref = inputs[:, :, idx:idx + 1, :, :].clone()
        inputs_other = torch.cat([inputs[:, :, :idx, :, :], inputs[:, :, idx + 1:, :, :]], dim=2)

        # Compute validation loss
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        if not self.val_begining:
            # Log all metrics from log_dict_ae
            for key, value in log_dict_ae.items():
                self.log(
                    key, 
                    value, 
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True
                )

            # Main validation loss
            self.log(
                "val/aeloss", 
                aeloss, 
                prog_bar=True, 
                logger=True, 
                on_step=True,
                on_epoch=True, 
                sync_dist=True
            )

            self._compute_and_log_psnr(inputs, reconstructions, prefix="val")
            self._compute_and_log_ssim(inputs, reconstructions, prefix="val")
            self._compute_and_log_lpips(inputs, reconstructions, prefix="val")

            rec_ref = reconstructions[:, :, idx:idx + 1, :, :].clone()
            self._compute_and_log_psnr(inputs_ref, rec_ref, prefix="val_ref")
            self._compute_and_log_ssim(inputs_ref, rec_ref, prefix="val_ref")
            self._compute_and_log_lpips(inputs_ref, rec_ref, prefix="val_ref")

            if inputs.size(2) != 1:
                reconstructions_other = torch.cat([reconstructions[:, :, :idx, :, :], reconstructions[:, :, idx + 1:, :, :]], dim=2)
                self._compute_and_log_psnr(inputs_other, reconstructions_other, prefix="val_other")
                self._compute_and_log_ssim(inputs_other, reconstructions_other, prefix="val_other")
                self._compute_and_log_lpips(inputs_other, reconstructions_other, prefix="val_other")

        if self.val_begining:
            self.val_begining = False 
            
        return aeloss

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluation on test dataset.
        Computes reconstruction metrics without discriminator loss.
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, idx, _ = self(inputs, sample_posterior=False)

        # Compute autoencoder loss (reconstruction + KL divergence)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="test",
        )

        # Only compute discriminator metrics if enabled
        if self.use_discriminator:
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="test",
            )
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # Move to CPU for logging
        reconstructions = reconstructions.cpu().detach()

        # Log all autoencoder metrics
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # Compute additional metrics
        with torch.no_grad():
            mse = F.mse_loss(reconstructions, inputs.cpu())
            self.log("test/mse", mse, prog_bar=True, logger=True, on_step=False, on_epoch=True)

            psnr = 10 * torch.log10(1.0 / mse)
            self.log("test/psnr", psnr, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {"test_loss": aeloss, "test_rec_loss": log_dict_ae["test/rec_loss"]}

    def configure_optimizers(self):
        lr = self.base_learning_rate

        # ========== Transformer / LoRA Parameters ==========
        transformer_params = []
        for param in self.transformer.parameters():
            if param.requires_grad:
                transformer_params.append(param)
        ref_conv_params = [
            p for p in self.ae.decoder.ref_conv_in.parameters() if p.requires_grad
        ]
        transformer_params.extend(ref_conv_params)

        # ========== VAE Parameters ==========
        ae_params = []
        ref_conv_param_ids = {id(p) for p in ref_conv_params}

        for param in self.ae.parameters():
            if param.requires_grad and id(param) not in ref_conv_param_ids:
                ae_params.append(param)

        # ========== Combine Parameter Groups ==========
        param_groups = []

        # Transformer/LoRA params use full learning rate
        # (LoRA can handle higher LR than full fine-tuning)
        if transformer_params:
            param_groups.append({"params": transformer_params, "lr": lr})

        # VAE params use reduced learning rate
        if ae_params:
            param_groups.append({"params": ae_params, "lr": lr * 0.1})

        # Create main optimizer with all parameter groups
        opt_ae = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
        )

        # warmup + cosine annealing scheduler
        warmup_steps = self.warmup_steps
        # Linear warmup from 0 to 1.0 over warmup_steps
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt_ae,
            start_factor=0.01,  # Start at 1% of lr
            end_factor=1.0,     # End at 100% of lr
            total_iters=warmup_steps
        )
        # Cosine annealing after warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ae,
            T_max=self.T_max - warmup_steps,  # Remaining steps after warmup
            eta_min=lr * 0.01  # Minimum learning rate (1% of base lr)
        )
        # Combine warmup and cosine
        scheduler_ae = torch.optim.lr_scheduler.SequentialLR(
            opt_ae,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        # Only create discriminator optimizer if enabled
        if self.use_discriminator:
            raise NotImplementedError(
                "Discriminator training requires manual optimization in Lightning 2.0+. "
                "Set use_discriminator: False in your config."
            )
        else:
            # Return single optimizer with scheduler
            return {"optimizer": opt_ae, "lr_scheduler": {"scheduler": scheduler_ae, "interval": "step"}}

    def get_last_layer(self):
        return self.ae.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        if not only_inputs:
            xrec, posterior, _, _ = self(x)

            # if self.use_reference and self.ae.decoder.use_reference:
            #     # Use the input's first frame as reference for random samples
            #     reference_frame = x[:, :, 0:1, :, :].clone()
            #     log["samples"] = self.decode(torch.randn_like(posterior.sample()), reference_frame=reference_frame)
            # else:
            #     log["samples"] = self.decode(torch.randn_like(posterior.sample()))

            xrec = xrec.cpu().detach()
            log["reconstructions"] = xrec

        x = x.cpu().detach()
        log["inputs"] = x
        return log

    # def train(self, mode=True):
    #     """Override to keep loss module in eval mode during training"""
    #     super().train(mode)
    #     print(f"[DEUBG] train being called")
    #     # CRITICAL: Loss module (especially perceptual networks) must stay in eval mode
    #     if hasattr(self, 'loss'):
    #         self.loss.eval()
    #     if hasattr(self, 'ae'):
    #         self.ae.train()
    #     return self
    
    def _compute_and_log_psnr(self, inputs, reconstructions, prefix="train"):
        """Helper to compute and log PSNR"""
        with torch.no_grad():
            # Convert to numpy
            inputs_np = inputs.detach().cpu().float().numpy()
            reconstructions_np = reconstructions.detach().cpu().float().numpy()
            
            # Convert from [-1, 1] to [0, 1]
            inputs_np = (inputs_np + 1.0) / 2.0
            reconstructions_np = (reconstructions_np + 1.0) / 2.0
            
            # Compute PSNR with data_range=1.0
            psnr_value = psnr(inputs_np, reconstructions_np, data_range=1.0)
            
            if self.val_begining:
                self.log(
                    f"{prefix}/psnr",
                    psnr_value,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            else:    
                self.log(
                    f"{prefix}/psnr",
                    psnr_value,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True
                )
            
            return psnr_value
    
    def _compute_and_log_ssim(self, inputs, reconstructions, prefix="train"):
        """Helper to compute and log SSIM"""
        with torch.no_grad():
            # Convert to numpy
            inputs_np = inputs.detach().cpu().float().numpy()
            reconstructions_np = reconstructions.detach().cpu().float().numpy()
            
            # Normalize
            inputs_np = (inputs_np + 1.0) / 2.0
            reconstructions_np = (reconstructions_np + 1.0) / 2.0
            
            # Reshape: (B, C, T, H, W) -> (B*T, H, W, C)
            b, c, t, h, w = inputs_np.shape
            ssim_values = []
        
            # Process each frame individually
            for batch_idx in range(b):
                for time_idx in range(t):
                    # Extract and transpose: [C, H, W] -> [H, W, C]
                    img1 = inputs_np[batch_idx, :, time_idx, :, :].transpose(1, 2, 0)
                    img2 = reconstructions_np[batch_idx, :, time_idx, :, :].transpose(1, 2, 0)
                    
                    # Adaptive window size based on image dimensions
                    min_dim = min(h, w)
                    win_size = 3 if min_dim < 7 else (7 if min_dim < 11 else 11)
                    
                    frame_ssim = ssim(
                        img1, img2, 
                        data_range=1.0,
                        win_size=win_size,
                        channel_axis=-1,
                        multichannel=True
                    )
                    ssim_values.append(frame_ssim)
            
            # Average across all frames
            ssim_value = sum(ssim_values) / len(ssim_values)
            
            if self.val_begining:
                self.log(
                    f"{prefix}/ssim",
                    ssim_value,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            else:    
                self.log(
                    f"{prefix}/ssim",
                    ssim_value,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True
                )
            
            return ssim_value

    def _compute_and_log_lpips(self, inputs, reconstructions, prefix="train"):
        """Helper to compute and log LPIPS"""
        with torch.no_grad():
            # Initialize LPIPS on correct device
            self.loss_fn = self.loss_fn.to(inputs.device)
            
            # Process frame by frame
            b, c, t, h, w = inputs.shape
            lpips_values = []
            
            for frame_idx in range(t):
                input_frame = inputs[:, :, frame_idx, :, :]  # [B, C, H, W]
                recon_frame = reconstructions[:, :, frame_idx, :, :]
                
                lpips_frame = self.loss_fn(input_frame, recon_frame).mean()
                lpips_values.append(lpips_frame.item())
            
            # Average across frames
            lpips_value = sum(lpips_values) / len(lpips_values)
            
            if self.val_begining:
                self.log(
                    f"{prefix}/lpips",
                    lpips_value,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            else:    
                self.log(
                    f"{prefix}/lpips",
                    lpips_value,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True
                )
            
            return lpips_value

    def on_train_start(self):
        """
        Called at the beginning of training by PyTorch Lightning.
        By this point, all checkpoint loading is complete.
        """
        if self.track_layers:
            self.layer_tracker = MinimalDecoderTracker(self.ae.decode, self.transformer.transformer)
        
        print("\n" + "=" * 60)
        if self._checkpoint_loaded:
            print(f"\n✓ Checkpoint loaded successfully")
            print(f"  - Via ckpt_path in config: {self._loading_from_ckpt_path}")
            print(f"  - Via --resume_from_checkpoint: {not self._loading_from_ckpt_path}")
            print(f"  - Current global step: {self.global_step}")
        else:
            print(f"\n✗ No checkpoint loaded - starting from pretrained")
            print(f"  - Current global step: {self.global_step} (should be 0)")
            
        if self.pretrained:
            print(f"  - Transformer: start with a pretrained model")
        else:
            print(f"  - Transformer: start with a new model")

        if self.reusing:
            print(f"  - Transformer: reusing same weight every stage")

        if self.use_lora:
            print(f"  - LoRA: Initialized with rank={self.lora_rank}")
        else:
            print(f"  - LoRA: Not using LoRA fine-tuning")
        
        if self.gradient_checkpointing:
            print(f"  - Gradient Checkpointing: {self.gradient_checkpointing}")
        else:
            print(f"  - Gradient Checkpointing: Not using gradient checkpointing")
        print("\n" + "=" * 60)
        
        # self.print_parameter_status() 

    def _freeze_encoder(self):
        """Freeze the encoder parameters"""
        # Freeze all encoder parameters
        for param in self.ae.encoder.parameters():
            param.requires_grad = False
    
    def _freeze_quant_conv(self):
        # Freeze quant_conv
        for param in self.ae.quant_conv.parameters():
            param.requires_grad = False

        # Freeze post_quant_conv
        for param in self.ae.post_quant_conv.parameters():
            param.requires_grad = False
        
        print("Quant Conv and Post Quant Conv frozen")
    
    def _freeze_decoder(self):
        """
        Freeze all decoder parameters,
        """
        for param in self.ae.decoder.parameters():
            param.requires_grad = False
             
        print("Decoder frozen.")

    def on_train_end(self):
        """Called at the end of training"""
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        if self.use_lora:
            print("\n💡 Tip: Save LoRA weights with:")
            print("   model.save_lora_weights('path/to/lora_weights.pth')")
            print(f"\n  LoRA checkpoint is ~{self.lora_rank * 4 * 6 * 30 / 1e6:.1f}MB")
            print(f"  vs full checkpoint ~{sum(p.numel() for p in self.parameters()) * 4 / 1e9:.1f}GB")

        print("=" * 60 + "\n")