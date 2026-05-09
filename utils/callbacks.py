import os
import shutil
import time
import logging

mainlogger = logging.getLogger("mainlogger")

import torch
import wandb
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images=8,
        clamp=True,
        rescale=True,
        save_dir=None,
        to_local=False,
        log_images_kwargs=None,
    ):
        print("ImageLogger: will log %d images every %d batches" % (max_images, batch_frequency))
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val"), exist_ok=True)

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=10):
        """log images and videos to tensorboard or wandb"""
        global_step = pl_module.global_step
        
        # Check if using WandB
        is_wandb = hasattr(pl_module.logger, 'experiment') and hasattr(pl_module.logger.experiment, 'log')
        
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s" % (global_step, split, filename, key)
            
            if isinstance(value, list) and isinstance(value[0], str):
                captions = " |------| ".join(value)
                if is_wandb:
                    pl_module.logger.experiment.log({tag: captions}, step=global_step)
                else:
                    pl_module.logger.experiment.add_text(
                        tag, captions, global_step=global_step
                    )
                    
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                # Skip video logging for wandb to prevent saving videos
                # Log videos to wandb
                video = value
                n = video.shape[0]
                if video.shape[1] != 1 and video.shape[1] != 3:
                    continue
                video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
                frame_grids = [
                    torchvision.utils.make_grid(framesheet, nrow=int(n))
                    for framesheet in video
                ]  # [3, n*h, 1*w]
                grid = torch.stack(
                    frame_grids, dim=0
                )  # stack in temporal dim [t, 3, n*h, w]

                if self.rescale:
                    grid = (grid + 1.0) / 2.0
                if self.clamp:
                    grid = torch.clamp(grid, 0.0, 1.0)
                
                if is_wandb:
                    # Convert to format wandb expects: (T, H, W, C) with values 0-255
                    grid = (grid * 255).to(torch.uint8).cpu().numpy()
                    pl_module.logger.experiment.log({
                        tag: wandb.Video(grid, fps=save_fps, format="mp4")
                    }, step=global_step)
                else:
                    grid = grid.unsqueeze(dim=0)
                    pl_module.logger.experiment.add_video(
                        tag, grid, fps=save_fps, global_step=global_step
                    )
                    
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n))
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                
                if is_wandb:
                    grid_numpy = grid.permute(1, 2, 0).cpu().numpy()  # h, w, c
                    pl_module.logger.experiment.log({
                        tag: wandb.Image(grid_numpy)
                    }, step=global_step)
                else:
                    pl_module.logger.experiment.add_image(
                        tag, grid, global_step=global_step
                    )
            else:
                pass

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        """generate images, then save and log to tensorboard"""
        skip_freq = self.batch_freq if split == "train" else 5
        if (batch_idx + 1) % skip_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                log_func = pl_module.log_images
                batch_logs = log_func(batch, split=split, **self.log_images_kwargs)

            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()

            filename = "ep{}_idx{}_rank{}".format(
                pl_module.current_epoch, batch_idx, pl_module.global_rank
            )
            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..." % (split, filename))
                filename = "gs{}_".format(pl_module.global_step) + filename
                log_local(
                    batch_logs,
                    os.path.join(self.save_dir, split),
                    filename,
                    save_fps=10,
                )
            else:
                mainlogger.info(
                    "Log [%s] batch <%s> to tensorboard ..." % (split, filename)
                )
                self.log_to_tensorboard(
                    pl_module, batch_logs, filename, split, save_fps=10
                )
            mainlogger.info("Finish!")

            if is_train:
                pl_module.train()


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


"""
class DataModeSwitcher(Callback):
    def on_epoch_start(self, trainer, pl_module):
        mode = 'image' if random.random() <= 0.3 else 'video'
        trainer.datamodule.dataset.set_mode(mode)
        if trainer.global_rank == 0:
            torch.distributed.barrier()
"""


class KeepLatestCheckpoints(Callback):
    """Delete older checkpoints every N steps, keeping only the K most recent."""

    def __init__(self, ckpt_dir, keep_k=2, cleanup_every_n_steps=1000):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.step_ckpt_dir = os.path.join(ckpt_dir, "trainstep_checkpoints")
        self.keep_k = keep_k
        self.cleanup_every_n_steps = cleanup_every_n_steps

    @rank_zero_only
    def _cleanup(self, target_dir, keep_k=None):
        keep_k = keep_k if keep_k is not None else self.keep_k
        if not os.path.isdir(target_dir):
            return
        ckpts = [f for f in os.listdir(target_dir) if f.endswith(".ckpt")]
        if keep_k > 0 and len(ckpts) <= keep_k:
            return
        ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(target_dir, f)))
        to_delete = ckpts if keep_k == 0 else ckpts[:-keep_k]
        for fname in to_delete:
            path = os.path.join(target_dir, fname)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            mainlogger.info(f"KeepLatestCheckpoints: removed old checkpoint {path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.cleanup_every_n_steps > 0 and trainer.global_step % self.cleanup_every_n_steps == 0:
            self._cleanup(self.step_ckpt_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        self._cleanup(self.ckpt_dir, keep_k=0)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update - use strategy.root_device for Lightning 1.7+ and 2.x
        gpu_index = trainer.strategy.root_device.index
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        gpu_index = trainer.strategy.root_device.index
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
