import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.common_utils import instantiate_from_config


class LatentImageDataset(Dataset):
    """
    Dataset that loads pre-computed latents and their paired reference images
    from a single (pre-merged) CSV file.

    CSV must have columns:
        latent_path          — path to .pt file containing {"latents": tensor [1, 16, T, H, W], ...}
        resized_image_path   — path to PNG reference image

    Each item returns:
        {
            "latent":    FloatTensor [16, T, H_lat, W_lat]  — raw encoder latent
            "ref_image": FloatTensor [3, 1, H, W]           — reference image, [-1, 1]
            "mode":      int 1
        }
    """

    def __init__(self, csv_file: str, resolution=(480, 832)):
        self.resolution = resolution
        self.metadata = pd.read_csv(csv_file)
        print(f"[LatentImageDataset] Loaded {len(self.metadata)} samples from {csv_file}")

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load PNG, resize to (H, W), normalize to [-1, 1]. Returns [3, H, W]."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution[1], self.resolution[0]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        arr = (arr - 0.5) * 2.0                        # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    def __getitem__(self, index):
        while True:
            try:
                row = self.metadata.iloc[index]
                data = torch.load(row["latent_path"], map_location="cpu")
                latent = data["latents"].squeeze(0).float()   # [16, T, H_lat, W_lat]
                ref_image = self._load_image(row["resized_image_path"])  # [3, H, W]
                ref_image = ref_image.unsqueeze(1)                       # [3, 1, H, W]
                return {"latent": latent, "ref_image": ref_image, "mode": 1}
            except Exception as e:
                print(f"[LatentImageDataset] Failed at index {index}: {e}")
                index = (index + 1) % len(self.metadata)


class CombinedLatentVideoDataset(Dataset):
    """
    Interleaves a video dataset (mode=0) and a latent dataset (mode=1).

    Even dataset indices → video sample   (mode=0)
    Odd  dataset indices → latent sample  (mode=1)

    The model's training_step checks batch["mode"] to select the training path,
    achieving roughly 50 % video / 50 % latent over the course of training.

    Use combined_collate_fn as the DataLoader's collate_fn.
    """

    def __init__(self, video_dataset: Dataset, latent_dataset: Dataset):
        self.video_dataset = video_dataset
        self.latent_dataset = latent_dataset

    def __len__(self):
        return 2 * max(len(self.video_dataset), len(self.latent_dataset))

    def __getitem__(self, idx):
        if idx % 2 == 0:
            video_idx = (idx // 2) % len(self.video_dataset)
            item = self.video_dataset[video_idx]
            item["mode"] = 0
            return item
        else:
            latent_idx = (idx // 2) % len(self.latent_dataset)
            return self.latent_dataset[latent_idx]

def combined_collate_fn(batch):
    """
    Collate for CombinedLatentVideoDataset.

    All items in a batch share the same mode (guaranteed by
    AlternatingModeBatchSampler), so every item has the same keys.
    Just stack tensors and pass through the mode int.
    """
    if not batch:
        return {}

    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values[0]  # same value for all items (e.g. mode int)
    return result


class AlternatingModeBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches that are entirely video (mode=0) OR entirely latent (mode=1),
    strictly alternating: video batch, latent batch, video batch, ...

    Assumes the combined dataset lays out indices as:
        even indices → video samples
        odd  indices → latent samples
    (which is what CombinedLatentVideoDataset* guarantee)
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        import math
        self.batch_size = batch_size
        self.drop_last = drop_last
        n = len(dataset)
        self.video_indices = list(range(0, n, 2))   # even → video
        self.latent_indices = list(range(1, n, 2))  # odd  → latent

    def _make_batches(self, indices):
        import random
        idx = indices.copy()
        random.shuffle(idx)
        batches = [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self):
        video_batches = self._make_batches(self.video_indices)
        latent_batches = self._make_batches(self.latent_indices)

        # Interleave: video, latent, video, latent, ...
        for v_batch, l_batch in zip(video_batches, latent_batches):
            yield v_batch
            yield l_batch

        # Yield any remaining batches from the longer list
        n = min(len(video_batches), len(latent_batches))
        for b in video_batches[n:]:
            yield b
        for b in latent_batches[n:]:
            yield b

    def __len__(self):
        import math
        if self.drop_last:
            n_v = len(self.video_indices) // self.batch_size
            n_l = len(self.latent_indices) // self.batch_size
        else:
            n_v = math.ceil(len(self.video_indices) / self.batch_size)
            n_l = math.ceil(len(self.latent_indices) / self.batch_size)
        return n_v + n_l


class CombinedLatentVideoDatasetFromConfig(Dataset):
    """
    Config-friendly version of CombinedLatentVideoDataset.

    Accepts standard instantiate_from_config dicts for both sub-datasets
    so it can be used directly as the train target in a YAML config.

    Example config:
        target: data.latent_dataset.CombinedLatentVideoDatasetFromConfig
        params:
          video_dataset_config:
            target: data.dataset.DatasetVideoLoader
            params: { ... }
          latent_dataset_config:
            target: data.latent_dataset.LatentImageDataset
            params: { csv_file: /path/to/metadata_merged.csv }
    """

    def __init__(self, video_dataset_config: dict, latent_dataset_config: dict):
        self.video_dataset = instantiate_from_config(video_dataset_config)
        self.latent_dataset = instantiate_from_config(latent_dataset_config)
        print(
            f"[CombinedLatentVideoDataset] video={len(self.video_dataset)}, "
            f"latent={len(self.latent_dataset)}, total={len(self)}"
        )

    def __len__(self):
        return 2 * max(len(self.video_dataset), len(self.latent_dataset))

    def __getitem__(self, idx):
        if idx % 2 == 0:
            video_idx = (idx // 2) % len(self.video_dataset)
            item = self.video_dataset[video_idx]
            item["mode"] = 0
            return item
        else:
            latent_idx = (idx // 2) % len(self.latent_dataset)
            return self.latent_dataset[latent_idx]
