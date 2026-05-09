import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import pandas as pd
from PIL import Image


class DatasetVideoLoader(Dataset):
    """
    Dataset for loading videos from a CSV file.
    CSV file contains a 'path' column for the path to the video file.
    """

    def __init__(
        self,
        csv_file,
        resolution,
        video_length,
        target_fps=15,
        subset_split="all",
        clip_length=1.0,
        mode="video",
        input_key="path",
        reference_key=None,
        style_gt_key=None,
    ):
        self.csv_file = csv_file
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.target_fps = target_fps
        self.clip_length = clip_length
        self.mode = mode
        self.input_key = input_key
        self.reference_key = reference_key
        self.style_gt_key = style_gt_key

        assert self.subset_split in ["train", "test", "val", "all"]
        self.video_exts = ["avi", "mp4", "webm", "mov", "mkv"]
        self.image_exts = ["jpg", "jpeg", "png", "bmp", "tiff"]

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]

        # Load dataset from CSV file
        self._make_dataset()

    def _make_dataset(self):
        """
        Load video paths and captions from the CSV file.
        """
        self.videos = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.videos)} videos from {self.csv_file}")

        if self.subset_split == "val":
            self.videos = self.videos[-300:]
        elif self.subset_split == "train":
            self.videos = self.videos[:-300]
        elif self.subset_split == "test":
            self.videos = self.videos[-30:]

        print(f"Number of videos = {len(self.videos)}")

        # Create video indices for image mode
        self.video_indices = list(range(len(self.videos)))
        self._validate_columns()

    def _validate_columns(self):
        required = [self.input_key]
        if self.reference_key is not None:
            required.append(self.reference_key)
        if self.style_gt_key is not None:
            required.append(self.style_gt_key)

        missing = [col for col in required if col not in self.videos.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in CSV {self.csv_file}. "
                f"Available columns: {list(self.videos.columns)}"
            )

    def set_mode(self, mode):
        self.mode = mode

    def _get_video_path(self, index):
        return self.videos.iloc[index][self.input_key]

    def __getitem__(self, index):
        if self.mode == "image":
            return self.__getitem__images(index)
        else:
            return self.__getitem__video(index)

    def __getitem__video(self, index):
        while True:
            row = self.videos.iloc[index]
            video_path = row[self.input_key]

            try:
                frames = self._load_clip_from_path(video_path, self.video_length, use_stride=True)
                # Try to load optional clips - if they fail, skip this sample
                reference_clip = self._load_optional_clip(row, self.reference_key, raise_on_error=True)
                style_gt_clip = self._load_optional_clip(row, self.style_gt_key, raise_on_error=True)
                break
            except Exception as e:
                print(f"Load media failed! path = {video_path}, error: {str(e)}")
                index = (index + 1) % len(self.videos)
                continue

        sample = {"video": frames, "is_video": True}
        # Always include keys for consistency in batch collation
        if self.reference_key is not None:
            sample["reference"] = reference_clip
        if self.style_gt_key is not None:
            sample["style_gt"] = style_gt_clip
        return sample

    def __getitem__images(self, index):
        while True:
            frames_list = []
            try:
                for i in range(self.video_length):
                    # Get a unique video for each frame
                    video_index = (index + i) % len(self.video_indices)
                    video_path = self._get_video_path(video_index)

                    if self._is_image_file(video_path):
                        frame_tensor = self._load_image_as_tensor(video_path).unsqueeze(0)
                    else:
                        video_reader = VideoReader(
                            video_path,
                            ctx=cpu(0),
                            width=self.resolution[1],
                            height=self.resolution[0],
                        )
                        rand_idx = random.randint(0, len(video_reader) - 1)
                        frame = video_reader[rand_idx]
                        frame_tensor = (
                            torch.tensor(frame.asnumpy()).permute(2, 0, 1).float().unsqueeze(0)
                        )

                    frames_list.append(frame_tensor)

                frames = torch.cat(frames_list, dim=0)
                frames = (frames / 255 - 0.5) * 2
                frames = frames.permute(1, 0, 2, 3)
                assert (
                    frames.shape[2] == self.resolution[0]
                    and frames.shape[3] == self.resolution[1]
                ), f"frame={frames.shape}, self.resolution={self.resolution}"

                row = self.videos.iloc[index]
                # Try to load optional clips - if they fail, skip this sample
                reference_clip = self._load_optional_clip(row, self.reference_key, raise_on_error=True)
                style_gt_clip = self._load_optional_clip(row, self.style_gt_key, raise_on_error=True)
                break
            except Exception as e:
                print(f"Load media failed! index = {index}, error = {e}")
                index = (index + 1) % len(self.video_indices)
                continue

        data = {"video": frames, "is_video": False}
        # Always include keys for consistency in batch collation
        if self.reference_key is not None:
            data["reference"] = reference_clip
        if self.style_gt_key is not None:
            data["style_gt"] = style_gt_clip
        return data

    def __len__(self):
        return len(self.videos)

    def _load_clip_from_path(self, video_path, clip_length, use_stride=False):
        if self._is_image_file(video_path):
            return self._load_image_clip(video_path, clip_length)
        
        video_reader = VideoReader(
            video_path,
            ctx=cpu(0),
            width=self.resolution[1],
            height=self.resolution[0],
        )
        total_frames = len(video_reader)
        if total_frames == 0:
            raise ValueError(f"Video {video_path} contains no frames.")

        clip_length = clip_length if clip_length is not None else self.video_length
        if use_stride:
            source_fps = video_reader.get_avg_fps()
            stride = max(1, round(source_fps / self.target_fps))

            frame_indices = list(range(0, total_frames, stride))
            if len(frame_indices) < clip_length:
                frame_indices = list(range(0, total_frames))
        else:
            frame_indices = list(range(0, total_frames))

        while len(frame_indices) < clip_length:
            frame_indices.append(frame_indices[-1])

        start = random.randint(0, len(frame_indices) - clip_length)
        frame_indices = frame_indices[start : start + clip_length]

        frames = video_reader.get_batch(frame_indices)
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frames = (frames / 255 - 0.5) * 2
        return frames

    def _load_optional_clip(self, row, column_name, raise_on_error=False):
        if column_name is None or column_name not in row:
            return None
        video_path = row[column_name]
        if isinstance(video_path, float) and pd.isna(video_path):
            return None
        try:
            clip = self._load_clip_from_path(video_path, None)
            return clip
        except Exception as e:
            print(f"Failed to load clip from {video_path}: {e}")
            if raise_on_error:
                raise
            return None

    def _is_image_file(self, path):
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        return ext in self.image_exts

    def _load_image_as_tensor(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.resolution[1], self.resolution[0]))
        np_img = np.array(image)
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).float()
        tensor = (tensor / 255 - 0.5) * 2
        return tensor

    def _load_image_clip(self, path, clip_length):
        clip_len = clip_length if clip_length is not None else self.video_length
        tensor = self._load_image_as_tensor(path).unsqueeze(1)  # c,1,h,w
        tensor = tensor.repeat(1, clip_len, 1, 1)
        return tensor
