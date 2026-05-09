from functools import partial
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import argparse, os, sys, glob

os.chdir(sys.path[0])
sys.path.append("..")

from utils.common_utils import instantiate_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    mode = "image" if worker_id < worker_info.num_workers * 0.2 else "video"
    print(f"Mode is {mode}")
    dataset.set_mode(mode)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        img_video_joint_train=False,
        shuffle_val_dataloader=False,
        train_img=None,
        test_max_n_samples=None,
        collate_fn=None,
        use_alternating_batch_sampler=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = img_video_joint_train
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        # train 2 dataset
        # if img_loader is not None:
        #     img_data = instantiate_from_config(img_loader)
        #     img_data.setup()
        if train_img is not None:
            if train_img["params"]["batch_size"] == -1:
                train_img["params"]["batch_size"] = (
                    batch_size * train["params"]["video_length"]
                )
                print(
                    "Set train_img batch_size to {}".format(
                        train_img["params"]["batch_size"]
                    )
                )
            img_data = instantiate_from_config(train_img)
            self.img_loader = img_data.train_dataloader()
        else:
            self.img_loader = None
        self.wrap = wrap
        self.test_max_n_samples = test_max_n_samples
        self.use_alternating_batch_sampler = use_alternating_batch_sampler

        # collate_fn: accepts a dotted import path string or None
        if collate_fn is not None:
            import importlib
            module_path, fn_name = collate_fn.rsplit(".", 1)
            self.collate_fn = getattr(importlib.import_module(module_path), fn_name)
        else:
            self.collate_fn = None

    def prepare_data(self):
        # for data_cfg in self.dataset_configs.values():
        #     instantiate_from_config(data_cfg)
        pass

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        if self.use_alternating_batch_sampler:
            from data.latent_dataset import AlternatingModeBatchSampler
            batch_sampler = AlternatingModeBatchSampler(
                self.datasets["train"],
                batch_size=self.batch_size,
                drop_last=True,
            )
            loader = DataLoader(
                self.datasets["train"],
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                worker_init_fn=init_fn,
                collate_fn=self.collate_fn,
            )
        else:
            loader = DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                worker_init_fn=init_fn,
                collate_fn=self.collate_fn,
            )
        if self.img_loader is not None:
            return {"loader_video": loader, "loader_img": self.img_loader}
        else:
            return loader

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        if self.test_max_n_samples is not None:
            dataset = torch.utils.data.Subset(
                self.datasets["test"], list(range(self.test_max_n_samples))
            )
        else:
            dataset = self.datasets["test"]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )
