import os
import torch
import hydra
import wandb

import torch.nn as nn

from hydra import instantiate
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf, DictConfig


class MMVTODataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.current_epoch = 0

    def setup(self, stage=None):
        self.train_dataset = instantiate(self.config.data.train)
        self.val_dataset = instantiate(self.config.data.val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def on_epoch_start(self):
        self.current_epoch += 1
        resolution = self.get_resolution(self.current_epoch)
        self.train_dataset.set_resolution(resolution)
        self.val_dataset.set_resolution(resolution)

    def get_resolution(self, epoch):
        # Implement logic to determine resolution based on current epoch
        pass

