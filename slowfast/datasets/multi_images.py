#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.utils.data

import slowfast.utils.logging as logging
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Multi_images(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.doube_image_dataset_length = True
        self.ds_names = cfg.IMAGE_TRAIN.DATASETS
        self.datasets = [
            DATASET_REGISTRY.get(name.capitalize())(cfg, mode) for name in self.ds_names
        ]
        assert len(self.datasets) > 0, "No datasets given in the config"
        self.lengths = [len(d) for d in self.datasets]
        self.offsets = [sum(self.lengths[:i]) for i in range(len(self.lengths))] + [float("inf")]

        self.max_samples = -1
        self.step = len(self.lengths) // self.max_samples
        assert self.max_samples < sum(len(d) for d in self.datasets), f"NUM_IMAGES ({self.max_samples}) is larger than the total number of samples ({sum(len(d) for d in self.datasets)})"

    def get_hash(self, n, limit):
        return hash(f"{n}_hands") % limit
    def __getitem__(self, index):
        if self.doube_image_dataset_length:
            index = index % sum(self.lengths)

        if self.max_samples > 0:
            orig_index = index
            index = orig_index * self.step
            assert index < sum(len(d) for d in self.datasets), f"index ({index}) is larger than the total number of samples ({sum(len(d) for d in self.datasets)})"

        for data_idx in range(len(self.datasets)):
            if index < self.offsets[data_idx + 1]: break
        offset = self.offsets[data_idx]
        index = index - offset
        return self.datasets[data_idx][index]

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        if self.max_samples > 0:
            ret = self.max_samples
        else:
            ret = sum(len(d) for d in self.datasets)
        if self.doube_image_dataset_length:
            ret = ret * 100
        return ret