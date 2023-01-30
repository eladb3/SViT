#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
from pexpect import ExceptionPexpect
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler

from . import utils as utils
from .build import build_dataset

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    if isinstance(labels[0], dict):
        labels = {k:torch.tensor(np.concatenate([l[k] for l in labels], axis=0)).float() for k in labels[0].keys()}
    else:
        labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key in ["boxes", "ori_boxes"]:
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]

    if split in ["train"]:
        return construct_loader_train(cfg, split)
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    if split == 'val' and cfg.DATA_LOADER.NUM_WORKERS_VAL > -1:
        num_workers = cfg.DATA_LOADER.NUM_WORKERS_VAL
        
    persistent_workers = cfg.DATA_LOADER.PERSISTENT_WORKERS and num_workers > 0
    if (split == 'train') and (num_workers > 0) and (not persistent_workers):
        persistent_workers = cfg.DATA_LOADER.PERSISTENT_WORKERS_TRAIN
    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
            persistent_workers = persistent_workers,
        )
    else:
        if (
            cfg.MULTIGRID.SHORT_CYCLE
            and split in ["train"]
            and not is_precise_bn
        ):
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            batch_sampler = ShortCycleBatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
            )
            # Create a loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
                persistent_workers = persistent_workers,
            )
        else:
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            # Create a loader
            if cfg.DETECTION.ENABLE:
                collate_func = detection_collate
            elif cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
                collate_func = partial(
                    multiple_samples_collate, fold="imagenet" in dataset_name
                )
            else:
                collate_func = None

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(False if sampler else shuffle),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=drop_last,
                collate_fn=collate_func,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
                persistent_workers = persistent_workers,
            )
    return loader

def construct_loader_train(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train"]
    assert cfg.NUM_GPUS > 1, f"NUM_GPUS must be greater than 1 for multigpus"
    from slowfast.utils import distributed as du
    local_rank = du.get_local_rank()
    
    img_gpus = list(sorted(cfg.IMAGE_TRAIN.GPU_IDS))
    vid_gpus = list(sorted([i for i in range(cfg.NUM_GPUS) if i not in img_gpus]))
    
    is_image = local_rank in cfg.IMAGE_TRAIN.GPU_IDS
    local_rank_data = img_gpus.index(local_rank) if is_image else vid_gpus.index(local_rank)
    
    video_dataset, image_dataset = cfg.TRAIN.DATASET, 'multi_images'
    dataset_name = image_dataset if is_image else video_dataset

    video_bs, image_bs = cfg.TRAIN.BATCH_SIZE, cfg.IMAGE_TRAIN.BATCH_SIZE
    bs = video_bs if (not is_image) else image_bs
    num_gpus = len(cfg.IMAGE_TRAIN.GPU_IDS) if is_image else (cfg.NUM_GPUS - len(cfg.IMAGE_TRAIN.GPU_IDS))
    batch_size = int(bs / max(1, num_gpus))

    if cfg.DETECTION.ENABLE and not is_image:
        collate_func = detection_collate
    else:
        collate_func = None

    shuffle = True
    drop_last = True
    print(f"local_rank: {local_rank}, is_image: {is_image}, dataset: {dataset_name}, bs: {batch_size}, local_rank_data: {local_rank_data}", flush=True)
    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    if split == 'val' and cfg.DATA_LOADER.NUM_WORKERS_VAL > -1:
        num_workers = cfg.DATA_LOADER.NUM_WORKERS_VAL
        
    persistent_workers = cfg.DATA_LOADER.PERSISTENT_WORKERS and num_workers > 0
    if (split == 'train') and (num_workers > 0) and (not persistent_workers):
        persistent_workers = cfg.DATA_LOADER.PERSISTENT_WORKERS_TRAIN
    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_func,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
            persistent_workers = persistent_workers,
        )
    else:
        # Create a sampler for multi-process training
        from torch.utils.data.distributed import DistributedSampler
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        sampler = DistributedSampler(
            dataset,
            rank=local_rank_data,
            num_replicas=len(img_gpus if is_image else vid_gpus),
            shuffle=shuffle,
            drop_last=drop_last
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_func,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
            persistent_workers = persistent_workers,
        )

    return loader

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "sampler"):
            sampler = loader.batch_sampler.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.batch_sampler.sampler
            if isinstance(loader.batch_sampler, (ShortCycleBatchSampler,))
            else loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, (DistributedSampler,)):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
