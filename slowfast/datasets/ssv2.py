#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
from itertools import chain as chain
import torch
import torch.utils.data
from torchvision import transforms
from iopath.common.file_io import g_pathmgr
import torch.nn.functional as F
import traceback
import slowfast.utils.logging as logging

from .ssv2_MF import autoaugment as autoaugment
from .ssv2_MF import transform as transform
from .ssv2_MF import utils as utils_MF
from .build import DATASET_REGISTRY
from .transform import create_random_augment
from .random_erasing import RandomErasing
from . import utils as utils

from slowfast.utils import box_ops
from functools import partial


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ssv2(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """

        dataroot = cfg.SSV2.DATA_ROOT
        assert os.path.isdir(dataroot), f"{dataroot} does not exists"
        self.data_root = dataroot
        # File list
        pfilelist = os.path.join(dataroot, 'frames_list.json')
        if os.path.isfile(pfilelist):
            with open(pfilelist, 'rt') as f:
                self.file_list = json.load(f)
        else:
            self.file_list = None
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        # Loading labels.
        split = self.cfg.SSV2.SPLIT
        dataroot = self.data_root
        data_split = self.mode
        self.data_split = data_split
        # dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        if split == 'compositional':
            self.file_labels = 'data/ssv2/dataset_splits/compositional/labels.json'
            label_file = f'data/ssv2/dataset_splits/compositional/{"train" if data_split == "train" else "validation"}.json'
        elif split == 'standard':
            self.file_labels = f'{dataroot}/sm/annotations/something-something-v2-labels.json'
            label_file = f'{dataroot}/json_files/something-something-v2-{"train" if data_split == "train" else "validation"}.json'
        elif split == 'fewshot-base':
            self.file_labels = f'data/ssv2/dataset_splits/fewshot/base_labels.json'
            label_file = f'data/ssv2/dataset_splits/fewshot/base_{"training" if data_split == "train" else "validation"}_set.json'
        elif split == 'fewshot-5finetune':
            self.file_labels = f'data/ssv2/dataset_splits/fewshot/finetune_labels.json'
            label_file = f'data/ssv2/dataset_splits/fewshot/finetune_5shot_{"training" if data_split == "train" else "validation"}.json'
        elif split == 'fewshot-10finetune':
            self.file_labels = f'data/ssv2/dataset_splits/fewshot/finetune_labels.json'
            label_file = f'data/ssv2/dataset_splits/fewshot/finetune_10shot_{"training" if data_split == "train" else "validation"}.json'
        else:
            raise NotImplementedError(f"split = {split}")

        # Loading label names.
        with g_pathmgr.open(self.file_labels,"r") as f:
            label_dict = json.load(f)

        with g_pathmgr.open(label_file, "r") as f:
            label_json = json.load(f)

        with open('data/ssv2/empty_bbox_{}.json'.format('train' if data_split == 'train' else 'val'), 'r') as f:
            sort_out = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            if video_name in sort_out: continue
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            self._video_names.append(video_name)
            self._labels.append(label)

        path_to_file = os.path.join(
            "datafiles", "ssv2", "my", # self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format("train" if self.data_split == "train" else "val"),
        )
        path_to_file = label_file

        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos, _ = utils_MF.ssv2_load_image_lists(
            path_to_file, prefix=self.cfg.DATA.PATH_PREFIX, sort_out = sort_out,
        )

        assert len(self._path_to_videos) == len(self._video_names), (
            len(self._path_to_videos),
            len(self._video_names),
        )

        # From dict to list.
        new_paths, new_labels = [], []
        for index in range(len(self._video_names)):
            if self._video_names[index] in self._path_to_videos:
                new_paths.append(self._path_to_videos[self._video_names[index]])
                new_labels.append(self._labels[index])

        self._labels = new_labels
        self._path_to_videos = new_paths

        # Extend self when self._num_clips > 1 (during testing).
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._video_names = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_names]
            )
        )

        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )
        logger.info(
            "Something-Something V2 dataloader constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, index, video_length):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        metadata = {}
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1 # center crop
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        label = self._labels[index]

        fpaths = self.get_frames_list(index)


        frames = torch.as_tensor(
            utils.retry_load_images(
                fpaths,
                self._num_retries,
            )
        ) # [T, H, W, C]
        boxes = None
        if self.aug:
            frames, boxes = self._aug_frame(
                frames,
                spatial_sample_index,
                min_scale,
                max_scale,
                crop_size,
                boxes = boxes,
            ) # [C, T, H, W] vsave(frames.permute(3,0,1,2)[[2,1,0], ], torch.zeros(16,0,4))
        else:
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)

            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames, boxes=boxes,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

        frames = utils.pack_pathway_output(self.cfg, frames)

        return frames, label, index, metadata # vsave(frames[0][[2,1,0], ], torch.zeros(16,0,4)), vsave(frames[0][[2,1,0]], boxes)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
        boxes=None,
    ):

        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
            with_boxes = False,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2) # [T, C, H, W]
        list_img = self._frame_to_list_img(frames) # [ [C, H, W] ... ]
        list_img = aug_transform(
            list_img,
            boxes=boxes,
        )
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1) # [T, H, W, C]

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )

        if boxes is not None:
            orig_shape = boxes.shape
            boxes = boxes.reshape([-1, 4])

        spatial_sampling_dict = {}
        frames = utils.spatial_sampling( # vsave(frames[[2,1,0]], torch.from_numpy(boxes.reshape(16, 5, 4)))
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                        if self.mode in ["train"]
                        else False,
            boxes=boxes,
            rand_params = spatial_sampling_dict,
        )

        if boxes is not None:
            frames, boxes = frames
            boxes = boxes.reshape(orig_shape)

        # vsave(frames[[2,1,0]], torch.from_numpy(boxes))
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)
        return frames, boxes

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos) 

    def get_frame_path(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        ppath = os.path.join(os.path.dirname(self.data_root), 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))
        ppath = os.path.join(self.data_root, 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))
        return ppath

    def get_frames_list(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        self.tracked_boxes = f"{self.data_root}/bbox_jsons"
        self.coord_nr_frames = self.cfg.DATA.NUM_FRAMES

        vid_id = self._video_names[index]

        folder_id = str(int(self._video_names[index]))
        json_path = '{}/{}.json'.format(self.tracked_boxes, folder_id)
        with open(json_path, 'r') as f:
            video_data = json.load(f)

        n_frame = len(video_data)
        coord_frame_list = self.get_seq_frames(index, n_frame)

        frames = []
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            frames.append(self.get_frame_path(vid_id, int(frame_data['name'].split('/')[-1][:-4]) - 1 ))
        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

