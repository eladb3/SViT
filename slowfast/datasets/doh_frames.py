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
from .build import DATASET_REGISTRY
from .transform import create_random_augment
from .random_erasing import RandomErasing
from . import utils as utils

from slowfast.utils import box_ops
from functools import partial

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Doh_frames(torch.utils.data.Dataset):
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

        self.dprefix = cfg.DOH.DATA_ROOT
        self.data_prefix = f'{self.dprefix}/raw_256'
        self.data_root = self.data_prefix

        logger.info(f"Using data dir: {self.data_prefix}")


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

        logger.info("Constructing 100DOH frames {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

        self.bad_idxs =  set()
        
        
        self.O = 4

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # assert self.mode not in 'test', f"Test mode not supported for 100DOH"
        # Loading labels.
        data_split = self.mode
        self.data_split = data_split
        # dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        _mode = 'val' if self.mode == 'test' else self.mode
        self.file_labels = f'{self.dprefix}/file/{_mode}.json' 

        # label names.
        self.label_names = [
            'boardgame','diy','drink','food','furniture','gardening','housework','packing','puzzle','repair','study','vlog'
        ]
        self.label_names_idx = {k:i for i, k in enumerate(self.label_names)}

        # data
        with g_pathmgr.open(self.file_labels,"r") as f:
            label_json = json.load(f)

        # with g_pathmgr.open(label_file, "r") as f:
        #     label_json = json.load(f)

        # with open('data/ssv2/empty_bbox_{}.json'.format('train' if data_split == 'train' else 'val'), 'r') as f:
        #     sort_out = json.load(f)
        sort_out = []
        self._video_names = []
        self._labels = []
        
        for frame_name, labels in label_json.items():
            self._video_names.append(frame_name)
            self._labels.append(labels)

        path_to_file = self.file_labels

        self._path_to_videos = [os.path.join(self.data_prefix, path) for path in self._video_names]


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
            "100DOH frames dataloader constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        idx = index
        while True:
            while idx in self.bad_idxs:
                idx = random.randint(0, len(self) - 1)
            try:
                return self.getitem(idx)
            except Exception as e:
                print(traceback.format_exc())
                self.bad_idxs.add(idx)
                print(f"Num of bad vids: {len(self.bad_idxs)}")
            idx = random.randint(0, len(self) - 1)
    def getitem(self, index):
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
        T = 1

        label = self._labels[index]

        boxes, contact_state = self.get_boxes(index)
        boxes = np.concatenate([boxes for _ in range(T)], axis=0) # [T, O, 4], xyxy, normalized

        fpaths = [self._path_to_videos[index]]
        
        frames = torch.as_tensor(
            utils.retry_load_images(
                fpaths,
                self._num_retries,
            )
        ) # [T, H, W, C]

        frames = frames.expand(T, -1, -1, -1) # [T, H, W, C]
        # unnormalize
        H, W = frames.shape[1:3]
        boxes = np.stack([b*s for b,s in zip([boxes[:, :, i] for i in range(4)], [W, H, W, H])], axis=2)# [T, O, 4], xyxy, normalized
    
        clean_img = None
        aug_params = None
        if self.aug:
            frames, boxes, clean_img = self._aug_frame(
                frames,
                spatial_sample_index,
                min_scale,
                max_scale,
                crop_size,
                boxes = boxes,
                extra_data = aug_params,
            ) # [C, T, H, W] vsave(frames.permute(3,0,1,2)[[2,1,0], ], torch.zeros(16,0,4))
        else:
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            if clean_img is not None:
                T = frames.shape[1]
                clean_img = clean_img.permute(3, 0, 1, 2)
                frames = torch.cat((frames, clean_img), dim = 1)
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
            if aug_params is not None:
                aug_params['spatial_sampling_args'] =  dict(
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            if boxes is not None: frames, boxes = frames
            if clean_img is not None:
                frames, clean_img = frames[:, :T], frames[:, T:]

        frames = utils.pack_pathway_output(self.cfg, frames)
        if boxes is not None:
            h, w = frames[0].shape[-2:]
            boxes[..., [0,2]] = boxes[..., [0,2]] / w
            boxes[..., [1,3]] = boxes[..., [1,3]] / h
            boxes = np.clip(boxes, 0, 1)
            boxes = torch.from_numpy(boxes) # [T, O, 4]
            boxes = box_ops.box_xyxy_to_cxcywh(boxes)
            boxes = box_ops.zero_empty_boxes(boxes, mode='cxcywh')
            metadata['haog_bboxes'] = boxes
        
        frame_name = metadata['vid'] = self._video_names[index]
        label_idx = self.label_names_idx[frame_name.split('/')[1]]
        if clean_img is not None: metadata['clean_img'] = clean_img.byte()
        if aug_params is not None: metadata['aug_params'] = aug_params
        metadata['contact_state'] = contact_state
        metadata['label_idx'] = label_idx
        
        return frames, -1, index, metadata # vsave(frames[0][[2,1,0], ], torch.zeros(16,0,4)), vsave(frames[0][[2,1,0]], boxes)


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
        extra_data = None,
    ):

        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
            with_boxes = boxes is not None,
        )
        if extra_data is not None:
            extra_data['aug_transform_state_dict'] = aug_transform._make_det_and_get_state_dict()
            aug_transform.use_tensor_op()
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2) # [T, C, H, W]
        list_img = self._frame_to_list_img(frames) # [ [C, H, W] ... ]
        
        list_img, boxes, clean_img = aug_transform(
            list_img,
            boxes=boxes,
            clean_img = None,
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
        if clean_img is not None:
            clean_img = (self._list_img_to_frames(clean_img) * 255).byte()
            clean_img = clean_img.permute(0, 2, 3, 1) # [T, H, W, C]
            # T H W C -> C T H W.
            clean_img = clean_img.permute(3, 0, 1, 2)
            T = clean_img.shape[1]
            frames = torch.cat((frames,clean_img), dim = 1)
        spatial_sampling_dict = {}
        _motion_shift = self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT and self.mode in ["train"]
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
            motion_shift=_motion_shift,
            boxes=boxes,
            rand_params = spatial_sampling_dict,
        )
        if extra_data is not None:
            extra_data['spatial_sampling_args'] =  dict(
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
                rand_params = spatial_sampling_dict,
            )

        if boxes is not None:
            frames, boxes = frames
            boxes = boxes.reshape(orig_shape)
        if clean_img is not None:
            frames, clean_img = frames[:, :T], frames[:, T:]

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
        return frames, boxes, clean_img

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)


    def get_boxes(self, index):
        EMPTY_TAG = {'x1':0, 'y1':0, 'x2':0, 'y2':0, 'obj_bbox':None, 'contact_state':-1}

        assert self.O == 4, f"Only support 4-objects -- [right hand , right hand object, left hand , left hand object]"
        # dict_keys(['x1', 'y1', 'x2', 'y2', 'contact_state', 'hand_side', 'width', 'height', 'obj_bbox'])
        labels = self._labels[index]
        rh = [v for v in labels if v['hand_side'] == 'r']
        lh = [v for v in labels if v['hand_side'] == 'l']
        rh = rh[0] if len(rh) > 0 else EMPTY_TAG
        lh = lh[0] if len(lh) > 0 else EMPTY_TAG
        rh_contact_state, lh_contact_state = rh['contact_state'], lh['contact_state']
        rh_box = [rh['x1'], rh['y1'], rh['x2'], rh['y2']]
        lh_box = [lh['x1'], lh['y1'], lh['x2'], lh['y2']]
        rh_obj_box = [rh['obj_bbox']['x1'], rh['obj_bbox']['y1'], rh['obj_bbox']['x2'], rh['obj_bbox']['y2']] if rh['obj_bbox'] is not None else [0,0,0,0]
        lh_obj_box = [lh['obj_bbox']['x1'], lh['obj_bbox']['y1'], lh['obj_bbox']['x2'], lh['obj_bbox']['y2']] if lh['obj_bbox'] is not None else [0,0,0,0]
        rh_box, lh_box, rh_obj_box, lh_obj_box = map(
            np.array,
            [rh_box, lh_box, rh_obj_box, lh_obj_box],
        )
        boxes = np.stack([rh_box, lh_box, rh_obj_box, lh_obj_box], axis=0).reshape(1,4,4) # T O 4, normalized xyxy
        return boxes, [rh_contact_state, lh_contact_state]



    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)


def gen_random_boxes(T, O):
    """
    Output:
        cxcywh : torch.Tensor([T, O, 4]) 
    """
    out = np.zeros([T, O, 4])
    out = np.random.rand(T, O, 4)
    cxcy, wh = out[:, :, :2], out[:, :, 2:]
    dmax = np.min(np.stack([cxcy, 1-cxcy], axis=0), axis = 0) * 2
    wh = wh * dmax
    assert np.all(wh >= 0)
    out = np.concatenate([cxcy, wh], axis=2)
    return torch.from_numpy(out)
