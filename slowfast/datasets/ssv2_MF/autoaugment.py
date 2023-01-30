# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
pulished under an Apache License 2.0.

COMMENT FROM ORIGINAL:
AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and
improvements and does not include any of the search code. AA and RA
Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection
    https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation...
    https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and
    Uncertainty https://arxiv.org/abs/1912.02781

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import math
import numpy as np
import random
import re
import os
from . import boxes_autoaugment
import PIL
from PIL import Image, ImageOps, ImageEnhance, ImageChops, ImageDraw
from torchvision.transforms import functional as TF
_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0

_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": _FILL,
}

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

def tf_affine_zero_default(img, **kwargs):
    angle = kwargs.pop('angle', 0)
    translate = kwargs.pop('translate', [0,0])
    scale = kwargs.pop('scale', 1)
    shear = kwargs.pop('shear', [0,0])

    return TF.affine(img, angle, translate, scale, shear, **kwargs)

def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs
    )

def shear_x_tensor(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return tf_affine_zero_default(
        img, shear=[factor*90, 0], fill=_FILL
    )


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs
    )

def shear_y_tensor(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return tf_affine_zero_default(
        img, shear=[0, factor*90], fill=_FILL
    )

def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )

def translate_x_rel_tensor(img, pct, **kwargs):
    _check_args_tf(kwargs)
    if isinstance(img, torch.Tensor):
        pixels = pct * img.shape[-1]
    else:
        pixels = pct * img.size[0]
    return tf_affine_zero_default(
        img, translate=[pixels, 0], fill=_FILL
    )

def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )

def translate_y_rel_tensor(img, pct, **kwargs):
    _check_args_tf(kwargs)
    if isinstance(img, torch.Tensor):
        pixels = pct * img.shape[-2]
    else:
        pixels = pct * img.size[1]
    return tf_affine_zero_default(
        img, shear=[0, pixels], fill=_FILL
    )

def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )

def translate_x_abs_tensor(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return tf_affine_zero_default(
        img, translate=[pixels, 0], fill=_FILL
    )

def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )

def translate_y_abs_tensor(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return tf_affine_zero_default(
        img, shear=[0, pixels], fill=_FILL
    )

def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0],
            -rotn_center[1] - post_trans[1],
            matrix,
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])

def rotate_tensor(img, degrees, **kwargs):
    return TF.rotate(img, degrees)


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)

def auto_contrast_tensor(img, **__):
    return TF.autocontrast(img)

def invert(img, **__):
    return ImageOps.invert(img)

def invert_tensor(img, **__):
    return TF.invert(img)

def equalize(img, **__):
    return ImageOps.equalize(img)

def equalize_tensor(img, **__):
    return TF.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)

def solarize_tensor(img, thresh, **__):
    return TF.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)

def posterize_tensor(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return TF.posterize(img, bits_to_keep)

def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)

def contrast_tensor(img, factor, **__):
    return TF.adjust_contrast(img, factor)

def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)

def color_tensor(img, factor, **__):
    return TF.adjust_saturation(img, factor)

def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)

def brightness_tensor(img, factor, **__):
    return TF.adjust_brightness(img, factor)

def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)

def sharpness_tensor(img, factor, **__):
    return TF.adjust_sharpness(img, factor)

def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


SHAPE_OPS = {
    "Rotate": rotate,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}

TENSOR_OPS = {
    "AutoContrast": auto_contrast_tensor,
    "Equalize": equalize_tensor,
    "Invert": invert_tensor,
    "Rotate": rotate_tensor,
    "Posterize": posterize_tensor,
    "PosterizeIncreasing": posterize_tensor,
    "PosterizeOriginal": posterize_tensor,
    "Solarize": solarize_tensor,
    "SolarizeIncreasing": solarize_tensor,
    # "SolarizeAdd": solarize_add_tensor,
    "Color": color_tensor,
    "ColorIncreasing": color_tensor,
    "Contrast": contrast_tensor,
    "ContrastIncreasing": contrast_tensor,
    "Brightness": brightness_tensor,
    "BrightnessIncreasing": brightness_tensor,
    "Sharpness": sharpness_tensor,
    "SharpnessIncreasing": sharpness_tensor,
    "ShearX": shear_x_tensor,
    "ShearY": shear_y_tensor,
    "TranslateX": translate_x_abs_tensor,
    "TranslateY": translate_y_abs_tensor,
    "TranslateXRel": translate_x_rel_tensor,
    "TranslateYRel": translate_y_rel_tensor,
}

class AugmentOp:
    """
    Apply for video.
    """

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None, seed=None, rand_params_dict={}):
        hparams = hparams or _HPARAMS_DEFAULT
        self.rand_params_dict = rand_params_dict
        self.aug_name = name
        self.aug_fn = NAME_TO_OP[name]
        self.box_aug_fn = boxes_autoaugment.NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = {
            "fillcolor": hparams["img_mean"]
            if "img_mean" in hparams
            else _FILL,
            "resample": hparams["interpolation"]
            if "interpolation" in hparams
            else _RANDOM_INTERPOLATION,
        }

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        self.magnitude_std = self.hparams.get("magnitude_std", 0)
        self.seed = seed

    def _use_tensor_op(self):
        if self.aug_name in TENSOR_OPS:
            self.aug_fn = TENSOR_OPS[self.aug_name]

    def _make_det_and_get_state_dict(self):
        self.rand_params_dict.update(
            self.get_aug_op_rand_params(self.magnitude, self.magnitude_std)
        )
        state_dict = {
            'rand_params_dict': self.rand_params_dict.copy(),
            'aug_name':self.aug_name,
            'prob':self.prob,
            'magnitude':self.magnitude,
            'hparams': self.hparams.copy(),
            'seed':self.seed,
        }
        return state_dict
    
    def init_from_state_dict(state_dict):
        ret = AugmentOp(
            state_dict['aug_name'],
            prob=state_dict['prob'],
            magnitude=state_dict['magnitude'],
            hparams=state_dict['hparams'],
            seed=state_dict['seed'],
            rand_params_dict=state_dict['rand_params_dict'],
        )
        return ret
    def get_aug_op_rand_params(self, magnitude, magnitude_std):
        ret = {
            'prob': random.random(),
            'magnitude': random.gauss(magnitude, magnitude_std) if magnitude_std > 0 else magnitude,
        }
        return ret

    def __call__(self, img_list, boxes=None, clean_img=None):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        if self.prob < 1.0 and self.rand_params_dict.get('prob', random.random()) > self.prob:
            return img_list, boxes, clean_img

        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = self.rand_params_dict.get(
                'magnitude',
                random.gauss(magnitude, self.magnitude_std),
            )
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else ()
        )

        if isinstance(img_list, list):
            ret_img = [
                self.aug_fn(img, *level_args, **self.kwargs) for img in img_list
            ]
        else:
            ret_img = self.aug_fn(img_list, *level_args, **self.kwargs)

        if clean_img is not None and self.aug_name in SHAPE_OPS:
            if isinstance(clean_img, list):
                clean_img = [
                    self.aug_fn(img, *level_args, **self.kwargs) for img in clean_img
                ]
            else:
                clean_img = self.aug_fn(clean_img, *level_args, **self.kwargs)


        if boxes is not None: # assert len(set([im.size for im in img_list])) == 1 and len(set([im.size for im in ret_img])) == 1
            size_before = img_list[0].size if isinstance(img_list, list) else img_list.size
            size_after = ret_img[0].size if isinstance(ret_img, list) else ret_img.size
            kwargs = {'size_before': size_before}
            zero_mask = np.repeat((boxes == 0).all(axis = -1, keepdims =True), 4, -1)
            ret_boxes = np.stack([self.box_aug_fn(boxes[i], *level_args, **kwargs) for i in range(len(boxes))])
            # ret_boxes[:, [0,2]] = np.clip(ret_boxes[:, [0,2]], 0, size_after[0])
            # ret_boxes[:, [1,3]] = np.clip(ret_boxes[:, [1,3]], 0, size_after[1])
            ret_boxes[zero_mask] = 0
            boxes = ret_boxes
        return ret_img, boxes, clean_img




_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None, seed=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams, seed=seed)
        for name in transforms
    ]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None, debug_path='', rand_params_dict={}):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights
        self.debug_path = debug_path
        self.rand_params_dict = rand_params_dict.copy()
        self._filter_non_geom_transforms = False
        self._use_tensor_op = False
    def use_tensor_op(self):
        self._use_tensor_op = True

    def filter_non_geom_transforms(self):
        self._filter_non_geom_transforms = True
    def _make_det_and_get_state_dict(self):
        
        # ops
        self.rand_params_dict.update(
            self.get_aug_op_rand_params()
        )
        state_dict = {
            'rand_params_dict': self.rand_params_dict.copy(),
            'ops_state_dict':[self.ops[i]._make_det_and_get_state_dict() if i in self.rand_params_dict['ops_choices'] else None for i in range(len(self.ops))],
            'num_layers':self.num_layers,
            'choice_weights':self.choice_weights,
            'debug_path': self.debug_path,
        }
        return state_dict
    
    def init_from_state_dict(state_dict):
        ops = [None if sd is None else AugmentOp.init_from_state_dict(sd) for sd in state_dict['ops_state_dict']]
        ret = RandAugment(
            ops,
            num_layers=state_dict['num_layers'],
            choice_weights=state_dict['choice_weights'],
            debug_path=state_dict['debug_path'],
            rand_params_dict=state_dict['rand_params_dict'],
        )
        return ret
    def get_aug_op_rand_params(self):
        ret = {
            'ops_choices': np.random.choice(
                                range(len(self.ops)),
                                self.num_layers,
                                replace=self.choice_weights is None,
                                p=self.choice_weights,
                            ).tolist(),
        }
        return ret


    def __call__(self, img, boxes=None, clean_img=None):    
        # no replacement when using weighted choice
        if 'ops_choices' in self.rand_params_dict:
            ops = [self.ops[i] for i in self.rand_params_dict['ops_choices']]
            assert all([op is not None for op in ops])
        else:
            ops = np.random.choice(
                self.ops,
                self.num_layers,
                replace=self.choice_weights is None,
                p=self.choice_weights,
            )

        if self.debug_path != '':
            # save image for debug
            for idx, im in enumerate(img):
                img1 = im.copy()
                if boxes is not None:
                    dimg1 = ImageDraw.Draw(img1)  
                    for box in boxes[idx]:
                        dimg1.rectangle(box, outline ="red")
                img1.save(os.path.join(self.debug_path, f'debug_img_{idx}.jpeg'))

        for op in ops:
            if self._filter_non_geom_transforms and op.aug_name not in SHAPE_OPS: continue 
            if self._use_tensor_op:
                op._use_tensor_op()
            img, boxes, clean_img = op(img, boxes=boxes, clean_img=clean_img)
            if self.debug_path != '':
                # save image for debug
                for idx, im in enumerate(img):
                    img1 = im.copy()
                    if boxes is not None:
                        dimg1 = ImageDraw.Draw(img1)  
                        for box in boxes[idx]:
                            dimg1.rectangle(box, outline ="red")
                    img1.save(os.path.join(self.debug_path, f'debug_img_{idx}.jpeg'))
        return img, boxes, clean_img


def rand_augment_transform(config_str, hparams, seed=None, debug_path=''):
    """
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719

    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split("-")
    assert config[0] == "rand"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert NotImplementedError
    ra_ops = rand_augment_ops(
        magnitude=magnitude, hparams=hparams, transforms=transforms, seed=seed
    )
    choice_weights = (
        None if weight_idx is None else _select_rand_weights(weight_idx)
    )
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights, debug_path=debug_path)