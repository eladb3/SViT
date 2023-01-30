#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
from .build import MODEL_REGISTRY

from logging import raiseExceptions
import slowfast.utils.logging as logging
from sklearn.utils import all_estimators
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from slowfast.utils import box_ops
from functools import partial
import slowfast.utils.distributed as du
from slowfast.utils import misc
logger = logging.get_logger(__name__)

def average_smoothing(tensor, dim, window_size, pool='sum'):
    """
    Apply average smoothing to a tensor along a dimension.
    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to apply smoothing.
        window_size (int): Size of the averaging window.
    Returns:
        torch.Tensor: Smoothed tensor.
    """
    tensor = tensor.transpose(dim, 0)
    out = []
    for i in range(tensor.size(0)):
        a, b = i - window_size // 2, i + window_size // 2 + 1
        a = max(0, a)
        b = min(tensor.size(0), b)
        t = tensor[a:b]
        if pool == 'mean':
            t = t.mean(dim = 0)
        elif pool == 'sum':
            t = t.sum(dim = 0)
        elif pool == 'max':
            t = t.amax(dim = 0)
        else:
            raise ValueError('Unknown pooling type: {}'.format(pool))
        out.append(t)
    tensor = torch.stack(out, dim = 0)
    tensor = tensor.transpose(dim, 0)
    return tensor    

def boxes_loss_(pred, tar):
    """[summary]
    Args:
        pred (torch.Tensor[B, T, O, 5]): pred boxes: (score, cx, cy, w, h)
        tar (torch.Tensor[B, T, O, 4]): GT boxes: (cx, cy, w, h)
        t_mask (torch.Tensor[B, T])
    Returns:
        loss: loss (no permutations)
    """
    
    if tar.size(-1) == 4:
        tar_mask = 1-torch.all(tar == 0, dim = -1).float() # [B, T, O]
        tar_mask_cont = tar_mask
    elif tar.size(-1) == 5:
        tar_mask_cont = tar[..., 0]
        tar_mask = (tar[..., 0] > 0.5).float()
        tar = tar[..., 1:]
    else:
        raise NotImplementedError("Boxes loss only supports 4 or 5 dimensional boxes")

    pred_mask = pred[..., 0] # [B, T, O]

    # BCE
    loss_mask = F.binary_cross_entropy_with_logits(pred_mask, tar_mask_cont, reduction="none")
    loss_mask = loss_mask.mean()
    
    if tar_mask.sum() > 0:
        # pred_boxes = pred[..., 1:].flatten(1,2) # [B, T*O, 4]
        pred_boxes, tar_boxes, mask = map(
            lambda x: x.flatten(1,2),
            (pred[..., 1:], tar, tar_mask.bool())
        )
        src_boxes, target_boxes = pred_boxes[mask], tar_boxes[mask]
        
        loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction='mean')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.mean()
    else:
        loss_l1 = torch.tensor(0, device=pred.device, requires_grad=True, dtype=torch.float32)
        loss_giou = torch.tensor(0, device=pred.device, requires_grad=True, dtype=torch.float32)

    return loss_l1, loss_mask, loss_giou

def boxes_giou_loss(pred, tar):
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(pred),
        box_ops.box_cxcywh_to_xyxy(tar)))
    loss_giou = loss_giou.mean()
    return loss_giou

class VideoImageLoss(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, cfg, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super().__init__()
        self.cfg = cfg
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction = self.reduction)
        self._lambda = misc.get_lambdas_dict(cfg)
        self.bce_sigmoid = False
        self._is_vid = du.get_local_rank() not in self.cfg.IMAGE_TRAIN.GPU_IDS

    def is_vid(self):
        return self._is_vid or (not self.training)
    def get_default_val(self):
        _device = torch.cuda.current_device() if self.cfg.NUM_GPUS > 0 else 'cpu'
        return torch.tensor(0, device=_device, requires_grad=True, dtype=torch.float32)

    def _consistency_loss(self, extra_preds, frames_extra_preds):
        ret = {}
        pred = extra_preds['obj_desc'] # [B, T, O, d]
        tar = frames_extra_preds['obj_desc'] # [BT, T, O, d]
        tar = tar.reshape(pred.shape).detach() # B T O d
        if 'video_image_desc_l1_loss' in self._lambda:
            ret['video_image_desc_l1_loss'] = F.l1_loss(pred, tar, reduction=self.reduction)
        if 'video_image_desc_l2_loss' in self._lambda:
            ret['video_image_desc_l2_loss'] = F.mse_loss(pred, tar, reduction=self.reduction)
        return ret

    def _haog_loss(self, extra_preds, metadata):
        ret = {}
        
        # boxes loss
        pred, tar = extra_preds['pred_bboxes'], metadata['haog_bboxes'] # [B, T, O, 4]
        boxes_l1_loss, boxes_bce_loss, boxes_giou_loss = boxes_loss_(
            pred, tar,
        )
        ret.update({'boxes_l1_loss': boxes_l1_loss, 'boxes_bce_loss': boxes_bce_loss, 'boxes_giou_loss':boxes_giou_loss})
        
        # contact state
        pred = extra_preds['pred_contact_state']  # [B, T=1, O//2, 5]
        tar = metadata['contact_state'] # [B, O//2]
        pred, tar = pred.flatten(0,2), tar.flatten() # [BTn, 4], [BTn]
        mask = tar >= 0
        pred, tar = pred[mask], tar[mask]
        ret['loss_contact_state'] = self.ce_loss(pred, tar) if mask.sum() > 0 else self.get_default_val()
        return ret
    def forward(self, x, extra_preds, y, metadata):
        ret = {}
        if self.is_vid():
            loss_ce = self.ce_loss(x, y)
            ret.update({'loss_ce':loss_ce})
            if self.cfg.TRAIN.FORWARD_VIDEO_FRAMES:
                frames_extra_preds = extra_preds['frames_output']['extra_preds']
                ret.update(self._consistency_loss(extra_preds, frames_extra_preds))
        else:
            ret.update(self._haog_loss(extra_preds, metadata))
        if 'safety_loss' in extra_preds:
            ret['safety_loss'] = extra_preds['safety_loss'] * 0
        return ret




class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError



_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "video_image_loss" :VideoImageLoss,
}

##

def get_loss_func(cfg, state = 'train'):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    loss_name = cfg.MODEL.LOSS_FUNC
    if state == 'val' and loss_name == 'soft_cross_entropy':
        # during evaluation measure cross entropy
        loss_name = 'cross_entropy'
    auto_args = {}
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    ret = _LOSSES[loss_name]

    if 'cfg' in ret.__init__.__code__.co_varnames: 
        auto_args['cfg'] = cfg
    if len(auto_args) > 0:
        ret = partial(ret, **auto_args)
    return ret
