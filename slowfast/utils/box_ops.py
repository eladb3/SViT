# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
import numpy as np 

def box_xywh_to_xyxy(x):
    # xywh --> (x, y, x+w, y+h)
    x, y, w, h = x.unbind(-1)
    b = [x, y, x + w, y + h]
    return torch.stack(b, dim=-1)

def box_xywh_to_cxcywh(x):
    # xywh --> (cx, cy, x+w, y+h)
    x, y, w, h = x.unbind(-1)
    x0, y0, x1, y1 = x, y, x + w, y + h
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def concat_boxes(b1, b2):
    """
    b1: [O1, 4], b2: [O2, 4]
    Return: permutation over indices of O2, List[int]
    """
    cost = 1-generalized_box_iou(b1, b2) # [O1, O2]
    row_ind, col_ind = linear_sum_assignment(cost.numpy())

    return torch.from_numpy(col_ind)

def zero_empty_boxes(boxes, mode='cxcywh', eps = 0.05):
    assert isinstance(boxes, torch.Tensor)
    oshape = boxes.shape
    boxes = boxes.reshape(-1 ,4) # [N, 4]
    if mode == 'xyxy':
        wh = boxes[..., [2,3]] - boxes[..., [0,1]]
    elif mode == 'cxcywh':
        wh = boxes[...,-2:]
    else:
        raise NotImplementedError(mode)
    assert torch.all(wh >= 0)
    empty_boxes = torch.any(wh <= eps, dim = -1) # [N]
    boxes[empty_boxes] = 0
    boxes = boxes.reshape(oshape)
    return boxes

def remove_empty_boxes(box, eps = 0.05, mode='xyxy'):
    assert mode in ['xyxy']
    assert len(box.shape) == 2
    H, W = (box[:, 3] - box[:, 1],  box[:, 2] - box[:, 0])
    mask = (H>eps) * (W>eps)
    box = box[mask]
    return box

def match_haog(haog, format='xyxy'):
    """_summary_

    Args:
        haog (torch.Tenosr[O=4, 4]): 4 boxes represents HAOG
    Return:
        haog (torch.Tensor[O=4, 4]): HAOG with matched hands and objects (by distance)
        contact_state (torch.Tensor([2])): contact state of HAOG
    """
    HIGH_COST = 1e8
    CONTACT_THERSHOLD = 0.1
    unsqueeze_haog = False
    if haog.ndim == 3:
        unsqueeze_haog = True
        assert haog.size(0) == 1, haog.size()
        haog = haog.squeeze(0)

    chaog = haog
    if format == 'xyxy':
        chaog = box_xyxy_to_cxcywh(haog) # 1 O 4
    elif format == 'cxcywh':
        pass
    else:
        raise NotImplementedError(format)

    chaog = haog.unsqueeze(0)[..., :2] # 1 O 4
    cost = torch.cdist(chaog[:, :2, ], chaog[:, 2:], p=2).squeeze(0) # 2 2
    obj_is_zero = torch.all(haog[2:] == 0, dim = -1)
    hand_is_zero = torch.all(haog[:2] == 0, dim = -1)
    cost[:, obj_is_zero] = HIGH_COST
    cost[:, hand_is_zero] = HIGH_COST

    ord1_cost = cost[0,0] + cost[1,1]
    ord2_cost = cost[0,1] + cost[1,0]
    
    
    if ord2_cost < ord1_cost:
        h1, h2, o1, o2 = haog[0], haog[2], haog[1], haog[3]
        haog = torch.stack((h1, h2, o2, o1), dim=0)
        c1_dist, c2_dist = cost[0, 1], cost[1, 0]
    else:
        c1_dist, c2_dist = cost[0, 0], cost[1, 1]
    
    def _get_contact_state(dist):
        if dist == HIGH_COST:
            return -1
        elif dist < CONTACT_THERSHOLD:
            return 3
        else:
            return 0
    contact_state = list(map(_get_contact_state, [c1_dist, c2_dist]))
    contact_state = torch.tensor(contact_state, dtype=torch.int64)
    if unsqueeze_haog:
        haog = haog.unsqueeze(0)
    return haog, contact_state
def match_hand_to_object_boxes(hands, obj):
    """_summary_

    Args:
        hands (H 4)
        obj (O 4)
        cxcywh format
    return torch.tensor([H]) 
        closest object for each hand
    """
    H, O = len(hands), len(obj)
    hands, obj = hands[: ,:2], obj[: ,:2] # cxcy
    _a, _b = hands.unsqueeze(1).expand(H, O, 2), obj.unsqueeze(0).expand(H, O, 2)
    dist = ((_a - _b) ** 2).mean(-1) ** 0.5 # H O
    row_ind, col_ind = linear_sum_assignment(dist.numpy())

    return torch.from_numpy(col_ind)
    
