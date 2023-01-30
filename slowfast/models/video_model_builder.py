# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from slowfast.utils import misc

from .attention import MultiScaleBlock
from .utils import (
    round_width,
)

from . import stem_helper
from .build import MODEL_REGISTRY



@MODEL_REGISTRY.register()
class SViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = misc.get_num_classes(cfg) # cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )

        self.patch_embed = patch_embed
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, pos_embed_dim, embed_dim)
                )

        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.cfg.DATA.NUM_FRAMES, embed_dim)
        ) # Need it anyway for the objects
        self.O = self.cfg.SVIT.O
        self.object_queries =  nn.Parameter(
                torch.zeros(
                    1, self.O, embed_dim
                )
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims
        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                separate_qkv=cfg.MVIT.SEPARATE_QKV,
            )

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.enable_detection = cfg.DETECTION.ENABLE
        self.head = SViTHead(
            cfg,
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed_temporal, std=0.02)
        trunc_normal_(self.object_queries, std=0.02)
        self.apply(self._init_weights)
        self._lambda = misc.get_lambdas_dict(cfg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append(["pos_embed"])
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")
            names.extend(["object_queries", "pos_embed_temporal"])

        return names

    def _get_pos_embed(self, pos_embed, bcthw):
        t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x, metadata=None, bboxes=None):
        x = x[0]
        if len(x.shape) == 4: # image
            x = x.unsqueeze(2)
        Tx, Hx, Wx = x.shape[-3:]

        x, bcthw = self.patch_embed(x)
        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0] if Tx > 1 else Tx
        H, W = bcthw[-2], bcthw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                if Tx == 1:
                    pos_embed = self.pos_embed_spatial.repeat(
                        1, T, 1
                    )
                else:
                    pos_embed = self.pos_embed_spatial.repeat(
                        1, self.patch_dims[0], 1
                    ) + torch.repeat_interleave(
                        self.pos_embed_temporal,
                        self.patch_dims[1] * self.patch_dims[2],
                        dim=1,
                    )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                pos_embed = self._get_pos_embed(pos_embed, bcthw)
                x = x + pos_embed
            else:
                assert False
                pos_embed = self._get_pos_embed(self.pos_embed, bcthw)
                x = x + pos_embed
        x_objects = self.object_queries.unsqueeze(1).expand(B, Tx, -1, -1) # [B, T, O, d]
        if Tx > 1:
            obj_pos_embed_temporal = self.pos_embed_temporal # 1 T d
            obj_pos_embed_temporal = obj_pos_embed_temporal.unsqueeze(2).expand(B, -1, self.O, -1)
        else:
            obj_pos_embed_temporal = self.pos_embed_temporal.sum() * 0 # avoid gradient issues
        x_objects = x_objects + obj_pos_embed_temporal # [B, T, O, d]
        x_objects = x_objects.flatten(1,2)
        O = x_objects.shape[1]
        x = torch.cat([x, x_objects], dim = 1) # [B, 1 + THW + O, d]

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)

        if self.cls_embed_on:
            patch_tokens = x[:, 1:-O]
            x, obj =  x[:, [0]], x[:, -O:]
        else:
            patch_tokens = x[:, :-O]
            x, obj = x[:, :-O].mean(1, keepdim=True), x[:, -O:]

        x = torch.cat((x, obj) ,dim = 1)
        if self.enable_detection:
            patch = patch_tokens.transpose(1, 2).reshape(B, -1, thw[0], thw[1], thw[2])
        else:
            patch = None


        if self.enable_detection:
            x, extra_preds = self.head(x, T=Tx, patches=patch, bboxes=bboxes) # [B, n_classes], dict
        else:
            x, extra_preds = self.head(x, T=Tx) # [B, n_classes], dict


        if isinstance(x, dict): extra_preds.update(x)
        return x, extra_preds

class ZeroLinear(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x_shape = list(x.shape)
        x_shape[-1] = 0
        return x.new_zeros(x_shape, requires_grad=True)

class SViTHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super().__init__()
        self.cfg = cfg
        self.T = self.cfg.DATA.NUM_FRAMES
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        if self.cfg.DETECTION.ENABLE:
            self.build_det(num_classes)
        else:
            if isinstance(num_classes, dict):
                self.projection = nn.ModuleDict({k:nn.Linear(dim_in, v, bias=True) for k,v in num_classes.items()})
            else:
                if num_classes == 0:
                    self.projection = ZeroLinear()
                else:
                    self.projection = nn.Linear(dim_in, num_classes, bias=True)
    
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

        # Boxes
        self.boxes_mlp = nn.Sequential(
            nn.Linear(dim_in, 4, bias=True),
            nn.Sigmoid(),
        )
        self.boxes_bce_mlp = nn.Linear(dim_in, 1, bias=True)
        self.contact_mlp = nn.Linear(dim_in, 5, bias=True)
        
        self.ret_obj_desc = False
        self._lambdas = misc.get_lambdas_dict(cfg)
        if cfg.TRAIN.FORWARD_VIDEO_FRAMES:
            self.ret_obj_desc = True
    
    def build_det(self, num_classes):
        from slowfast.models.head_helper import ResNetRoIHead as ResNetRoIHead
        cfg = self.cfg
        embed_dim = 768
        if self.cfg.DETECTION.ENABLE:
            _num_classes =  num_classes
            self.det_head = ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=_num_classes,
                pool_size=[[cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                roi_head_act_during_training=cfg.MODEL.ROI_HEAD_ACT_DURING_TRAINING,
            )

    def do_det(self, patches, bboxes):
        return self.det_head([patches], bboxes)


    def do_classification(self, x):
        if isinstance(self.projection, nn.ModuleDict):
            x = {k:self.projection[k](x) for k in self.projection}
            if not self.training:
                x = {k:self.act(v) for k,v in x.items()}

        else:
            x = self.projection(x)
            if not self.training:
                x = self.act(x)
        return x

        
    def forward(self, x, T = None, patches=None, bboxes=None):
        if T is None: T = self.T
        
        extra_preds = {}
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        B = x.size(0)
        x = x + (sum([p.sum() for p in self.parameters()]) * 0) # avoid gradient issues
        x, xobj = x[:, 0], x[:, 1:]
        extra_preds['obj_desc'] = xobj.reshape(B,T,-1,xobj.size(-1))
        
        if self.cfg.DETECTION.ENABLE:
            if None not in (patches, bboxes):
                x = self.do_det(patches, bboxes)
        else:
            if isinstance(self.projection, nn.ModuleDict):
                x = {k:self.projection[k](x) for k in self.projection}
                if not self.training:
                    for k,v in x.items():
                        x[k] = self.act(v)
                extra_preds.update(x)
            else:
                x = self.projection(x)
                if not self.training:
                    x = self.act(x)

    
        # HAOG
        xobj = xobj.reshape(B, T, -1, xobj.size(-1)) # B T O d
        boxes = self.boxes_mlp(xobj) # B T O 4        
        boxes_bce = self.boxes_bce_mlp(xobj) # B T O 1
        pred_contact = self.contact_mlp(xobj[:, :, :2]) # B T O 5

        if not self.training:
            boxes_bce = boxes_bce.sigmoid()
            pred_contact = pred_contact.softmax(dim=-1)

        boxes = torch.cat((boxes_bce, boxes), dim = -1)
        extra_preds['pred_bboxes'] = boxes
        extra_preds['pred_contact_state'] = pred_contact

        # safety loss
        if 'safety_loss' in self._lambdas:
            extra_preds['safety_loss'] = sum([p.sum() for p in self.parameters()])
        return x, extra_preds

