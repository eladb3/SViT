#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn
# from . import tools
import slowfast.utils.logging as logging
import os
from slowfast.utils.misc import get_time_str

logger = logging.get_logger(__name__)

class VitHooks:
    """
    A class used to get weights and activations from specified layers from a Pytorch model.
    """

    def __init__(self, cfg, model):
        """
        Args:
            model (nn.Module): the model containing layers to obtain weights and activations from.
            layers (list of strings): a list of layer names to obtain weights and activations from.
                Names are hierarchical, separated by /. For example, If a layer follow a path
                "s1" ---> "pathway0_stem" ---> "conv", the layer path is "s1/pathway0_stem/conv".
        """
        self.cfg = cfg
        self.model = model
        self.hooks = {}
        self.layers_names = [f'blocks/{i}/attn' for i in range(self.cfg.MVIT.DEPTH)]
        self.current_batch_idx = 0
        self.base_path = os.path.join(cfg.OUTPUT_DIR, 'Visualizations', get_time_str())
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Save visualizations to {self.base_path}")
        self._register_main_hook()
        self._register_hooks()

    def _get_save_path(self, is_img=False):
        name = f'{self.current_batch_idx}'
        if is_img:
            name = f"{name}_img"
        ret = os.path.join(self.base_path, name)
        if not is_img:
            self.current_batch_idx += 1
        return ret

    def _get_layer(self, layer_name):
        """
        Return a layer (nn.Module Object) given a hierarchical layer name, separated by /.
        Args:
            layer_name (str): the name of the layer.
        """
        layer_ls = layer_name.split("/")
        prev_module = self.model
        for layer in layer_ls:
            l = prev_module._modules
            if isinstance(l, nn.ModuleList):
                layer = int(layer)
            prev_module = l[layer]

        return prev_module

    def _register_single_hook(self, layer_name):
        """
        Register hook to a layer, given layer_name, to obtain activations.
        Args:
            layer_name (str): name of the layer.
        """

        def hook_fn(module, input, output):
            self.hooks[layer_name] = output[1] # attn_map

        try:
            layer = get_layer(self.model, layer_name)
            layer.register_forward_hook(hook_fn)
        except Exception as e:
            print(f"Failed to get layer: {layer_name}")
            print(e)
    def _register_hooks(self):
        """
        Register hooks to layers in `self.layers_names`.
        """
        for layer_name in self.layers_names:
            self._register_single_hook(layer_name)

    def _register_main_hook(self):
        """
        Register hook to a layer, given layer_name, to obtain activations.
        Args:
            layer_name (str): name of the layer.
        """

        def hook_fn(module, input, output):
            self.hooks['input'] = input
            self.hooks['output'] = output
            is_img = input[0][0].size(2) == 1
            # np.save(self._get_save_path(), iter_to_cpu(self.hooks))
            d = iter_to_cpu(self.hooks)
            save_path = self._get_save_path(is_img)
            torch.save(d, save_path)
            self.hooks = {} # clean for next iteration

        layer = self.model
        layer.register_forward_hook(hook_fn)


def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split("/")
    prev_module = model
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module


def iter_to_cpu(d):
    iter = []
    if isinstance(d, torch.Tensor):
        return d.detach().cpu()
    elif isinstance(d, dict):
        iter = d.items( )
    elif isinstance(d, (list, tuple)):
        d = list(d)
        iter = enumerate(d)
    for k,v in iter:
        d[k] = iter_to_cpu(v)
    return d
