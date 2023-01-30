#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter, eval_extra_metrics
from slowfast.utils.multigrid import MultigridSchedule
import os

logger = logging.get_logger(__name__)

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg) #{'noun': 300, 'verb': 97} if cfg.TRAIN.DATASET == 'epickitchens' else cfg.MODEL.NUM_CLASSES,
        )

    all_lengths = du.all_gather_unaligned(len(train_loader))
    min_length = min(all_lengths)
    wloss = misc.get_lambdas_dict(cfg)

    logger.info(f"Dataloadr length: {all_lengths}, min_length: {min_length}")
    logger.info(f"Epoch: {cur_epoch}, lambda: {wloss}")

    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):
        if cur_iter >= min_length:
            break
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta], non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr, log = cur_iter == 0)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples


        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if cfg.DETECTION.ENABLE and 'boxes' in meta:
                preds = model(inputs, meta, bboxes=meta["boxes"])
            else:
                preds = model(inputs, meta)
            if isinstance(preds, tuple): preds, extra_preds = preds
            else: extra_preds = None

            if cfg.TRAIN.FORWARD_VIDEO_FRAMES and inputs[0].size(2) > 1:
                with torch.no_grad():
                    _preds, _extra_preds = model(
                        [inputs[0].transpose(1,2).flatten(0,1).unsqueeze(2)], {}, # BT C 1 H W
                    )
                extra_preds['frames_output'] = {'preds': _preds, 'extra_preds': _extra_preds}
                
                
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg)(
                reduction="mean"
            )
            if cfg.NUM_GPUS > 1:
                loss_fun = loss_fun.to(model.device)


            # Compute the loss.
            if hasattr(loss_fun, 'forward') and 'extra_preds' in loss_fun.forward.__code__.co_varnames:
                loss_dict = loss_fun(preds, extra_preds, labels, meta)
                loss = sum([wloss[k] * loss_dict[k] for k in loss_dict])
                loss_dict['loss'] = loss
            else:
                loss = loss_fun(preds, labels)
                loss_dict = {'loss':loss}

        # check Nan Loss.
        misc.check_nan_losses(loss, extra_msg = str(loss_dict) + ", " + str(preds) + ", ")
        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )

        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        list_loss_dict = du.all_gather_unaligned({k:v.item() for k,v in loss_dict.items()})
        loss_dict = {}

        for d in list_loss_dict:
            for k,v in d.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        loss_dict = {k:sum(v)/len(v) for k,v in  loss_dict.items()}

        train_meter.update_stats(
            lr,
            inputs[0].size(0) * cfg.NUM_GPUS,
            dloss = loss_dict,
        )




        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta, bboxes=meta["boxes"])
            if isinstance(preds, tuple): preds, extra_preds = preds
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = misc.iter_to_cpu(preds)
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                if isinstance(preds, dict):
                    preds = du.all_gather_unaligned(preds)
                    preds = {k:torch.cat([p[k] for p in preds], dim=0) for k in preds[0].keys()}
                    ori_boxes = du.all_gather_unaligned(ori_boxes)
                    # fix batch index
                    _max = 0
                    for iboxes in ori_boxes:
                        iboxes[:, 0] = iboxes[:, 0] + _max
                        _max = max(iboxes[:, 0]) + 1
                    ori_boxes = torch.cat(ori_boxes, dim=0)
                else:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs, meta)
            if isinstance(preds, tuple): preds, extra_preds = preds
            else: extra_preds = None
            if cfg.TRAIN.FORWARD_VIDEO_FRAMES and inputs[0].size(2) > 1:
                with torch.no_grad():
                    _preds, _extra_preds = model(
                        [inputs[0].transpose(1,2).flatten(0,1).unsqueeze(2)], {}, # BT C 1 H W
                    )
                extra_preds['frames_output'] = {'preds': _preds, 'extra_preds': _extra_preds}
            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET in ['epickitchens', 'epickitchens_with_100doh', 'epickitchens_with_100doh_videos']:
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(
                    extra_preds['verb'], labels['verb'], (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [verb_top1_acc, verb_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(
                    extra_preds['noun'], labels['noun'], (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [noun_top1_acc, noun_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                    (extra_preds['verb'], extra_preds['noun']),
                    (labels['verb'], labels['noun']),
                    (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()
                # additional metrices
                extra_metrics = eval_extra_metrics(cfg, preds, extra_preds, labels, meta)
                if cfg.NUM_GPUS > 1:
                    all_keys, all_values = zip(*list(extra_metrics.items()))
                    all_values = du.all_reduce(all_values)
                    extra_metrics = dict(zip(all_keys, all_values))

                val_meter.iter_toc()
                
                # Update and log stats.
                val_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    inputs[0].size(0) * cfg.NUM_GPUS,
                    extra_metrices = extra_metrics,
                )
                
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Val/verb_top1_acc": verb_top1_acc,
                            "Val/verb_top5_acc": verb_top5_acc,
                            "Val/noun_top1_acc": noun_top1_acc,
                            "Val/noun_top5_acc": noun_top5_acc,
                            "Val/action_top1_acc": action_top1_acc,
                            "Val/action_top5_acc": action_top5_acc,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
            else:
                if cfg.DATA.MULTI_LABEL or (isinstance(preds, torch.Tensor) and preds.size(-1) == 0):
                    if cfg.NUM_GPUS > 1:
                        preds, labels = du.all_gather([preds, labels])
                else:
                    # Compute the errors.
                    ks = (1,5) if cfg.MODEL.NUM_CLASSES > 5 else (1,1)
                    if cfg.TRAIN.DATASET == 'ego4d_oscc':
                        preds = (preds if isinstance(preds, dict) else extra_preds)['cls']
                    num_topks_correct = metrics.topks_correct(preds, labels, ks)

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                    # additional meterices
                    extra_metrics = eval_extra_metrics(cfg, preds, extra_preds, labels, meta)
                    if cfg.NUM_GPUS > 1:
                        all_keys, all_values = zip(*list(extra_metrics.items()))
                        all_values = du.all_reduce(all_values)
                        extra_metrics = dict(zip(all_keys, all_values))

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err,
                        top5_err,
                        inputs[0].size(0)
                        * max(
                            cfg.NUM_GPUS, 1
                        ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                        extra_metrices = extra_metrics,
                    )
                    # write to tensorboard format if available.
                    if writer is not None:
                        writer.add_scalars(
                            {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                            global_step=len(val_loader) * cur_epoch + cur_iter,
                        )

            val_meter.update_predictions(preds, labels, metadata=meta, extra_preds=extra_preds)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            if cfg.TRAIN.DATASET == 'visualgenome':
                writer.add_scalars(
                    {f"Val/{k}":v  for k,v in val_meter.full_map.items()}, global_step=cur_epoch
                )

            else:
                writer.add_scalars(
                    {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
                )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train" if not cfg.TRAIN.VAL_ONLY else "val")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    if cfg.TRAIN.VAL_ONLY:
        start_epoch = max(start_epoch - 1, 0)

    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                writer,
            )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
        if cfg.TRAIN.VAL_ONLY:
            break
    if writer is not None:
        writer.close()