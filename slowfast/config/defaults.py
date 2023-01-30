#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.DDP_FIND_UNUSED_PARAMETERS = False
_C.DEBUG = False

# ---------------------------------------------------------------------------- #
# SViT options
# ---------------------------------------------------------------------------- #
_C.SVIT = CfgNode()

# Number of objects -- only supports 4 
_C.SVIT.O = 4 

# Losses Lamabdas
_C.SVIT.LAMBDA_NODES = 1.0
_C.SVIT.LAMBDA_EDGES = 1.0
_C.SVIT.LAMBDA_CON = 1.0


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# ---------------------------------------------------------------------------- #
# Image Training options.
# ---------------------------------------------------------------------------- #
_C.IMAGE_TRAIN = CfgNode()

# Total mini-batch size for images.
_C.IMAGE_TRAIN.BATCH_SIZE = 63

# GPUs to use for image training.
_C.IMAGE_TRAIN.GPU_IDS = [7]

# GPUs to use for image training.
_C.IMAGE_TRAIN.DATASETS = ['ssv2_frames']

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True
_C.TRAIN.ENABLE_DOH = False

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 63

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If set, replace all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_REPLACE_NAME_PATTERN = []

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False

# Forward video frames for Consistency Loss
_C.TRAIN.FORWARD_VIDEO_FRAMES = True

# Do single validation run.
_C.TRAIN.VAL_ONLY = False
# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Whether to enable randaug.
_C.AUG.ENABLE = False

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# MixUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"
# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CfgNode()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

_C.MODEL.LOAD_IN_PRETRAIN = ""

_C.MODEL.ROI_HEAD_ACT_DURING_TRAINING = False
# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()
_C.MVIT.USE_MLP = False

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# The initial value of layer scale gamma. Set 0.0 to disable layer scale.
_C.MVIT.LAYER_SCALE_INIT_VALUE = 0.0

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the ratio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0

# When using POOL_KV_STRIDE_ADAPTIVE, if stride == 1 for all dims then ignore
_C.MVIT.POOL_KV_IGNORE_111_KERNEL = False

_C.MVIT.IMAGE_KERNEL_FULL_PAD = False
_C.MVIT.OBJECTS_MASKING = False

# Mvit v2
# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = True

# Activation checkpointing enabled or not to save GPU memory.
_C.MVIT.ACT_CHECKPOINT = False

_C.MVIT.PATCH_AVG_TEMP = -1

##### MViT v2 new options #####

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = True

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = False

# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = False

# If True, using separate linear layers for Q, K, V in attention blocks.
_C.MVIT.SEPARATE_QKV = False

# The initialization scale factor for the head parameters.
_C.MVIT.HEAD_INIT_SCALE = 1.0

# Whether to use the mean pooling of all patch tokens as the output.
_C.MVIT.USE_MEAN_POOLING = False

# If True, use frozen sin cos positional embedding.
_C.MVIT.USE_FIXED_SINCOS_POS = False

# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# Target res for additional model input.
_C.DATA.TARGET_RES = [28,28]
# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

_C.CUDA_VISIBLE_DEVICES = ''

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.NUM_WORKERS_VAL = -1
# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

_C.DATA_LOADER.PERSISTENT_WORKERS = False
_C.DATA_LOADER.PERSISTENT_WORKERS_TRAIN = False
# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# Visualgenome Dataset options
# -----------------------------------------------------------------------------
_C.VG = CfgNode()

_C.VG.NUM_OBJECTS_CLASSES = 261
_C.VG.NUM_RELATIONS_CLASSES = 67


# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

# Path to Epic-Kitchens RGB data directory
_C.EPICKITCHENS.VISUAL_DATA_DIR = "/home/gamir/datasets/datasets/EPIC-KITCHENS/epic-kitchens-download-scripts-master/EPIC-KITCHENS"

# Path to Epic-Kitchens Annotation directory
_C.EPICKITCHENS.ANNOTATIONS_DIR = "/home/gamir/datasets/datasets/EPIC-KITCHENS"

# List of EPIC-100 TRAIN files
_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

# List of EPIC-100 VAL files
_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

# List of EPIC-100 TEST files
_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

# Testing split
_C.EPICKITCHENS.TEST_SPLIT = "validation"

# Use Train + Val
_C.EPICKITCHENS.TRAIN_PLUS_VAL = False


# -----------------------------------------------------------------------------
# SSV2 Dataset options
# -----------------------------------------------------------------------------
_C.SSV2 = CfgNode()

# SSV2 data root.
_C.SSV2.DATA_ROOT = '/home/gamir/DER-Roei/datasets/smthsmth'

# SSV2 split, for SmthElse set to 'compositional'.
_C.SSV2.SPLIT = 'compositional'


# -----------------------------------------------------------------------------
# DOH Dataset options
# -----------------------------------------------------------------------------
_C.DOH = CfgNode()

# DOH data root.
_C.DOH.DATA_ROOT = '/home/gamirdataset/datasets/100DOH/downloads'


# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = ""

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    ""
)

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = (
    ""
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# Center crop test
_C.AVA.CENTER_CROP_TEST = True



# -----------------------------------------------------------------------------
# SURREACT Dataset options
# -----------------------------------------------------------------------------

_C.SURREACT = CfgNode()

_C.SURREACT.SURREACT_VERSION = 'ntu/vibe'

_C.SURREACT.IMG_FOLDER = '/home/gamir/ofir1080/datasets/synt/surreact/data/surreact'
_C.SURREACT.MATFILE = 'surreact_data.mat'
_C.SURREACT.INP_RES = 256
_C.SURREACT.NUM_IN_FRAMES = 16
_C.SURREACT.POSE_REP = 'xyz'
_C.SURREACT.SURREACT_VIEWS = '0-45-90-135-180-225-270-315'
_C.SURREACT.JOINTSIX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
_C.SURREACT.RANDFRAMES = 1
_C.SURREACT.USE_SEGM = 'as_target' # '' | 'as_input' | 'mask_rgb'
_C.SURREACT.USE_FLOW = '' # '' | 'as_input'
_C.SURREACT.RANDBGVID = 0
_C.SURREACT.SEGM_RESOLUTION = 28
_C.SURREACT.NUM_CLASSES = 15
_C.SURREACT.SCALE_FACTOR = .25
_C.SURREACT.EVALUATE_VIDEO = 0
_C.SURREACT.HFLIP = 0
_C.SURREACT.NUM_CROPS = 1

_C.SURREACT.DATA_MEAN = [0.5, 0.5, 0.5]
_C.SURREACT.DATA_STD = [1.0, 1.0, 1.0]


_C.PHAV =  CfgNode()
_C.PHAV.DATA_ROOT = '/home/gamir/ofir1080/datasets/synt/phav'
_C.PHAV.TARGET_TYPES = ['depth_maps', 'semantic_seg']
_C.PHAV.TARGET_RES = [28,28]
_C.PHAV.MAX_DEPTH = 1500


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = True
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input video if set.
# If -1, use input video instead.
_C.DEMO.WEBCAM = -1

# Path to input video for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# only used for detection tasks.
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Path to Detectron2 object detection model pre-trained weights.
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Threshold for choosing predicted bounding boxes by Detectron2.
_C.DEMO.DETECTRON2_THRESH = 0.9
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo video reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
_C.DEMO.CLIP_VIS_SIZE = 10
# Number of processes to run video visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Path to pre-computed predicted boxes
_C.DEMO.PREDS_BOXES = ""
# Whether to run in with multi-threaded video reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Path to ground-truth boxes and labels (optional)
_C.DEMO.GT_BOXES = ""
# The starting second of the video w.r.t bounding boxes file.
_C.DEMO.STARTING_SECOND = 900
# Frames per second of the input video/folder of images.
_C.DEMO.FPS = 30
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1

# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    if cfg.NUM_GPUS > 0 and cfg.TRAIN.ENABLE:
        num_gpus_video = cfg.NUM_GPUS - len(cfg.IMAGE_TRAIN.GPU_IDS)
        num_gpus_images = len(cfg.IMAGE_TRAIN.GPU_IDS)
        assert cfg.TRAIN.BATCH_SIZE % num_gpus_video == 0
        assert cfg.IMAGE_TRAIN.BATCH_SIZE % num_gpus_images == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    
    assert cfg.SVIT.O == 4
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
