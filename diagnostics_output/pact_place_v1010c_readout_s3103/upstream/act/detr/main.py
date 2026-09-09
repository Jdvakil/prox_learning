# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--experiment_manifest', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    # new flags added by the pla_house1_mug pipeline (--onscreen_render is
    # already declared above by the original ACT stub list).
    parser.add_argument('--use_wandb', action='store_true')
    # build_ACT_model_and_optimizer re-parses the full sys.argv, so every imitate_episodes
    # flag must be accepted here (as a no-op) or model build dies with "unrecognized
    # arguments". --no_wandb is one such flag.
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='act-pla-house1')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    # action_dim is plumbed through args_override in build_ACT_model_and_optimizer,
    # so it doesn't need a CLI flag, but declaring it keeps argparse from
    # complaining if someone passes it on the command line.
    parser.add_argument('--action_dim', type=int, default=None)
    # P+ACT: number of proximity sensors. Default 0 makes the model bit-identical
    # to vanilla ACT (no proximity tokens, no extra Linear, the additional
    # position embedding stays at (2, hidden_dim)).
    parser.add_argument('--n_proximity_sensors', type=int, default=0)
    # P+ACT: K tokens per sensor. K=1 (default) = original behaviour.
    # K>1 expands each prox feature into K hidden_dim encoder tokens so the
    # prox stream has total-token parity with the image stream (~160 image tokens).
    parser.add_argument('--prox_tokens_per_sensor', type=int, default=1)
    # P+ACT: dimension of each proximity feature vector in proximity_positions.
    # 3 = original predicted-3D-position fusion; 256 = Safety-CVAE decoder-trunk
    # feature; 7 = CVAE retreat-delta. Plumbed through args_override (policy_config),
    # declared here so the re-parse in build_ACT_model_and_optimizer accepts it.
    parser.add_argument('--prox_feat_dim', type=int, default=3)
    parser.add_argument('--proximity_feature_dim', type=int, default=None)
    # P+ACT: which frozen Safety-CVAE feature to inject ('trunk' 256-d or 'delta' 7-d).
    # Consumed by imitate_episodes.py to build the extractor; a no-op here.
    parser.add_argument('--prox_feature', type=str, default='raw',
                        choices=('trunk', 'delta', 'raw', 'peak_closeness',
                                 'nearest_surface', 'surface_embedding',
                                 'xyz', 'embedding'))
    parser.add_argument('--prox_layout', type=str, default='per_sensor',
                        choices=('global', 'per_sensor'))
    parser.add_argument('--prox_pool', type=str, default=None, choices=('mean', 'min'))
    # P+ACT: trainer-side flags consumed by `pact.act_prox.imitate_episodes_with_prox`.
    # ACT's internal argparse must accept them here (as no-ops) so they don't
    # error out when the trainer constructs the policy.
    parser.add_argument('--use_proximity', action='store_true')
    parser.add_argument('--prox_encoder_ckpt', type=str, default=None)
    parser.add_argument('--finetune_prox_encoder', action='store_true')
    parser.add_argument('--prox_policy_tap', type=str, default=None,
                        choices=('embedding', 'readout', 'xyz'))
    parser.add_argument('--prox_encoder_lr', type=float, default=None)
    parser.add_argument('--prox_mapping_json', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    # FACTR visual-curriculum flags consumed by imitate_episodes.py; declared here
    # as no-ops so the sys.argv re-parse in build_*_model_and_optimizer accepts them.
    parser.add_argument('--blur_sigma0', type=float, default=0.0)
    parser.add_argument('--blur_curriculum_steps', type=int, default=None)
    parser.add_argument('--blur_mode', type=str, default='curriculum',
                        choices=('curriculum', 'constant'))
    # Modality-dropout flags consumed by imitate_episodes.py; declared here as
    # no-ops so the sys.argv re-parse in build_*_model_and_optimizer accepts them.
    parser.add_argument('--image_dropout_p', type=float, default=0.0)
    parser.add_argument('--prox_dropout_p', type=float, default=0.0)
    parser.add_argument('--image_dropout_mode', type=str, default='all',
                        choices=('all', 'single'))
    parser.add_argument('--no_zero_latent_on_drop', action='store_true')

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

