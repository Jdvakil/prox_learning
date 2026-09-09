# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

try:
    import IPython
    e = IPython.embed
except ImportError:
    e = None


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names,
                 action_dim=None, n_proximity_sensors=0, prox_tokens_per_sensor=1,
                 prox_feat_dim=3, proximity_feature_dim=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state (qpos) dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            action_dim: action vector dimension. Defaults to state_dim (Aloha-style symmetric)
                but may differ — e.g. Franka skin tasks have qpos=9 (arm7+2 fingers) and
                action=8 (arm7+gripper_cmd1).
            n_proximity_sensors: when > 0, the model expects an extra
                `proximity_positions` argument of shape
                (B, n_proximity_sensors, prox_feat_dim) in `forward` and
                projects+concatenates extra encoder tokens. Default 0 keeps the
                model bit-identical to vanilla ACT. For the P+ACT CVAE-feature
                fusion this is 1 (a single global skin feature).
            prox_tokens_per_sensor: K. Each of the `n_proximity_sensors`
                feature vectors is expanded into K encoder tokens (default 1, the
                original behaviour). K>1 gives prox total token-count parity with
                image features so cross-attention mass is comparable across
                modalities.
            prox_feat_dim: dimension of each proximity feature vector fed in
                `proximity_positions`. 3 (default) is the original predicted-3D-
                position fusion; 256 is the Safety-CVAE decoder-trunk feature;
                7 is the CVAE retreat-delta. Sets `input_proj_proximity`'s
                in_features.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        if action_dim is None:
            action_dim = state_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_proximity_sensors = int(n_proximity_sensors)
        self.prox_tokens_per_sensor = int(prox_tokens_per_sensor)
        if proximity_feature_dim is not None:
            prox_feat_dim = proximity_feature_dim
        self.prox_feat_dim = int(prox_feat_dim)
        self.proximity_feature_dim = self.prox_feat_dim
        if self.prox_tokens_per_sensor < 1:
            raise ValueError(
                f"prox_tokens_per_sensor must be >= 1, got {self.prox_tokens_per_sensor}"
            )
        if self.prox_feat_dim < 1:
            raise ValueError(
                f"prox_feat_dim must be >= 1, got {self.prox_feat_dim}"
            )
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = state_dim + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # Position embedding covers [latent, proprio] plus K slots per
        # proximity sensor (when enabled). Default size matches vanilla ACT.
        n_prox_tokens = self.n_proximity_sensors * self.prox_tokens_per_sensor
        self.additional_pos_embed = nn.Embedding(2 + n_prox_tokens, hidden_dim)
        if self.n_proximity_sensors > 0:
            # Per-sensor projection from a prox_feat_dim feature vector into K
            # hidden_dim tokens. Shared across all sensors.
            self.input_proj_proximity = nn.Linear(
                self.prox_feat_dim, self.prox_tokens_per_sensor * hidden_dim
            )
        else:
            self.input_proj_proximity = None

    def forward(self, qpos, image, env_state, actions=None, is_pad=None,
                proximity_positions=None, image_dropped=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        proximity_positions: when self.n_proximity_sensors > 0, a tensor of
            shape (batch, n_proximity_sensors, prox_feat_dim) holding the
            proximity conditioning feature(s) — for P+ACT, the frozen
            Safety-CVAE skin feature, shape (batch, 1, prox_feat_dim). When None
            and n_proximity_sensors == 0, the model is bit-identical to vanilla
            ACT.
        image_dropped: optional (batch,) bool mask from modality dropout. On
            True samples the sampled CVAE style latent z is zeroed (matching
            the inference branch), so z — inferred from the ground-truth action
            chunk at training time — cannot leak the answer when the image
            tokens carry nothing. None (default) changes nothing.
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            if image_dropped is not None:
                # Modality dropout: zero the style latent for vision-dropped
                # samples. Zeroing latent_sample (not latent_input) reproduces
                # the inference branch below exactly, latent_out_proj bias
                # included. KL is still computed on all samples.
                latent_sample = torch.where(image_dropped.unsqueeze(1),
                                            torch.zeros_like(latent_sample),
                                            latent_sample)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Project proximity positions into K hidden_dim tokens per sensor.
            # Shape: (n_proximity_sensors * K, bs, hidden_dim) for the transformer.
            proximity_input = None
            if self.n_proximity_sensors > 0:
                if proximity_positions is None:
                    raise ValueError(
                        "DETRVAE was built with n_proximity_sensors > 0 but "
                        "forward() got proximity_positions=None"
                    )
                if proximity_positions.shape[1] != self.n_proximity_sensors:
                    raise ValueError(
                        f"proximity_positions has {proximity_positions.shape[1]} "
                        f"sensors but model expects {self.n_proximity_sensors}"
                    )
                if proximity_positions.ndim != 3 or proximity_positions.shape[2] != self.prox_feat_dim:
                    raise ValueError(
                        "proximity feature width "
                        f"{tuple(proximity_positions.shape)} but model expects "
                        f"(B, {self.n_proximity_sensors}, {self.prox_feat_dim})"
                    )
                K = self.prox_tokens_per_sensor
                hidden_dim = self.input_proj_proximity.out_features // K
                bs_ = proximity_positions.shape[0]
                # (B, N, 3) -> (B, N, K*hidden) -> (B, N*K, hidden) -> (N*K, B, hidden)
                proximity_input = self.input_proj_proximity(proximity_positions)
                proximity_input = proximity_input.reshape(
                    bs_, self.n_proximity_sensors * K, hidden_dim
                )
                proximity_input = proximity_input.permute(1, 0, 2)

            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, proximity_input)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.state_dim = state_dim
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=state_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: state_dim
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = getattr(args, 'state_dim', 14)  # Use provided state_dim or default to 14
    action_dim = getattr(args, 'action_dim', state_dim)

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    n_proximity_sensors = int(getattr(args, "n_proximity_sensors", 0) or 0)
    prox_tokens_per_sensor = int(getattr(args, "prox_tokens_per_sensor", 1) or 1)
    prox_feat_dim = int(getattr(args, "prox_feat_dim", 3) or 3)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        action_dim=action_dim,
        n_proximity_sensors=n_proximity_sensors,
        prox_tokens_per_sensor=prox_tokens_per_sensor,
        prox_feat_dim=prox_feat_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

