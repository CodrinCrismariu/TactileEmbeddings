"""Frozen V-JEPA 2.1 clips -> ordered per-frame latents -> sensor codes.

Video tokens represent pairs (0,1), (2,3), ... . A 768 -> 2*latent_dim
projection gives each member its own latent, without resampling touch targets.
"""
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from .aligned_frames import file_hash
from .alignment_model import JointSensorAlignment

MODEL_TYPE = 'joint_vjepa21_alignment_v1'
BACKBONE_ID = 'facebook/vjepa2.1-vitb16-384'
REVISION = '204698b45b3712590f06245fbfba32d3be539812'
CHECKPOINT_SHA256 = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'
DEFAULT_REPO = 'processing/upstream/vjepa2'
DEFAULT_CHECKPOINT = 'processing/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt'


def verify_backbone_files(repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT):
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION or subprocess.check_output(
            ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('V-JEPA implementation differs from pinned official revision')
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('V-JEPA checkpoint hash mismatch')
    return dict(id=BACKBONE_ID, revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA256,
                checkpoint=str(checkpoint), repo=str(repo), checkpoint_key='ema_encoder',
                rgb_size=384, frames=8, stride=2, tubelet_size=2, features='last_normalized_block',
                spatial_pool='mean', camera_pool='masked_mean', temporal_mapping='pair_slots_0_then_1',
                rgb_preprocessing='direct_square_resize_384_imagenet_normalization',
                frozen=True)


class VJEPABackbone(nn.Module):
    def __init__(self, repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT):
        super().__init__()
        self.provenance = verify_backbone_files(repo, checkpoint)
        sys.path.insert(0, str(Path(repo).resolve()))
        from app.vjepa_2_1.models.vision_transformer import vit_base
        # Match official Base factory exactly, without constructing its predictor.
        self.encoder = vit_base(img_size=(384, 384), num_frames=64, tubelet_size=2,
            use_sdpa=True, use_rope=True, uniform_power=False,
            img_temporal_dim_size=1, interpolate_rope=True)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        state = {k.replace('module.', '').replace('backbone.', ''): v
                 for k, v in saved['ema_encoder'].items()}
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.dim = 768
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[1:] != (3, 8, 384, 384):
            raise ValueError('V-JEPA expects [videos,3,8,384,384]')
        tokens = self.encoder((clips - self.mean) / self.std)
        if tokens.shape[1:] != (4 * 24 * 24, self.dim):
            raise ValueError('Unexpected V-JEPA temporal/spatial token shape')
        return tokens.reshape(len(clips), 4, 576, self.dim).float().mean(2)


def clip_view_mask(mask):
    if mask.ndim != 3 or mask.shape[1] != 8:
        raise ValueError('Expected eight-frame view mask')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError('V-JEPA requires camera availability constant within each clip')
    available = mask[:, 0].bool()
    if not available.any(-1).all():
        raise ValueError('Every clip needs an available camera')
    return available


@torch.no_grad()
def video_features(backbone, batch, device, chunk_size=2):
    """[B,8,V,3,384,384] -> [B,4,V,768]; absent views never enter encoder."""
    available = clip_view_mask(batch['view_mask'].to(device))
    rgb = batch['rgb'].to(device)
    b, t, v, c, h, w = rgb.shape
    clips = rgb.permute(0, 2, 3, 1, 4, 5).reshape(b*v, c, t, h, w)
    selected = clips[available.flatten()]
    encoded = torch.cat([backbone(part) for part in selected.split(chunk_size)])
    restored = encoded.new_zeros(b*v, 4, backbone.dim)
    restored[available.flatten()] = encoded
    return restored.reshape(b, v, 4, backbone.dim).permute(0, 2, 1, 3).contiguous()


class PairLatentEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, latent_mode='gaussian'):
        super().__init__()
        self.latent_dim, self.latent_mode = latent_dim, latent_mode
        self.prior_mu = nn.Linear(input_dim, 2 * latent_dim)
        if latent_mode == 'gaussian':
            self.prior_logvar = nn.Linear(input_dim, 2 * latent_dim)

    def encode_context(self, features, view_mask):
        available = clip_view_mask(view_mask)
        if features.ndim != 4 or features.shape[:3] != (len(view_mask), 4, available.shape[1]):
            raise ValueError('Expected [batch,4,cameras,width] pooled video features')
        clean = features.masked_fill(~available[:, None, :, None], 0.)
        return clean.sum(2) / available.sum(-1)[:, None, None]

    def encode(self, features, view_mask):
        context = self.encode_context(features, view_mask)
        mu = self.prior_mu(context).reshape(len(context), 8, self.latent_dim)
        if self.latent_mode == 'deterministic':
            return mu, torch.full_like(mu, -torch.inf)
        logvar = self.prior_logvar(context).reshape_as(mu).clamp(-8., 4.)
        return mu, logvar


class JointVJEPAAlignment(JointSensorAlignment):
    """Reuse sensor-head forward/denormalization, with no DINO-era trunk."""
    def __init__(self, sensor_code_dims, numeric_shapes, statistics, input_dim=768,
                 latent_dim=32, adapter_hidden_dims=(128, 128), latent_mode='gaussian'):
        nn.Module.__init__(self)
        if latent_mode not in ('gaussian', 'deterministic') or latent_dim < 1:
            raise ValueError('Invalid latent mode or size')
        if set(sensor_code_dims) & set(numeric_shapes):
            raise ValueError('Overlapping sensor heads')
        self.latent_mode, self.model_type = latent_mode, MODEL_TYPE
        self.stochastic = latent_mode == 'gaussian'
        self.sensor_code_dims, self.numeric_shapes = dict(sensor_code_dims), dict(numeric_shapes)
        self.visual = PairLatentEncoder(input_dim, latent_dim, latent_mode)
        self.adapters, self.numeric_heads, self.normalization = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for source, dimension in sensor_code_dims.items():
            layers, previous = [], latent_dim
            for hidden in adapter_hidden_dims:
                layers.extend([nn.Linear(previous, hidden), nn.GELU()]); previous = hidden
            layers.append(nn.Linear(previous, dimension))
            self.adapters[source] = nn.Sequential(*layers)
            norm = nn.Module()
            mean = torch.as_tensor(statistics[source]['mean'], dtype=torch.float32)
            std = torch.as_tensor(statistics[source]['std'], dtype=torch.float32)
            if mean.shape != (dimension,) or std.shape != (dimension,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 1e-6).any():
                raise ValueError('Invalid train-only code normalization')
            norm.register_buffer('mean', mean); norm.register_buffer('std', std)
            self.normalization[source] = norm
        for source, shape in numeric_shapes.items():
            self.numeric_heads[source] = nn.Sequential(nn.Linear(latent_dim, int(torch.tensor(shape).prod())), nn.Sigmoid())


def matched_models(model_config, seed=45):
    torch.manual_seed(seed)
    deterministic = JointVJEPAAlignment(**model_config, latent_mode='deterministic')
    torch.manual_seed(seed)
    stochastic = JointVJEPAAlignment(**model_config, latent_mode='gaussian')
    # Extra variance-head construction must not shift adapter initialization.
    stochastic.load_state_dict({**stochastic.state_dict(), **deterministic.state_dict()}, strict=True)
    return {'deterministic': deterministic, 'stochastic': stochastic}


def model_from_state(state):
    if state.get('model_type') != MODEL_TYPE:
        raise ValueError('Not a V-JEPA alignment checkpoint')
    model = JointVJEPAAlignment(**state['model_config'])
    model.load_state_dict(state['model'], strict=True)
    return model
