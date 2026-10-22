"""Load pinned frozen V-JEPA weights and retain every spatial token.

Inputs are eight RGB frames per available scene camera, scaled to [0, 1] and
resized by the dataset reader to 384x384. Outputs retain four temporal groups
of 24x24 patch vectors; this module does no spatial or camera averaging.
For configured training, use ``tl train --preflight-only`` before ``tl train``.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from .attention import clip_view_mask

BACKBONE_ID = "facebook/vjepa2.1-vitb16-384"
REVISION = "204698b45b3712590f06245fbfba32d3be539812"
CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
DEFAULT_REPO = "processing/upstream/vjepa2"
DEFAULT_CHECKPOINT = "processing/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_backbone_files(repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT):
    """Verify the official revision and weight bytes, returning their provenance.

    ``repo`` and ``checkpoint`` must already exist. Verification is read-only;
    it never downloads weights or repairs a modified upstream checkout.
    """
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    modified = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != REVISION or modified:
        raise ValueError("V-JEPA implementation differs from the pinned official revision")
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("V-JEPA checkpoint hash mismatch")
    return dict(
        id=BACKBONE_ID,
        revision=REVISION,
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint=str(checkpoint),
        repo=str(repo),
        checkpoint_key="ema_encoder",
        rgb_size=384,
        frames=8,
        stride=2,
        tubelet_size=2,
        features="last_normalized_block",
        spatial_pool="none",
        camera_pool="learned_cross_attention",
        temporal_mapping="one_current_frame_query_with_seven_history_frames",
        rgb_preprocessing="direct_square_resize_384_imagenet_normalization",
        frozen=True,
    )


class VJEPABackbone(nn.Module):
    """Frozen video encoder with ImageNet input conditioning.

    ``forward`` maps [videos, 3, 8, 384, 384] RGB to [videos, 4, 576, 768]
    float tokens. Each temporal token group covers a pair of input frames and
    attends within the complete clip. Calling ``train()`` on a parent model
    does not enable backbone dropout or parameter gradients.
    """

    def __init__(self, repo=DEFAULT_REPO, checkpoint=DEFAULT_CHECKPOINT):
        super().__init__()
        self.provenance = verify_backbone_files(repo, checkpoint)
        sys.path.insert(0, str(Path(repo).resolve()))
        from app.vjepa_2_1.models.vision_transformer import vit_base

        self.encoder = vit_base(
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_rope=True,
            uniform_power=False,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        weights = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in saved["ema_encoder"].items()
        }
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.dim = 768
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def train(self, mode=True):
        """Keep the official encoder in evaluation mode under parent training."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[1:] != (3, 8, 384, 384):
            raise ValueError("V-JEPA expects [videos,3,8,384,384]")
        tokens = self.encoder((clips - self.mean) / self.std)
        if tokens.shape[1:] != (4 * 576, self.dim):
            raise ValueError("Unexpected V-JEPA temporal/spatial token layout")
        return tokens.reshape(len(clips), 4, 576, self.dim).float()


@torch.no_grad()
def spatial_video_features(backbone, batch, device, chunk_size=2):
    """Encode available cameras once into [B, 4, 3, 576, 768] memory.

    ``batch.visual.rgb`` has shape [B, 8, 3, 3, 384, 384]; camera availability
    is boolean [B, 8, 3], with True meaning available. Missing camera slots
    become zeros and remain masked by the attention reader. ``chunk_size``
    limits simultaneously encoded camera clips without changing their order.
    Returned features are detached and may be reused during SAM replay.
    """
    if isinstance(batch, dict) and "video_features" in batch:
        raise ValueError("Spatial training requires live RGB, not pooled cached features")
    visual = batch.visual if hasattr(batch, "visual") else batch
    rgb = visual["rgb"] if isinstance(visual, dict) else visual.rgb
    mask = visual["view_mask"] if isinstance(visual, dict) else visual.view_mask
    rgb = rgb.to(device, non_blocking=True)
    available = clip_view_mask(mask.to(device))
    batch_size, frames, views, channels, height, width = rgb.shape
    if (frames, views, channels, height, width) != (8, 3, 3, 384, 384):
        raise ValueError("Expected RGB [batch,8,3,3,384,384]")
    videos = rgb.permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size * views, channels, frames, height, width
    )
    videos = videos[available.flatten()]
    encoded = []
    for part in videos.split(chunk_size):
        tokens = backbone.encoder((part - backbone.mean) / backbone.std)
        if tokens.shape[1:] != (4 * 576, 768):
            raise ValueError("Unexpected full spatial token layout")
        encoded.append(tokens.reshape(len(part), 4, 576, 768).float())
    encoded = torch.cat(encoded)
    restored = encoded.new_zeros(batch_size * views, 4, 576, 768)
    restored[available.flatten()] = encoded
    return restored.reshape(batch_size, views, 4, 576, 768).permute(0, 2, 1, 3, 4).contiguous()
