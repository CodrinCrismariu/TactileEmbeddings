"""Image-only EgoTouch pressure predictor.

The model has two parts:
1. A frozen DINOv3 image encoder turns each RGB frame into patch features.
2. A trainable visual/temporal model turns those features into one Gaussian
   latent per frame, then decodes the latent into two pressure grids.

The pressure target is NEVER an input to this model. For a full teaching guide
with tensor shapes and parameter counts, see ARCHITECTURE_GUIDE.md.
"""

import torch
from torch import nn

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


class DinoBackbone(nn.Module):
    """Frozen DINOv3-B/16, returning only its 14x14 image-patch tokens.

    At 224x224 resolution: RGB [images,3,224,224] -> [images,196,768].
    DINO also makes CLS/register tokens internally; they are not passed on.
    """

    def __init__(self, model_id=MODEL_ID, revision=REVISION, cache_dir="processing/hf_cache"):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        # DINO supplies pretrained visual features; only the later model learns.
        self.encoder.requires_grad_(False).eval()
        self.dim = self.encoder.config.hidden_size
        self.patch_size = self.encoder.config.patch_size
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        """Keep DINO's own dropout/statistics in evaluation mode always."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb):
        """Normalize RGB, run DINO without gradients, and keep patch tokens."""
        patches = (rgb.shape[-2] // self.patch_size) * (rgb.shape[-1] // self.patch_size)
        output = self.encoder(pixel_values=(rgb - self.mean) / self.std).last_hidden_state
        return output[:, -patches:]  # Discard CLS and register tokens.


class DividedSpaceTimeBlock(nn.Module):
    """One transformer block that separates time and image-space attention.

    Input/output: [batch, time, patches, width]. This block is bidirectional
    in time: it has no causal mask, so earlier frames can use later clip frames.
    """

    def __init__(self, width, heads, dropout=.1):
        super().__init__()
        # Short attribute names are retained for checkpoint compatibility:
        # tn/ta = temporal norm/attention; sn/sa = spatial norm/attention;
        # fn/ff = feed-forward norm/network.
        self.tn = nn.LayerNorm(width)
        self.sn = nn.LayerNorm(width)
        self.fn = nn.LayerNorm(width)
        self.ta = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.sa = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, x):
        batch, times, patches, width = x.shape

        # For EACH image position, attend across the 8 clip times.
        # [B,T,N,D] -> [B*N,T,D]; every sequence is one patch over time.
        y = x.permute(0, 2, 1, 3).reshape(batch * patches, times, width)
        y = y + self.ta(self.tn(y), self.tn(y), self.tn(y), need_weights=False)[0]

        # For EACH time, attend across the 196 image-patch positions.
        # [B*N,T,D] -> [B*T,N,D]; every sequence is one frame's patch grid.
        x = y.reshape(batch, patches, times, width).permute(0, 2, 1, 3)
        x = x.reshape(batch * times, patches, width)
        x = x + self.sa(self.sn(x), self.sn(x), self.sn(x), need_weights=False)[0]

        # A feed-forward network refines each token; all three additions are
        # residual connections that help retain and train earlier features.
        x = x + self.ff(self.fn(x))
        return x.reshape(batch, times, patches, width)


class EgoTouchLatentModel(nn.Module):
    """Predict a Gaussian tactile latent per frame using image features alone.

    Args to forward:
        features: [B,T,3,196,768] frozen DINO patch vectors for three cameras.
        view_mask: [B,T,3] booleans marking cameras the model may use.
        sample_latent: None uses samples in training and the mean at evaluation;
            True or False overrides that behavior.

    Returns pressure [B,T,2,21,21], mu [B,T,32], logvar [B,T,32] at the
    active configuration. The same weights work for ego-only and all views.
    """

    def __init__(self, input_dim=768, width=768, heads=8, temporal_layers=3,
                 cross_view_layers=2, latent_dim=32, dropout=.1, stochastic=True):
        super().__init__()
        # Adapt frozen DINO's patch vectors to the learned model's width.
        self.project = nn.Linear(input_dim, width)
        self.view_embeddings = nn.Parameter(torch.randn(3, width) * .02)

        # Cross-view attention operates on THREE camera summaries per frame,
        # not all 3*196 patch tokens. Its result controls one camera weight
        # per time; that weight applies to all patches from that camera.
        cv_layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                               batch_first=True, norm_first=True)
        self.cross_view = nn.TransformerEncoder(cv_layer, cross_view_layers,
                                                enable_nested_tensor=False)
        self.view_gate = nn.Sequential(nn.Linear(width, width // 4), nn.GELU(),
                                       nn.Linear(width // 4, 1))
        self.view_norm = nn.LayerNorm(width)
        # Fixed positional values identify the time within a clip. This is a
        # buffer, not a learned parameter, and is regenerated when loading.
        self.register_buffer("temporal_pe", self._sinusoid(64, width), persistent=False)
        self.temporal = nn.ModuleList([DividedSpaceTimeBlock(width, heads, dropout)
                                       for _ in range(temporal_layers)])
        self.temporal_norm = nn.LayerNorm(width)
        self.image_pool = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        # Two heads describe a diagonal Gaussian: mean and log(variance).
        # Historical checkpoint keys use "prior_*"; keeping those names lets
        # us load the existing image-only VAE weights without changing them.
        self.prior_mu = nn.Linear(width, latent_dim)
        self.prior_logvar = nn.Linear(width, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(512, 2 * 21 * 21), nn.Sigmoid())
        self.stochastic = stochastic

    @staticmethod
    def _sinusoid(length, width):
        """Make fixed time-position vectors with no learned weights."""
        position = torch.arange(length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2) * (-torch.log(torch.tensor(10000.)) / width))
        pe = torch.zeros(length, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        return pe

    def decode_latent(self, z):
        """Map [...,32] latent vectors to [...,2,21,21] pressure grids."""
        return self.decoder(z).reshape(*z.shape[:-1], 2, 21, 21)

    def encode_context(self, features, view_mask):
        """Shared spatial/temporal trunk, before the latent projection heads."""
        batch, times, views, _, _ = features.shape
        if views != 3 or view_mask.shape != (batch, times, 3):
            raise ValueError("Expected three camera slots and B,T,3 camera mask")
        if times > len(self.temporal_pe) or not view_mask.any(dim=-1).all():
            raise ValueError("Invalid temporal length or timestep without a camera")

        # Step 1: project each DINO patch and add a camera-identity vector.
        # x: [B,T,3,196,768] in the active configuration.
        x = self.project(features) + self.view_embeddings[None, None, :, None]

        # Step 2: let the three camera summaries inspect each other and
        # produce softmax camera weights. A masked camera gets weight zero.
        summaries = x.mean(dim=3).reshape(batch * times, views, -1)
        active = view_mask.reshape(batch * times, views)
        summaries = self.cross_view(summaries, src_key_padding_mask=~active)
        weights = self.view_gate(summaries).squeeze(-1).masked_fill(~active, -torch.inf).softmax(-1)

        # Fuse the three cameras into ONE patch grid per frame: [B,T,196,768].
        x = (x * weights.reshape(batch, times, views, 1, 1)).sum(dim=2)
        x = self.view_norm(x) + self.temporal_pe[:times][None, :, None]

        # Step 3: reason across time, then across image locations, three times.
        for block in self.temporal:
            x = block(x)

        # Step 4: average image patches; produce a Gaussian per frame.
        # context: [B,T,768]; mu/logvar: [B,T,32].
        context = self.image_pool(self.temporal_norm(x).mean(dim=2))
        return context

    def encode(self, features, view_mask):
        """Encode DINO features into image-conditioned Gaussian parameters."""
        context = self.encode_context(features, view_mask)
        mu = self.prior_mu(context)
        logvar = self.prior_logvar(context).clamp(-8, 4)

        return mu, logvar

    def forward(self, features, view_mask, sample_latent=None):
        mu, logvar = self.encode(features, view_mask)

        # Step 5: sample by reparameterization during training so gradients
        # flow through both Gaussian heads. At normal evaluation use z=mu.
        if sample_latent is None:
            sample_latent = self.stochastic and self.training
        z = mu + torch.randn_like(mu) * (.5 * logvar).exp() if sample_latent else mu

        # A shared decoder maps the same 32D z to both 21x21 hand grids.
        return {"pressure": self.decode_latent(z), "mu": mu, "logvar": logvar}


def load_vae_weights(model, weights):
    """Read old image-only VAE checkpoints, ignoring known inactive tensors.

    The old file stored extra experiment modules that the VAE forward path
    never used. We require every active VAE tensor to be present and reject
    unexpected names other than those known historical extras.
    """
    names = model.state_dict()
    selected = {name: tensor for name, tensor in weights.items() if name in names}
    missing = names.keys() - selected.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing VAE tensors: {sorted(missing)}")
    unexpected = weights.keys() - names.keys()
    allowed = ("tactile_encoder.", "posterior.", "posterior_mu.",
               "posterior_logvar.", "posterior_pose.")
    if any(not name.startswith(allowed) for name in unexpected):
        raise ValueError(f"Unsupported checkpoint tensors: {sorted(unexpected)}")
    model.load_state_dict(selected)
    return len(unexpected)
