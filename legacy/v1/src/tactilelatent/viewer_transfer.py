"""Inference-only adapters for evaluated transfer heads, including fit z-scores."""
import torch
from torch import nn

from .frozen_probe import head


def validate_normalization(saved,dimension):
    mode=saved.get('feature_normalization','none')
    if mode=='none':return
    if mode!='fit_zscore':raise ValueError('Unsupported transfer feature normalization')
    mean,std=saved['mean'],saved['std']
    if (mean.shape!=(dimension,) or std.shape!=(dimension,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std<1e-6).any()):
        raise ValueError('Invalid fit-only transfer normalization')


class NormalizedTransferHead(nn.Module):
    def __init__(self,network,mean,std):
        super().__init__();self.network=network
        self.register_buffer('mean',mean.detach().clone().float())
        self.register_buffer('std',std.detach().clone().float())

    def forward(self,latent):
        # Probe fitting/scoring used float32 even when frozen visual extraction
        # used BF16 autocast. Preserve that boundary in the viewer too.
        with torch.autocast(latent.device.type,enabled=False):
            return self.network((latent.float()-self.mean)/self.std)


def load_transfer_head(saved,dimension):
    validate_normalization(saved,dimension)
    network=head(dimension,6,hidden_dims=saved['config'].get('hidden_dims',[512]))
    network.load_state_dict(saved['head'],strict=True)
    if saved.get('feature_normalization')=='fit_zscore':
        return NormalizedTransferHead(network,saved['mean'],saved['std'])
    return network
