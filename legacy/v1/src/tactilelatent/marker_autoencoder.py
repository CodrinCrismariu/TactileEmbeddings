"""Masked point-set tactile AE. Every displacement passes through its code."""
import torch
from torch import nn

from .marker_model import MarkerDisplacementHead,marker_loss


class MarkerAutoencoder(nn.Module):
    def __init__(self,latent_dim=32,width=128,queries=8,heads=4,displacement_scale=1.,architecture='marker_set'):
        super().__init__()
        if architecture!='marker_set' or min(latent_dim,width,queries,heads)<1 or width%heads or displacement_scale<=0:
            raise ValueError('Invalid marker autoencoder configuration')
        self.config=dict(architecture=architecture,latent_dim=latent_dim,width=width,queries=queries,
                         heads=heads,displacement_scale=float(displacement_scale))
        self.register_buffer('displacement_scale',torch.tensor(float(displacement_scale)))
        self.surface=nn.Embedding(2,8)
        self.point_encoder=nn.Sequential(nn.Linear(13,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.queries=nn.Parameter(torch.empty(queries,width));nn.init.normal_(self.queries,std=.02)
        self.null=nn.Parameter(torch.zeros(1,1,width))
        self.attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.to_code=nn.Sequential(nn.Flatten(),nn.Linear(queries*width,width),nn.GELU(),nn.Linear(width,latent_dim))
        self.decoder=MarkerDisplacementHead(latent_dim,width)

    def encode(self,target,valid,reference,surface):
        if target.ndim!=3 or target.shape[-1]!=2 or valid.shape!=target.shape[:-1] or reference.shape!=target.shape or surface.shape!=valid.shape:
            raise ValueError('Expected [frames,points,2] vectors/geometry and matching masks/surfaces')
        valid=valid.bool() & torch.isfinite(target).all(-1)
        clean=torch.where(valid[...,None],target,0.).float()/self.displacement_scale
        features=torch.cat((clean,reference.float(),valid[...,None].float(),self.surface(surface)),dim=-1)
        memory=self.point_encoder(features)
        # Null support prevents NaNs for a completely missing frame; its code
        # is never supervised or included in teacher statistics.
        memory=torch.cat((memory,self.null.expand(len(memory),-1,-1)),dim=1)
        padding=torch.cat((~valid,torch.zeros(len(valid),1,dtype=torch.bool,device=valid.device)),dim=1)
        query=self.queries[None].expand(len(memory),-1,-1)
        pooled,_=self.attention(query,memory,memory,key_padding_mask=padding,need_weights=False)
        return self.to_code(pooled)

    def decode(self,code,reference,surface):
        # Geometry contains only calibration coordinates/surface IDs. Current
        # displacement, validity, and encoder point features never skip the code.
        return self.decoder(code[:,None],reference,surface)[:,0]*self.displacement_scale

    def forward(self,target,valid,reference,surface):
        return self.decode(self.encode(target,valid,reference,surface),reference,surface)

    def loss(self,prediction,target,valid,surface):
        return marker_loss(prediction[:,None],target[:,None],valid[:,None],surface,self.displacement_scale)


def load_marker_autoencoder(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('model_type')!='marker_autoencoder_v1':raise ValueError('Expected a marker-vector AE')
    model=MarkerAutoencoder(**state['model_config']);model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval().requires_grad_(False),state
