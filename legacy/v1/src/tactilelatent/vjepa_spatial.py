"""Frozen full V-JEPA tokens -> attentive queries -> configurable frame latents.

No arithmetic spatial/camera pooling and no dataset feature cache.
"""
import math

import torch
from torch import nn

from .vjepa_model import VJEPABackbone, JointVJEPAAlignment, clip_view_mask

MODEL_TYPE = 'joint_vjepa21_spatial_attention_v1'


@torch.no_grad()
def spatial_video_features(backbone, batch, device, chunk_size=2):
    if 'video_features' in batch:
        raise ValueError('Spatial training requires live RGB, never pooled cached features')
    rgb = batch['rgb'].to(device, non_blocking=True)
    available = clip_view_mask(batch['view_mask'].to(device))
    b,t,v,c,h,w=rgb.shape
    if (t,v,c,h,w)!=(8,3,3,384,384):
        raise ValueError('Expected native clip contract [B,8,3,3,384,384]')
    videos=rgb.permute(0,2,3,1,4,5).reshape(b*v,c,t,h,w)[available.flatten()]
    encoded=[]
    for part in videos.split(chunk_size):
        tokens=backbone.encoder((part-backbone.mean)/backbone.std)
        if tokens.shape[1:]!=(4*576,768):
            raise ValueError('Unexpected full spatial token layout')
        encoded.append(tokens.reshape(len(part),4,576,768).float())
    encoded=torch.cat(encoded)
    restored=encoded.new_zeros(b*v,4,576,768)
    restored[available.flatten()]=encoded
    return restored.reshape(b,v,4,576,768).permute(0,2,1,3,4).contiguous()


def positional_encoding(length,width):
    positions=torch.arange(length,dtype=torch.float32)[:,None]
    scale=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
    result=torch.zeros(length,width)
    result[:,0::2]=torch.sin(positions*scale)
    result[:,1::2]=torch.cos(positions*scale)
    return result


class SpatialFrameEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, width=256, heads=4,
                 layers=2, feedforward=1024, dropout=.1, grid_size=24,
                 prediction_mode='all_frames'):
        super().__init__()
        if width%4 or width%heads or layers<1 or grid_size<1:
            raise ValueError('Invalid attention dimensions')
        if prediction_mode not in ('all_frames','current_frame'):
            raise ValueError('Unknown frame prediction mode')
        self.latent_dim,self.grid_size=latent_dim,grid_size
        self.prediction_mode=prediction_mode
        self.project=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width))
        self.camera_roles=nn.Embedding(3,width)  # scene/main, left, right
        self.frame_queries=nn.Parameter(torch.empty(1 if prediction_mode=='current_frame' else 8,width))
        self.pair_slot=nn.Embedding(2,width)
        nn.init.trunc_normal_(self.frame_queries,std=.02)
        row=positional_encoding(grid_size,width//2)[:,None,:].expand(-1,grid_size,-1)
        col=positional_encoding(grid_size,width//2)[None,:,:].expand(grid_size,-1,-1)
        self.register_buffer('spatial_position',torch.cat((row,col),-1).reshape(grid_size**2,width))
        self.register_buffer('group_position',positional_encoding(4,width))
        self.register_buffer('frame_position',positional_encoding(8,width))
        layer=nn.TransformerDecoderLayer(width,heads,feedforward,dropout,
            activation='gelu',batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,layers)
        self.bottleneck=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.GELU(),nn.Linear(128,latent_dim))

    def encode_context(self,features,view_mask,camera_roles=None):
        available=clip_view_mask(view_mask)
        b=features.shape[0];p=self.grid_size**2
        if features.ndim!=5 or features.shape[1:4]!=(4,3,p):
            raise ValueError('Expected full [B,4,3,patches,width] spatial tokens')
        if camera_roles is None:
            camera_roles=torch.arange(3,device=features.device)[None].expand(b,-1)
        if camera_roles.shape!=(b,3):
            raise ValueError('Invalid verified camera-role mapping')
        # Mask before projection too: absent NaNs must never poison attention.
        clean=features.masked_fill(~available[:,None,:,None,None],0.)
        memory=self.project(clean)
        memory=(memory+self.spatial_position[None,None,None]
                +self.group_position[None,:,None,None]
                +self.camera_roles(camera_roles)[:,None,:,None])
        memory=memory.reshape(b,4*3*p,-1)
        padding=(~available[:,None,:,None]).expand(b,4,3,p).reshape(b,-1)
        slots=(torch.tensor([7],device=features.device) if self.prediction_mode=='current_frame'
               else torch.arange(8,device=features.device))
        query=(self.frame_queries+self.frame_position[slots]
               +self.pair_slot(slots%2))[None].expand(b,-1,-1)
        return self.decoder(query,memory,memory_key_padding_mask=padding)

    def encode(self,features,view_mask,camera_roles=None):
        mu=self.bottleneck(self.encode_context(features,view_mask,camera_roles))
        return mu,torch.full_like(mu,-torch.inf)


class JointSpatialVJEPA(JointVJEPAAlignment):
    def __init__(self,sensor_code_dims,numeric_shapes,statistics,input_dim=768,latent_dim=32,
                 adapter_hidden_dims=(128,128),latent_mode='deterministic',width=256,heads=4,
                 layers=2,feedforward=1024,dropout=.1,grid_size=24,prediction_mode='all_frames'):
        if latent_mode!='deterministic':
            raise ValueError('This experiment is deterministic, without KL or variance head')
        super().__init__(sensor_code_dims,numeric_shapes,statistics,input_dim,latent_dim,
                         adapter_hidden_dims,latent_mode)
        self.prediction_mode=prediction_mode
        self.visual=SpatialFrameEncoder(input_dim,latent_dim,width,heads,layers,feedforward,dropout,grid_size,prediction_mode)
        self.model_type=MODEL_TYPE

    def forward(self,features,view_mask,source,sample_latent=None):
        roles=torch.arange(3,device=features.device)[None].expand(len(features),-1).clone()
        # Native manifest puts FreeTacMan's scene camera in slot 1; do not
        # mislabel it as a left wrist or change any physical camera mapping.
        if source=='freetacman':roles[:,1]=0
        mu,logvar=self.visual.encode(features,view_mask,roles)
        output=dict(mu=mu,logvar=logvar,latent_mode='deterministic')
        if source in self.adapters:output['sensor_code']=self.adapters[source](mu)
        elif source in self.numeric_heads:
            output['pred']=self.numeric_heads[source](mu).reshape(*mu.shape[:-1],*self.numeric_shapes[source])
        else:raise ValueError(f'Untrained source: {source}')
        return output


def model_from_state(state):
    if state.get('model_type')!=MODEL_TYPE:raise ValueError('Not a spatial V-JEPA checkpoint')
    model=JointSpatialVJEPA(**state['model_config'])
    model.load_state_dict(state['model'],strict=True)
    return model
