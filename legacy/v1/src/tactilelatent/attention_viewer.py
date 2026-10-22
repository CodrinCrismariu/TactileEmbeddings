"""Read-only cross-attention inspection for the spatial V-JEPA frame queries.

Compute query/key probabilities alongside the ordinary forward. Never replace
the model forward, request a different attention kernel, or alter predictions.
"""
import base64
import math

import numpy as np
import torch
from torch.nn import functional as F


def cross_attention_weights(module,args,kwargs):
    """Reconstruct eval-mode MHA probabilities from its actual query/key input."""
    if (module.training or not module.batch_first or not module._qkv_same_embed_dim
            or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn
            or kwargs.get('is_causal',False)):
        raise ValueError('Unsupported attention layout for spatial inspection')
    query,key=args[:2]
    width=module.embed_dim;heads=module.num_heads;head_width=width//heads
    # Match MHA's sequence-first packed projection for the cross-attention case.
    bias=module.in_proj_bias
    q=F.linear(query.transpose(0,1),module.in_proj_weight[:width],None if bias is None else bias[:width])
    kv=F.linear(key.transpose(0,1),module.in_proj_weight[width:],None if bias is None else bias[width:])
    k=kv[...,:width]
    b=query.shape[0];queries=query.shape[1];tokens=key.shape[1]
    q=q.reshape(queries,b,heads,head_width).permute(1,2,0,3)
    k=k.reshape(tokens,b,heads,head_width).permute(1,2,0,3)
    with torch.autocast(query.device.type,enabled=False):
        score=(q.float()@k.float().transpose(-1,-2))/math.sqrt(head_width)
        padding=kwargs.get('key_padding_mask')
        if padding is not None:
            if padding.dtype==torch.bool:score=score.masked_fill(padding[:,None,None,:],-torch.inf)
            else:score=score+padding[:,None,None,:].float()
        mask=kwargs.get('attn_mask')
        if mask is not None:
            mask=mask[None,None] if mask.ndim==2 else mask.reshape(b,heads,queries,tokens)
            if mask.dtype==torch.bool:score=score.masked_fill(mask,-torch.inf)
            else:score=score+mask.float()
        return score.softmax(-1)


@torch.inference_mode()
def capture_cross_attention(visual,features,view_mask,query_index=0,source=None):
    """Return [layers,heads,4,3,grid_y,grid_x] for the selected frame query.

    Hooks only observe MHA inputs. Later layers therefore see the exact normal
    model outputs, including the normal SDPA kernel and residual connections.
    """
    if visual.training or len(features)!=1:
        raise ValueError('Attention inspection requires one eval-mode clip')
    layers=getattr(getattr(visual,'decoder',None),'layers',None)
    if layers is None or not hasattr(visual,'grid_size'):
        raise ValueError('Attention overlays require a spatial-attention V-JEPA model')
    collected={};handles=[]
    def hook(index):
        def collect(module,args,kwargs):
            weights=cross_attention_weights(module,args,kwargs)
            if not 0<=query_index<weights.shape[2]:raise ValueError('Invalid frame query')
            collected[index]=weights[0,:,query_index].cpu()
        return collect
    try:
        for i,layer in enumerate(layers):
            handles.append(layer.multihead_attn.register_forward_pre_hook(hook(i),with_kwargs=True))
        roles=torch.arange(3,device=features.device)[None]
        if source=='freetacman':roles=roles.clone();roles[:,1]=0
        visual.encode(features,view_mask,roles)
    finally:
        for handle in handles:handle.remove()
    if len(collected)!=len(layers):raise ValueError('Not every cross-attention layer was observed')
    weights=torch.stack([collected[i] for i in range(len(layers))])
    grid=visual.grid_size
    weights=weights.reshape(len(layers),weights.shape[1],4,3,grid,grid).float()
    if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('Nonfinite attention weights')
    if not torch.allclose(weights.sum((2,3,4,5)),torch.ones(weights.shape[:2]),atol=1e-5):
        raise ValueError('Attention probability mass is not normalized')
    return weights


def encode_weights(weights):
    values=weights.numpy().astype('<f4',copy=False)
    return dict(shape=list(values.shape),dtype='float32-le',weights_b64=base64.b64encode(values.tobytes()).decode())
