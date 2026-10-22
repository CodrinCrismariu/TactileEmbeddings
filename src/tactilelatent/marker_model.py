"""Sensor-specific displacement heads for a shared per-frame visual latent."""
import torch
from torch import nn
from torch.nn import functional as F

from .marker_targets import MARKER_SOURCES


class MarkerDisplacementHead(nn.Module):
    """Predict signed (dx,dy) in marker-spacing units at reference coordinates.

    Coordinates and surface identity are static geometry, never the measured
    current displacement. The variable-size point representation avoids
    assigning a universal marker index across episodes/sensors.
    """
    def __init__(self,latent_dim=32,hidden=128,surfaces=2):
        super().__init__()
        self.surface=nn.Embedding(surfaces,8)
        self.mlp=nn.Sequential(nn.Linear(latent_dim+2+8,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.normal_(self.mlp[-1].weight,std=.001)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,z,reference,surface):
        if z.ndim!=3 or reference.ndim!=3 or reference.shape[-1]!=2 or surface.shape!=reference.shape[:2]:
            raise ValueError('Expected frame latents [B,T,D] and marker geometry [B,N,2]')
        geometry=torch.cat((reference,self.surface(surface)),dim=-1)
        value=torch.cat((z[:,:,None,:].expand(-1,-1,reference.shape[1],-1),
                         geometry[:,None,:,:].expand(-1,z.shape[1],-1,-1)),dim=-1)
        return self.mlp(value)  # Signed, unbounded output; no sigmoid/clipping.


def marker_loss(prediction,target,valid,surface,scale=1.):
    """Robust point loss, equal weighting of supported frames/surfaces.

    Lost markers and NaNs do not contribute, including to gradients. Whole
    missing frames/surfaces are excluded rather than treated as stationary.
    """
    if prediction.shape!=target.shape or valid.shape!=target.shape[:-1] or surface.shape!=(target.shape[0],target.shape[2]):
        raise ValueError('Mismatched marker predictions, masks or surface IDs')
    valid=valid.bool() & torch.isfinite(target).all(-1)
    safe=torch.where(valid[...,None],target,torch.zeros_like(target)).detach().float()
    scale=torch.as_tensor(scale,device=prediction.device,dtype=torch.float32).clamp_min(1e-6)
    error=F.smooth_l1_loss(prediction.float()/scale,safe/scale,reduction='none',beta=.1).mean(-1)
    losses=[]
    for sid in surface.unique():
        mask=valid & (surface[:,None,:]==sid)
        count=mask.sum(-1);supported=count>0
        values=torch.where(mask,error,0.).sum(-1)/count.clamp_min(1)
        if supported.any():losses.append(values[supported].mean())
    return torch.stack(losses).mean() if losses else prediction.sum()*0.


class MarkerJointModel(nn.Module):
    """Wrap a NEW joint model; old checkpoints retain their old architecture.

    The base model must already exclude the four replaced AE adapters. Existing
    visual encoders (including the proposed attention encoder) expose encode().
    """
    def __init__(self,base_model,latent_dim=32,hidden=128):
        super().__init__()
        if set(MARKER_SOURCES)&set(base_model.adapters):
            raise ValueError('Remove replaced RGB/code adapters from new model configuration')
        self.base=base_model
        self.marker_heads=nn.ModuleDict({s:MarkerDisplacementHead(latent_dim,hidden) for s in MARKER_SOURCES})

    def forward(self,features,view_mask,source,marker_reference=None,marker_surface=None,sample_latent=None):
        if source not in self.marker_heads:
            return self.base(features,view_mask,source,sample_latent=sample_latent)
        if marker_reference is None or marker_surface is None:
            raise ValueError('Marker prediction requires static reference geometry')
        mu,logvar=self.base.visual.encode(features,view_mask)
        sample=self.base.stochastic and self.training if sample_latent is None else sample_latent
        sample=sample and self.base.stochastic
        z=mu+torch.randn_like(mu)*(.5*logvar).exp() if sample else mu
        return dict(marker_displacement=self.marker_heads[source](z,marker_reference,marker_surface),
                    mu=mu,logvar=logvar,latent_mode=self.base.latent_mode)


def mixed_forward(model,features,batch,sample_latent=None):
    source=batch['source'][0]
    if len(set(batch['source']))!=1:
        raise ValueError('Expected single-source batch')
    return model(features,batch['view_mask'],source,
                 marker_reference=batch.get('marker_reference'),marker_surface=batch.get('marker_surface'),
                 sample_latent=sample_latent)


def mixed_loss(output,batch,beta=0.):
    if 'marker_displacement' not in output:
        from .train_alignment import alignment_loss
        return alignment_loss(output,batch,beta)
    task=marker_loss(output['marker_displacement'],batch['marker_target'],batch['marker_valid'],batch['marker_surface'])
    mu,logvar=output['mu'].float(),output['logvar'].float()
    if output['latent_mode']=='deterministic':
        if beta!=0:raise ValueError('Deterministic latent cannot use KL')
        kl=mu.sum()*0
    else:
        valid=(batch['marker_valid'] & torch.isfinite(batch['marker_target']).all(-1)).any(-1)
        values=.5*(mu.square()+logvar.exp()-1-logvar).mean(-1)
        kl=values[valid].mean() if valid.any() else mu.sum()*0
    return task+beta*kl,task.detach(),kl.detach()


from .vjepa_spatial import JointSpatialVJEPA

MODEL_TYPE='joint_vjepa21_spatial_markers_v1'


class MarkerSpatialVJEPA(JointSpatialVJEPA):
    """Full spatial attention, retaining five AE heads and two numeric heads.

    Four optical heads predict signed vectors queried at reference dot locations.
    No optical RGB encoder/decoder participates for these four sources.
    """
    def __init__(self,marker_scales,marker_hidden=128,**kwargs):
        if set(kwargs['sensor_code_dims']) & set(MARKER_SOURCES):
            raise ValueError('Marker sources must not retain RGB AE adapters')
        if set(marker_scales)!=set(MARKER_SOURCES) or any(not 0<float(v)<float('inf') for v in marker_scales.values()):
            raise ValueError('Expected positive train-only marker normalization for every replaced source')
        super().__init__(**kwargs)
        self.marker_heads=nn.ModuleDict({s:MarkerDisplacementHead(kwargs.get('latent_dim',32),marker_hidden) for s in MARKER_SOURCES})
        self.marker_scales=nn.ParameterDict({s:nn.Parameter(torch.tensor(float(v)),requires_grad=False) for s,v in marker_scales.items()})
        self.model_type=MODEL_TYPE

    def forward(self,features,view_mask,source,sample_latent=None,marker_reference=None,marker_surface=None):
        if source not in self.marker_heads:
            return super().forward(features,view_mask,source,sample_latent=sample_latent)
        if marker_reference is None or marker_surface is None:
            raise ValueError('Marker prediction needs episode-local static reference geometry')
        roles=torch.arange(3,device=features.device)[None].expand(len(features),-1).clone()
        if source=='freetacman':roles[:,1]=0
        mu,logvar=self.visual.encode(features,view_mask,roles)
        return dict(mu=mu,logvar=logvar,latent_mode='deterministic',
                    marker_displacement=self.marker_heads[source](mu,marker_reference,marker_surface))


def spatial_marker_loss(model,output,batch):
    source=batch['source'][0]
    if source not in model.marker_heads:
        from .train_alignment import alignment_loss
        return alignment_loss(output,batch,0.)
    task=marker_loss(output['marker_displacement'],batch['marker_target'],batch['marker_valid'],
                     batch['marker_surface'],model.marker_scales[source])
    return task,task.detach(),task.detach()*0


def model_from_state(state):
    if state.get('model_type')!=MODEL_TYPE:raise ValueError('Not a marker V-JEPA checkpoint')
    model=MarkerSpatialVJEPA(**state['model_config'])
    model.load_state_dict(state['model'],strict=True)
    return model
