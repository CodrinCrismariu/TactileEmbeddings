"""Sparse pressure reconstruction with a single continuous sigmoid output.

The dataset's native 224x224 values and train-fitted normalization are unchanged.
Positive pressure and measured zero get separate per-frame loss normalization.
Missing pixels never enter either group. Logit BCE uses continuous pressure
targets, not a second contact-classification head.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sensor_autoencoder import SensorAutoencoder


class PressureAutoencoder(SensorAutoencoder):
    """Same 64D convolutional AE with fixed train-only input conditioning.

    Decoder targets and outputs retain the existing bounded pressure scale.
    Conditioning makes sparse pressure visible beside the constant valid mask.
    """
    def __init__(self,input_mean=0.,input_scale=1.,**kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(input_mean) or not math.isfinite(input_scale) or input_scale<=0:
            raise ValueError('Input statistics must be finite with positive scale')
        self.input_mean=float(input_mean);self.input_scale=float(input_scale)
        self.config.update(input_mean=self.input_mean,input_scale=self.input_scale)

    def encode(self,target,valid=None):
        return super().encode((target-self.input_mean)/self.input_scale,valid)


def pressure_logits(model,target,valid):
    if not isinstance(model.decoder[-1],nn.Sigmoid):
        raise ValueError('Pressure teacher requires its historical sigmoid decoder')
    code=model.encode(target,valid)
    return model.decoder[:-1](code)


def balanced_frame_mean(value,target,valid):
    """Equal group weight when both groups exist; keep empty-contact frames."""
    axes=tuple(range(1,value.ndim))
    positive=valid & (target>0)
    zero=valid & (target==0)
    total=torch.zeros(value.shape[0],device=value.device,dtype=value.dtype)
    groups=torch.zeros_like(total)
    for mask in (positive,zero):
        count=mask.sum(axes)
        average=torch.where(mask,value,0.).sum(axes)/count.clamp_min(1)
        total=total+average
        groups=groups+(count>0).to(value.dtype)
    supported=groups>0
    if not supported.any():
        return value.sum()*0.
    return (total[supported]/groups[supported]).mean()


def reconstruction_loss(logits,target,valid):
    valid=valid.bool() & torch.isfinite(target)
    clean=torch.where(valid,target,0.)
    if ((clean[valid]<0)|(clean[valid]>1)).any():
        raise ValueError('Pressure targets must use the existing [0,1] normalization')
    prediction=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,clean,reduction='none')
    magnitude=(prediction-clean).abs()+.5*(prediction-clean).square()
    return (balanced_frame_mean(bce,clean,valid)
            +balanced_frame_mean(magnitude,clean,valid)
            +.5*bce[valid].mean() if valid.any() else logits.sum()*0.)


def initialize_pressure_bias(model,probability=.02):
    if not 0<probability<1:raise ValueError('Initial probability must be inside (0,1)')
    layer=model.decoder[-2]
    if not isinstance(layer,nn.Conv2d):raise ValueError('Expected native convolutional decoder')
    with torch.no_grad():layer.bias.fill_(math.log(probability/(1-probability)))


def choose_training_indices(valid_indices,positive_indices,budget,rng,positive_fraction=.75):
    """Balance measured states without resizing or fabricating target values."""
    valid=np.asarray(valid_indices,np.int64);positive=np.asarray(positive_indices,np.int64)
    zero=np.setdiff1d(valid,positive,assume_unique=True)
    budget=min(int(budget),len(valid))
    n_positive=min(len(positive),int(math.ceil(budget*positive_fraction)))
    n_zero=min(len(zero),budget-n_positive)
    n_positive=min(len(positive),budget-n_zero)
    chosen=np.r_[rng.choice(positive,n_positive,replace=False),rng.choice(zero,n_zero,replace=False)]
    rng.shuffle(chosen)
    return chosen


def metric_sums(prediction,target,valid):
    valid=valid.bool()&torch.isfinite(target)
    clean=torch.where(valid,target,0.)
    active=valid&(clean>0);background=valid&(clean==0)
    error=(prediction-clean).abs()
    result={}
    for name,mask in [('all',valid),('contact',active),('background',background)]:
        result[name+'_absolute_sum']=float(error[mask].sum().item())
        result[name+'_zero_absolute_sum']=float(clean.abs()[mask].sum().item())
        result[name+'_count']=int(mask.sum().item())
    result.update(frames=len(target),contact_frames=int(active.flatten(1).any(1).sum()),
        predicted_positive_pressure_sum=float(prediction[active].sum().item()),
        measured_positive_pressure_sum=float(clean[active].sum().item()),
        predicted_background_pressure_sum=float(prediction[background].sum().item()))
    return result


def summarize_metrics(totals):
    out={}
    for name in ['all','contact','background']:
        count=totals.get(name+'_count',0)
        out[name+'_mae']=totals.get(name+'_absolute_sum',0)/max(count,1)
        out[name+'_zero_baseline_mae']=totals.get(name+'_zero_absolute_sum',0)/max(count,1)
        out[name+'_count']=count
    out['balanced_mae']=.5*(out['contact_mae']+out['background_mae'])
    out['balanced_zero_baseline_mae']=.5*out['contact_zero_baseline_mae']
    out['contact_to_zero_ratio']=out['contact_mae']/max(out['contact_zero_baseline_mae'],1e-12)
    out['predicted_contact_pressure_ratio']=totals.get('predicted_positive_pressure_sum',0)/max(totals.get('measured_positive_pressure_sum',0),1e-12)
    out['predicted_background_to_contact_mass']=totals.get('predicted_background_pressure_sum',0)/max(totals.get('measured_positive_pressure_sum',0),1e-12)
    out['frames']=totals.get('frames',0)
    out['contact_frames']=totals.get('contact_frames',0)
    return out
