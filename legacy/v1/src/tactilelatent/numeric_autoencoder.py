"""Small native-vector teachers. Numeric codes are separate from visual latents.

Scale-only asinh normalization preserves native zero and signed wrenches. Missing
coordinates are excluded from losses and supplied as an input mask. Native F1
direction coordinates are retained verbatim; no unverified angle/sentinel rule
or force calibration is inferred from them.
"""
import torch
from torch import nn


MODEL_TYPE = 'native-numeric-ae-v1'


class NumericAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, scale, nonnegative,
                 architecture='mlp', output_mode='native numeric'):
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or latent_dim > input_dim or not hidden_dims:
            raise ValueError('Positive native/code dimensions and hidden widths required')
        scale = torch.as_tensor(scale, dtype=torch.float32)
        nonnegative = torch.as_tensor(nonnegative, dtype=torch.bool)
        if scale.shape != (input_dim,) or nonnegative.shape != scale.shape or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid fixed native scaling or channel constraints')
        self.register_buffer('scale', scale)
        self.register_buffer('nonnegative', nonnegative)
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.config = dict(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=list(hidden_dims),
            scale=scale.tolist(), nonnegative=nonnegative.tolist(), architecture=architecture,
            output_mode=output_mode)
        layers=[];previous=input_dim*2
        for width in hidden_dims:
            if width < 1:raise ValueError('Hidden widths must be positive')
            layers.extend([nn.Linear(previous,width),nn.GELU()]);previous=width
        layers.append(nn.Linear(previous,latent_dim));self.encoder=nn.Sequential(*layers)
        layers=[];previous=latent_dim
        for width in reversed(hidden_dims):
            layers.extend([nn.Linear(previous,width),nn.GELU()]);previous=width
        layers.append(nn.Linear(previous,input_dim));self.decoder=nn.Sequential(*layers)

    def normalize(self, values):
        return torch.asinh(values/self.scale)

    def encode(self, values, valid=None):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError('Expected [frames,native channels]')
        valid = torch.isfinite(values) if valid is None else valid.bool() & torch.isfinite(values)
        if valid.shape != values.shape:raise ValueError('Validity mask shape differs')
        values=torch.where(valid,values,0.)
        return self.encoder(torch.cat([self.normalize(values),valid.to(values.dtype)],dim=-1))

    def decode_normalized(self, code):
        return self.decoder(code)

    def from_normalized(self, values):
        # This bound prevents numerical sinh overflow, not target clipping.
        native=torch.sinh(values.clamp(-15,15))*self.scale
        return torch.where(self.nonnegative,native.clamp_min(0),native)

    def decode(self, code):
        return self.from_normalized(self.decode_normalized(code))

    def forward(self, values, valid=None):
        return self.decode(self.encode(values,valid))


def numeric_loss(prediction, target, valid, nonnegative, nonzero_weight=3.):
    """Loss in fixed transformed coordinates; train even negative raw outputs.

    Physical nonnegativity is enforced on native decoding. Computing this loss
    before that projection avoids dead gradients at zero. No second contact head.
    """
    valid=valid.bool() & torch.isfinite(target)
    target=torch.where(valid,target,0.)
    weights=valid*(1+(nonzero_weight-1)*(nonnegative & (target != 0)))
    error=prediction-target
    loss=torch.nn.functional.smooth_l1_loss(prediction,target,reduction='none',beta=.1)+.25*error.abs()
    counts=weights.sum(0);active=counts>0
    if not active.any():raise ValueError('No observed numeric targets')
    return ((loss*weights).sum(0)/counts.clamp_min(1))[active].mean()


def load_numeric_teacher(path, device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('model_type')!=MODEL_TYPE:raise ValueError('Not a native numeric teacher')
    model=NumericAutoencoder(**state['model_config'])
    model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval(),state
