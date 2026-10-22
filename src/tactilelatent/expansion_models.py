"""New four-surface teachers, without changing historical two-surface models."""
import torch
from torch import nn

from .marker_autoencoder import MarkerAutoencoder
from .marker_model import MarkerDisplacementHead
from .sensor_autoencoder import SensorAutoencoder


class ExpansionMarkerAutoencoder(MarkerAutoencoder):
    def __init__(self, surfaces=4, **kwargs):
        super().__init__(**kwargs)
        if surfaces != 4: raise ValueError('Baihu requires exactly four ordered surfaces')
        self.surface = nn.Embedding(surfaces,8)
        self.decoder = MarkerDisplacementHead(self.config['latent_dim'],self.config['width'],surfaces)
        self.config['surfaces'] = surfaces


def load_teacher(path, device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['model_type']=='expansion_marker_autoencoder_v1':
        model=ExpansionMarkerAutoencoder(**state['model_config'])
    elif state['model_type']=='expansion_marker_autoencoder_v2':
        from .baihu_autoencoder import BaihuMarkerAutoencoder
        model=BaihuMarkerAutoencoder(**state['model_config'])
    elif state['model_type']=='expansion_pressure_autoencoder_v1':
        model=SensorAutoencoder(**state['model_config'])
    elif state['model_type']=='expansion_pressure_autoencoder_v2':
        from .pressure_teacher import PressureAutoencoder
        model=PressureAutoencoder(**state['model_config'])
    else:raise ValueError('Not a data-expansion teacher')
    model.load_state_dict(state['model'],strict=True)
    return model.to(device).eval().requires_grad_(False),state
