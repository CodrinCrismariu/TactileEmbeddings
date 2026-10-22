"""Four-surface marker AE with a normalized sensor-code projection.

The historical unnormalized AE remains loadable through its original class.
Only the hidden activation before the 32D sensor code changes here.
"""
from torch import nn

from .expansion_models import ExpansionMarkerAutoencoder


class BaihuMarkerAutoencoder(ExpansionMarkerAutoencoder):
    def __init__(self, code_normalization='pre_gelu_layer_norm', **kwargs):
        if code_normalization != 'pre_gelu_layer_norm':
            raise ValueError('Unsupported Baihu sensor-code normalization')
        super().__init__(**kwargs)
        flatten, hidden, activation, output = self.to_code.children()
        self.to_code = nn.Sequential(flatten, hidden, nn.LayerNorm(self.config['width']),
                                     activation, output)
        self.config['code_normalization'] = code_normalization
