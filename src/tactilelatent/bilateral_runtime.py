"""Load additive bilateral checkpoints while retaining historical factories."""
from .bilateral_autoencoder import BilateralVectorAutoencoder,load_bilateral_teacher


def install_viewer_loaders():
    from . import sensor_autoencoder,autoencoder_viewer,mirror_viewer,marker_autoencoder
    original=sensor_autoencoder.make_autoencoder
    marker_original=marker_autoencoder.load_marker_autoencoder
    def make(config):
        return BilateralVectorAutoencoder(**config) if config.get('bilateral') else original(config)
    def load_marker(path,device='cpu'):
        import torch
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        return load_bilateral_teacher(path,device) if state.get('model_type')=='bilateral_marker_autoencoder_v1' else marker_original(path,device)
    sensor_autoencoder.make_autoencoder=make;autoencoder_viewer.make_autoencoder=make
    mirror_viewer.make_autoencoder=make;marker_autoencoder.load_marker_autoencoder=load_marker
