"""Visual encoders, attention readers, task adapters, and native sensor autoencoders.

Modules load lazily. Importing this namespace does not initialize a model,
contact a dataset host, start training, or run a viewer.
"""
from importlib import import_module as _import_module

__all__ = ('alignment_model', 'baihu_autoencoder', 'bilateral_autoencoder', 'expansion_models', 'joint_model', 'marker_autoencoder', 'marker_code_model', 'marker_model', 'model', 'numeric_autoencoder', 'sensor_autoencoder', 'tactile_autoencoder', 'vjepa_model', 'vjepa_spatial')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
