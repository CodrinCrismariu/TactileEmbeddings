"""Sensor reconstruction, ego-to-tactile predictions, attention, mirror review, and scalar history monitoring.

Modules load lazily. Importing this namespace does not initialize a model,
contact a dataset host, start training, or run a viewer.
"""
from importlib import import_module as _import_module

__all__ = ('acquisition_joint_viewer', 'acquisition_scene_viewer', 'acquisition_viewer', 'attention_viewer', 'autoencoder_viewer', 'bilateral_runtime', 'expansion_render', 'expansion_viewer', 'joint_viewer', 'marker_viewer', 'mirror_mapping_review', 'mirror_viewer', 'opentouch_layout', 'pressure_surface', 'tacquad_viewer', 'training_dashboard', 'viewer_checkpoints', 'viewer_transfer')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
