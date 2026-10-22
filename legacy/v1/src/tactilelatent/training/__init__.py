"""Optimizers, objective functions, teacher preparation, and current and historical experiment trainers.

Modules load lazily. Importing this namespace does not initialize a model,
contact a dataset host, start training, or run a viewer.
"""
from importlib import import_module as _import_module

__all__ = ('acquisition_training', 'alternating_adversary', 'bilateral_mirror_preparation', 'continuation_objectives', 'expansion_runtime', 'expansion_training', 'fgw_mixer', 'fgw_objective', 'fgw_pilot', 'fgw_study', 'gel_mirror_preparation', 'losses', 'meta_learning', 'mirror_balanced_training', 'mirror_preparation', 'pressure_teacher', 'pretrain_egotouch_autoencoder', 'pretrain_sensor_autoencoder', 'relational', 'sam', 'tactile_teacher', 'train', 'train_alignment', 'train_continuation', 'train_invariance', 'train_joint', 'train_joint_adversarial', 'train_joint_alternating', 'train_marker_autoencoders', 'train_vjepa', 'train_vjepa_markers', 'train_vjepa_spatial', 'trajectory_matching', 'vjepa_cached_training')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
