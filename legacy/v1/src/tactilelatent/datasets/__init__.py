"""Acquisition, chronological clips, native targets, camera roles, mirror transforms, and verified caches.

Modules load lazily. Importing this namespace does not initialize a model,
contact a dataset host, start training, or run a viewer.
"""
from importlib import import_module as _import_module

__all__ = ('acquisition_archives', 'acquisition_data', 'acquisition_joint', 'acquisition_pilots', 'acquisition_quality', 'acquisition_queue', 'acquisition_sources', 'aligned_frames', 'alignment_data', 'baihu_ae_data', 'bimanual_mirror', 'current_frame', 'data', 'data_expansion', 'expand_pilot', 'expansion_baihu', 'expansion_joint', 'expansion_tracker', 'feel_hand_layout', 'fgw_data', 'joint_all_config', 'joint_data', 'marker_ae_data', 'marker_code_cache', 'marker_data', 'marker_storage', 'marker_targets', 'media', 'mirror_cache_verification', 'mirror_joint', 'native_optical', 'prepare_joint', 'prepare_joint_all', 'prepare_pilot', 'prepare_relational', 'pressure_selection', 'sensor_cache', 'sensor_code_cache', 'sensor_data', 'storage', 'tar_zarr', 'vjepa_cache', 'zero_reference')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
