"""Frozen representation probes, transfer, calibration, dataset classifiers, and data audits.

Modules load lazily. Importing this namespace does not initialize a model,
contact a dataset host, start training, or run a viewer.
"""
from importlib import import_module as _import_module

__all__ = ('audit', 'audit_alignment', 'evaluate', 'evaluate_dataset_classifier', 'evaluate_direct_sensor', 'evaluate_direct_vla_touch', 'evaluate_feeling_success', 'evaluate_fgw_calibration', 'evaluate_holdout_decoder', 'evaluate_uncertainty', 'frozen_probe', 'infer', 'invariance_validation', 'paired_probe_cache', 'representation_probe')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
