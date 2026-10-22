"""Canonical access to the preserved implementation: :mod:`tactilelatent.sensor_cache`.

The import names share one module object, including private helpers and patched
globals. See docs/compatibility.md before moving provenance-pinned source.
"""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("tactilelatent.sensor_cache")

if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
