"""Canonical access to the preserved implementation: :mod:`tactilelatent.pressure_surface`.

The import names share one module object, including private helpers and patched
globals. See docs/compatibility.md before moving provenance-pinned source.
"""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("tactilelatent.pressure_surface")

if __name__ != "__main__":
    _sys.modules[__name__] = _implementation
