"""Source I/O adapters. Their outputs share contracts, not sensor dimensions."""

from .definitions import SourceDefinition, register_source_definition, source_definition

__all__ = ["SourceDefinition", "source_definition", "register_source_definition"]
