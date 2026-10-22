"""Compatibility entrypoint for the archived audit_saved_visual_video.py command.

Execute the unchanged historical body with this module's original filename,
arguments, globals and import search path. Keep the old command for reproducibility.
"""
from pathlib import Path as _ArchivePath

exec(
    compile(_ArchivePath(__file__).parent.joinpath("archive", "audit_saved_visual_video.py").read_bytes(), __file__, "exec"),
    globals(),
    globals(),
)
