"""Use an existing kernel-matched NVIDIA runtime, scoped to this process."""
import os
from pathlib import Path
import re
import sys


def configure():
    version=Path('/proc/driver/nvidia/version')
    if not version.exists():return
    match=re.search(r'Module.*?\s(\d+\.\d+\.\d+)',version.read_text())
    if not match:return
    project=Path(__file__).resolve().parents[2]
    runtime=project/f'processing/runtime/nvidia-{match.group(1)}/root/usr/lib/x86_64-linux-gnu'
    if runtime.is_dir() and str(runtime) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.environ['LD_LIBRARY_PATH']=str(runtime)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        os.execv(sys.executable,[sys.executable,*sys.argv])
