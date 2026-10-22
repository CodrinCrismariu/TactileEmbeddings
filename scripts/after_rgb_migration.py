"""Finish native-data checks and cache preparation when background conversion ends."""
import json
from pathlib import Path
import subprocess
import time

root=Path('processing/storage_v1')
while True:
    status=json.loads((root/'status.json').read_text())
    if status['stage']=='failed':raise SystemExit('RGB conversion failed; postprocessing was not run')
    if status['stage']=='complete':break
    time.sleep(20)
with (root/'finalization.log').open('a') as log:
    result=subprocess.run(['.venv/bin/python','scripts/finalize_rgb_migration.py'],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise SystemExit(result.returncode)
