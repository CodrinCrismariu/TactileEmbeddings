"""Continuously summarize the live storage migration in a workspace log."""
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

root=Path('processing/storage_v1');last=None
with Path('STORAGE_MIGRATION.log').open('a',buffering=1) as log:
    while True:
        try:
            overall=json.loads((root/'status.json').read_text())
            unit=json.loads((root/'conversion_status.json').read_text())
            state=(overall.get('stage'),overall.get('source'),unit.get('stage'),unit.get('jobs_done'),unit.get('path'),unit.get('elapsed_seconds'))
            if state!=last:
                timestamp=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
                log.write(json.dumps(dict(time=timestamp,source=overall.get('source'),stage=overall['stage'],
                    completed_sources=overall.get('completed_sources'),unit=unit))+'\n');last=state
            if overall['stage'] in ('complete','failed'):break
        except (OSError,ValueError):pass
        time.sleep(10)
