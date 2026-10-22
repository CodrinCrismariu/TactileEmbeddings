"""Run verified source conversions sequentially so reclaimed space arrives early."""
import json
import os
from pathlib import Path
import subprocess
import time

from tactilelatent.aligned_frames import write_json

ROOT=Path('processing/storage_v1')
SOURCES=['ftp_motiontrans','ftp_rdp','ftp_rdp_bimanual','ftp_unit','ftp_unit_bimanual',
         'ftp_vitamin','opentouch','egotactile','opengraph','egotouch','ftp_vla_touch']


def usage():
    values=subprocess.check_output(['du','-sb','datasets','processing'],text=True)
    return sum(int(row.split()[0]) for row in values.splitlines())


def main():
    started=time.time();complete=[]
    for source in SOURCES:
        status=dict(stage='running',source=source,completed_sources=complete,pid=os.getpid(),started_at=started,
                    project_bytes=usage(),codec='H264 CRF18 384x384 camera inputs only; original frame counts')
        write_json(ROOT/'status.json',status)
        with (ROOT/f'{source}.log').open('a') as log:
            result=subprocess.run(['.venv/bin/python','scripts/migrate_jepa_rgb.py','--workers','4','--sources',source],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(ROOT/'status.json',dict(**{**status,'stage':'failed'},returncode=result.returncode,log=str(ROOT/f'{source}.log')))
            raise SystemExit(result.returncode)
        complete.append(source)
    write_json(ROOT/'status.json',dict(stage='complete',completed_sources=complete,project_bytes=usage(),elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
