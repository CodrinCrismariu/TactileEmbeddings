"""Persistent acquisition queue. Only scene RGB is resized or compressed.

Each job writes an independent manifest; existing joint manifests/checkpoints are
never rewritten. A storage pause is resumable and never triggers deletion.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from .aligned_frames import file_hash, json_hash, write_json

ROOT = Path('processing/data_queue_v2')
CONFIG = Path('configs/data_queue_v2.json')
RESEARCH = Path('processing/research/storage_after_expansion_v1')


class StoragePause(RuntimeError):
    def __init__(self, message, *, scope='source'):
        super().__init__(message)
        self.scope = scope


class QualityBlock(RuntimeError):
    """Source completed its audit but exclusions require a schema/data fix."""
    pass


class CompletedWithExclusions(RuntimeError):
    """All candidates audited; unavailable upstream observations are documented."""
    pass


def retryable(error):
    import httpx
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in (408, 429) or error.response.status_code >= 500
    return isinstance(error,(httpx.TransportError,TimeoutError,ConnectionError))


def allocated(paths):
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return 0
    result = subprocess.run(['du', '-s', '-B1', *paths], capture_output=True, text=True, check=True)
    return sum(int(line.split('\t', 1)[0]) for line in result.stdout.splitlines())


class Job:
    def __init__(self, spec, config=None):
        self.config = config if config is not None else json.loads(CONFIG.read_text())
        self.spec = spec
        self.id = spec['id']
        self.root = ROOT / self.id
        self.raw = Path('datasets/acquisition_v2') / self.id
        self.staging = self.raw / 'staging'
        for p in (self.root, self.raw, self.staging):
            p.mkdir(parents=True, exist_ok=True)
        self._usage = (0, 0, 0)
        self._lock = threading.RLock()
        self.state = dict(job=self.id, stage='queued', completed_units=0, total_units=None)
        state = self.root / 'status.json'
        if state.exists():
            self.state.update(json.loads(state.read_text()))

    def report(self, **details):
        with self._lock:
            if 'stage' in details and details['stage'] not in (
                    'download_retry', 'failed', 'paused_storage', 'blocked_quality'):
                details.setdefault('error', None)
            self.state.update(details, updated_at=time.time(), pid=os.getpid())
            write_json(self.root / 'status.json', self.state)
            with (self.root / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(self.state, allow_nan=False) + '\n')

    def check_space(self, additional=0, staging=False, force=False):
        # Include all datasets/processing, including the independently running
        # EgoPressure/Baihu jobs. Cache du briefly; check free space every call.
        now = time.monotonic()
        if force or now - self._usage[0] > 45 or not self._usage[0]:
            project = allocated([Path('datasets'), Path('processing')])
            own = allocated([self.raw, self.root])
            temporary = allocated([self.staging])
            self._usage = (now, project, own - temporary)
            self.report(project_bytes=project, retained_bytes=own-temporary,
                        staging_bytes=temporary, free_bytes=shutil.disk_usage(ROOT).free)
        _, project, retained = self._usage
        free = shutil.disk_usage(ROOT).free
        if free - additional < self.config['physical_reserve_bytes']:
            raise StoragePause('100 GiB physical free-space reserve would be crossed', scope='global')
        if project + additional > self.config['project_cap_bytes'] - self.config['experiment_reserve_bytes']:
            raise StoragePause('700 GB project cap / 30 GB experiment reserve would be crossed', scope='global')
        if staging:
            if allocated([self.staging]) + additional > self.config['staging_limit_bytes']:
                raise StoragePause('65 GB staging limit would be crossed')
        elif retained + additional > self.spec['budget_bytes']:
            raise StoragePause(f'{self.id} retained-storage allocation would be crossed')

    def inventory(self):
        name = {'trex': 'zekaiwang__trex_dataset.json', 'feel': 'edessa__feel.json',
                'tacquad': 'xxuan01__TacQuad.json', 'hrdexdb': 'HRDexDB_metadata.json'}[self.id]
        d = json.loads((RESEARCH/name).read_text())
        if d['revision'] != self.spec['revision']:
            raise ValueError('Release inventory revision mismatch')
        return d

    def download(self, filename, staging=False):
        """Pinned public HF file, bounded writes, Range resume, SHA256 receipt.

        No second full-size Hub cache. Upstream LFS SHA256 is checked when
        available; Git blobs get a local SHA256 and pinned revision/etag.
        """
        import httpx
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        if Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('Unsafe repository filename')
        dest = (self.staging if staging else self.raw)/filename
        receipt = self.root/'downloads'/(json_hash(filename)+'.json')
        if receipt.exists() and dest.exists():
            d = json.loads(receipt.read_text())
            if d['revision'] != self.spec['revision'] or d['sha256'] != file_hash(dest):
                raise ValueError(f'Changed downloaded artifact: {filename}')
            return dest
        url = hf_hub_url(self.spec['repo'], filename, revision=self.spec['revision'], repo_type='dataset')
        # DNS/redirect failures used to escape before the streaming retry loop.
        for attempt in range(5):
            try:
                meta = get_hf_file_metadata(url)
                break
            except Exception as error:
                if not retryable(error) or attempt == 4:
                    raise
                self.report(stage='download_retry', file=filename, attempt=attempt+1,
                            error=f'metadata: {error}')
                time.sleep(min(2**attempt, 20))
        size = meta.size
        if size is None:
            raise ValueError('Pinned download lacks a byte length')
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name+'.partial')
        # Reuse previously inventoried research samples without duplicating
        # allocated storage. Validate against upstream hash before accepting.
        cached = RESEARCH / {'trex':'trex_metadata','hrdexdb':'hrdex_metadata'}.get(self.id, '_none') / filename
        if not dest.exists() and not part.exists() and cached.exists() and cached.stat().st_size == size:
            digest = file_hash(cached)
            if len(meta.etag or '') != 64 or digest == meta.etag:
                os.link(cached, dest)
        if dest.exists():
            digest = file_hash(dest)
            if dest.stat().st_size != size or (len(meta.etag or '') == 64 and digest != meta.etag):
                raise ValueError(f'Unverified existing download: {filename}')
        else:
            last = None
            for attempt in range(5):
                offset = part.stat().st_size if part.exists() else 0
                if offset > size:
                    raise ValueError('Resume file is larger than release')
                self.check_space(size-offset, staging=staging, force=True)
                self.report(stage='downloading', file=filename, file_bytes_done=offset, file_bytes_total=size)
                try:
                    if offset < size:
                        headers = {'Accept-Encoding':'identity'}
                        if offset:
                            headers['Range'] = f'bytes={offset}-'
                        with httpx.stream('GET', url, headers=headers, follow_redirects=True,
                                          timeout=httpx.Timeout(120, connect=30)) as response:
                            response.raise_for_status()
                            if offset and response.status_code != 206:
                                # Server ignored Range: overwrite rather than append.
                                offset = 0
                            if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                                raise ValueError('Resume Content-Range differs')
                            started = time.monotonic(); tick = started; initial = offset
                            with part.open('ab' if offset else 'wb') as f:
                                for chunk in response.iter_bytes(4*2**20):
                                    if offset+len(chunk) > size:
                                        raise ValueError('Server exceeded pinned download size')
                                    self.check_space(len(chunk), staging=staging)
                                    f.write(chunk); offset += len(chunk)
                                    if time.monotonic()-tick > 10:
                                        speed = (offset-initial)/max(time.monotonic()-started, .001)
                                        self.report(stage='downloading', file_bytes_done=offset, download_bytes_per_second=speed,
                                                    file_eta_seconds=(size-offset)/speed if speed else None)
                                        tick = time.monotonic()
                                f.flush(); os.fsync(f.fileno())
                    if part.stat().st_size != size:
                        raise ValueError('Incomplete pinned download')
                    digest = file_hash(part)
                    if len(meta.etag or '') == 64 and digest != meta.etag:
                        # Never append to a corrupt complete file on retry.
                        part.rename(part.with_name(part.name+'.corrupt-'+str(int(time.time()))))
                        raise ValueError('SHA256 differs from pinned LFS checksum')
                    part.replace(dest)
                    break
                except StoragePause:
                    raise
                except Exception as error:
                    last = error
                    self.report(stage='download_retry', file=filename, attempt=attempt+1, error=str(error))
                    time.sleep(min(2**attempt, 20))
            else:
                raise last
        receipt.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt, dict(repo=self.spec['repo'], revision=self.spec['revision'],
            path=str(dest), file=filename, bytes=size, sha256=digest, upstream_etag=meta.etag))
        self.report(stage='downloaded', file=filename, file_bytes_done=size, file_bytes_total=size)
        return dest

    def retire_rgb(self, path, replacement, receipt, *, expected_sha256=None):
        from .data_expansion import safe_delete
        if not Path(replacement).is_file() or file_hash(replacement) != receipt['sha256']:
            raise ValueError('Replacement failed verification; retain staging')
        path = Path(path)
        digest = expected_sha256 or file_hash(path)
        safe_delete([path], self.staging, self.root/'deletions.jsonl',
                    'verified scene RGB replacement '+str(replacement), expected={str(path):digest})
        self._usage = (0,0,0)

    def recover_rgb_cleanup(self, filename, replacement, receipt, *, expected_frames):
        """Recover interruption between publishing a verified RGB and retirement.

        Call only after the source's native target/frame mapping has passed.
        No incomplete downloads, tactile payloads, or arbitrary files are removed.
        """
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe staged RGB filename')
        path = self.staging / relative
        if not path.exists():
            return False
        if receipt['frames'] != expected_frames or Path(receipt['path']) != Path(replacement):
            raise ValueError('Resumed RGB frame count/path differs; retain original')
        download_receipt = self.root/'downloads'/(json_hash(filename)+'.json')
        member_receipt = self.root/'member_receipts'/(json_hash(filename)+'.json')
        evidence = download_receipt if download_receipt.exists() else member_receipt
        if not evidence.exists():
            raise ValueError('Staged original lacks a pinned receipt; retain original')
        saved = json.loads(evidence.read_text())
        if download_receipt.exists() and saved['revision'] != self.spec['revision']:
            raise ValueError('Staged original release changed; retain original')
        if evidence == member_receipt and (saved['archive_id'] != self.spec['archive_id'] or
                saved['archive_index_sha256'] != self.spec['archive_index_sha256']):
            raise ValueError('Staged archive member release changed; retain original')
        if path.stat().st_size != saved['bytes']:
            raise ValueError('Staged original size changed; retain original')
        self.report(stage='verifying_cleanup', file=str(path))
        # safe_delete computes the original checksum immediately before unlink.
        self.retire_rgb(path, replacement, receipt, expected_sha256=saved['sha256'])
        self.report(stage='recovered_cleanup', file=str(path), retired_bytes=saved['bytes'])
        return True


def snapshot(config=None):
    config = config or json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in config['jobs']:
        p = ROOT/spec['id']/'status.json'
        row = json.loads(p.read_text()) if p.exists() else dict(job=spec['id'],stage='queued')
        row.update(budget_bytes=spec['budget_bytes'], scope=spec['scope'])
        rows.append(row)
    result = dict(updated_at=time.time(), jobs=rows, project_cap_bytes=config['project_cap_bytes'],
        acquisition_budget_bytes=sum(s['budget_bytes'] for s in config['jobs']),
        free_bytes=shutil.disk_usage(ROOT).free, joint_training_started=False)
    write_json(ROOT/'status.json', result)
    lines = ['# Download and processing queue', '', 'Updated: '+datetime.now(timezone.utc).isoformat(), '',
        '**240 GB acquisition budget · 700 GB project cap · 100 GiB physical reserve · 30 GB experiment reserve**', '',
        '| Source | Scope | Budget GB | Stage | Progress | Current work |', '|---|---|---:|---|---|---|']
    for r in rows:
        done = r.get('completed_units',0); total = r.get('total_units')
        progress = f'{done}/{total}' if total is not None else str(done)
        detail = r.get('file',r.get('episode',r.get('error','')))
        if r.get('stage') in ('paused_storage','failed','blocked_quality'):
            detail = r.get('error',detail)
        if r.get('stage') == 'downloading' and r.get('file_bytes_total'):
            progress += f' · {100*r.get("file_bytes_done",0)/r["file_bytes_total"]:.1f}% of file'
        lines.append(f'| {r["job"]} | {r["scope"]} | {r["budget_bytes"]/1e9:g} | {r["stage"]} | {progress} | {str(detail).replace(chr(124),chr(47))} |')
    lines += ['', 'Each source runs download → native targets → RGB conversion → alignment/quality checks → manifest.',
        'Independent workers retry with backoff; a source-local storage pause does not block other datasets.',
        'Only scene RGB is resized/lossy-compressed. Tactile payloads retain native shape and signed values; missing observations remain masked.',
        'T-Dex and HRDexDB are pilots. TacQuad is an alignment benchmark, excluded from joint visual training.',
        'No new joint training starts. MotionTrans stays held out; FreeTacMan stays excluded.', '',
        'Per-source: `<source>/status.json`, `progress.jsonl`, `worker.log`, `manifest.json`, `quality.json`, `deletions.jsonl`.',
        'Existing acquisition: [EgoPressure / Baihu status](../data_expansion_v1/STATUS.md).', '']
    tmp = ROOT/'STATUS.tmp'; tmp.write_text('\n'.join(lines)); tmp.replace(ROOT/'STATUS.md')
    return result


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run_source(source):
    from .acquisition_sources import PROCESSORS
    config = json.loads(CONFIG.read_text())
    spec = next(s for s in config['jobs'] if s['id']==source)
    job = Job(spec, config)
    with exclusive(job.root/'worker.lock'):
        job.report(stage='starting', error=None)
        implementation=[Path(__file__),Path('src/tactilelatent')/'acquisition_sources.py',
            Path('src/tactilelatent')/'acquisition_archives.py',Path('src/tactilelatent')/'acquisition_pilots.py',
            Path('src/tactilelatent')/'acquisition_data.py',Path('src/tactilelatent')/'acquisition_quality.py',
            Path('src/tactilelatent')/'data_expansion.py',CONFIG]
        write_json(job.root/'implementation.json',dict(files={str(p):file_hash(p) for p in implementation},
            source_specification_sha256=json_hash(spec),started_at=time.time()))
        try:
            PROCESSORS[source](job)
            from .acquisition_data import finalize_numeric
            finalize_numeric(job)
            job.check_space(force=True)
            job.report(stage='complete', file='', error=None)
        except CompletedWithExclusions as error:
            from .acquisition_data import finalize_numeric
            finalize_numeric(job)
            job.report(stage='complete_with_exclusions', file='', error=None, exclusion_note=str(error))
        except QualityBlock as error:
            from .acquisition_data import finalize_numeric
            finalize_numeric(job)
            job.report(stage='blocked_quality', error=str(error))
        except StoragePause as error:
            job.report(stage='paused_storage', error=str(error), storage_scope=error.scope)
            raise
        except Exception as error:
            job.report(stage='failed', error=f'{type(error).__name__}: {error}')
            raise


class QueueSupervisor:
    """Independent source workers: a long run cannot starve another retry."""
    terminal = {'complete', 'complete_with_exclusions', 'blocked_quality'}

    def __init__(self, config):
        self.config = config
        self.running = {}
        self.retry_at = {}
        self.failures = {}
        self.global_retry_at = 0

    def state(self, source):
        path = ROOT/source/'status.json'
        return json.loads(path.read_text()) if path.exists() else {'stage': 'queued'}

    def step(self, now=None):
        import sys
        now = time.time() if now is None else now
        for source, process in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            del self.running[source]
            state = self.state(source)
            if state['stage'] in self.terminal:
                self.failures.pop(source, None)
                self.retry_at.pop(source, None)
                continue
            self.failures[source] = self.failures.get(source, 0) + 1
            delay = min(self.config.get('retry_initial_seconds', 60) *
                        2**min(self.failures[source]-1, 8),
                        self.config.get('retry_max_seconds', 900))
            if state['stage'] == 'paused_storage':
                delay = self.config.get('storage_retry_seconds', 300)
                if state.get('storage_scope') == 'global':
                    self.global_retry_at = max(self.global_retry_at, now+delay)
            self.retry_at[source] = now + delay
        limit = self.config.get('max_parallel_jobs', 5)
        for spec in self.config['jobs']:
            source = spec['id']
            if (len(self.running) >= limit or now < self.global_retry_at):
                break
            if (source in self.running or now < self.retry_at.get(source, 0) or
                    self.state(source)['stage'] in self.terminal):
                continue
            log = ROOT/source/'worker.log'
            with log.open('a') as handle:
                process = subprocess.Popen(
                    [sys.executable, '-u', 'scripts/run_dataset_queue.py', 'source', source],
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            self.running[source] = process
        write_json(ROOT/'supervisor.json', dict(updated_at=now,
            active_sources=list(self.running), retry_at=self.retry_at,
            failure_counts=self.failures, global_retry_at=self.global_retry_at,
            max_parallel_jobs=limit))
        return not self.running and all(self.state(s['id'])['stage'] in self.terminal
                                        for s in self.config['jobs'])


def run_queue():
    config = json.loads(CONFIG.read_text())
    with exclusive(ROOT/'queue.lock'):
        for spec in config['jobs']:
            Job(spec,config)
        supervisor = QueueSupervisor(config)
        try:
            while True:
                done = supervisor.step()
                snapshot(config)
                if done:
                    return
                time.sleep(config.get('poll_seconds', 5))
        finally:
            for process in supervisor.running.values():
                process.terminate()
            for process in supervisor.running.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
