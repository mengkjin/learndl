"""Bounded, restartable task log publication over the Syncthing share."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

CHUNK_BYTES = 256 * 1024
ROUND_BYTES = 4 * 1024 * 1024
RETENTION = 24 * 3600


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def task_key(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()


def settings():
    from src.automation.artifact_exchange.core import Config, sandboxed
    config = Config.load()
    from src.proj import PATH
    installed = PATH.runtime / 'scheduling' / 'installed.json'
    jobs = json.loads(installed.read_text()).get('watchdog', {}).get('jobs', {}) if installed.exists() else {}
    return config, (config.enabled and config.role == 'publisher' and not sandboxed()
                    and jobs.get('remote_monitor', {}).get('enabled', False))


def retain_log(path: Path, task_id: str) -> bool:
    """Move a closed capture before its normal deletion, without copying its bytes."""
    config, enabled = settings()
    if not enabled:
        return False
    destination = config.state / 'remote_monitor' / 'pending' / (task_key(task_id) + '.md')
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination.with_suffix('.json'), {'task_id': task_id})
    path.replace(destination)
    return True


from .remote_transport import Publisher


def collect(repository, *, now: float, pending_ids=(), since=None) -> list[dict]:
    """Read scripts changed within the retention window, plus all active scripts."""
    tasks = []
    since = now - RETENTION if since is None else since
    if repository.db_path.exists():
        with repository._connect() as db:
            records = db.execute("SELECT * FROM task_records WHERE status IN ('starting','running') OR end_time >= ?", (since,)).fetchall()
            found = {row['task_id'] for row in records}
            for task_id in pending_ids:
                if task_id not in found:
                    row = db.execute('SELECT * FROM task_records WHERE task_id=?', (task_id,)).fetchone()
                    if row is not None:
                        records.append(row)
        from dataclasses import replace
        captures = {path.name.partition('.')[2].removesuffix('.md'): path
                    for path in (repository.runtime_dir / 'crash_protector').glob('*.md')}
        for row in records:
            task = repository._snapshot(row)
            log = captures.get(task.task_id.replace('/', '_'))
            if log is None:
                with repository._connect() as db:
                    output = db.execute('SELECT file_path FROM task_exit_files WHERE task_id=? ORDER BY id LIMIT 1', (task.task_id,)).fetchone()
                log = Path(output[0]) if output and Path(output[0]).suffix.lower() in {'.md', '.txt', '.log', '.html'} else None
            tasks.append(replace(task, crash_log=log))
    # Managed adapter applies the same deduplication rules as the local monitor.
    tasks.extend(repository._managed_tasks(detail=True, since=since))
    found = {task.task_id for task in tasks}
    for task_id in pending_ids:
        if task_id.startswith('managed:') and task_id not in found:
            tasks.extend(repository._managed_tasks(detail=True, task_id=task_id, since=0))
    result = []
    for task in tasks:
        log = task.crash_log or (task.exit_files[0] if task.exit_files else None)
        result.append(dict(task_id=task.task_id, script=task.script, cmd=task.cmd,
                           status=task.status, pid_alive=task.pid_alive, start_time=task.effective_start,
                           end_time=task.end_time, log=str(log) if log else None,
                           phase=task.managed_phase))
    return result


def publish_once() -> int:
    import portalocker
    from .core import TaskMonitorRepository
    config, enabled = settings()
    if not enabled:
        return 0
    private = config.state / 'remote_monitor'
    private.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(private / 'publish.lock'), timeout=0):
            publisher = Publisher(config.shared_root, config.state, config.source_host)
            state_path = private / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {'tasks': {}}
            finalized = {identity for identity, record in state['tasks'].items() if record.get('expires_at')}
            pending_ids = [json.loads(path.read_text())['task_id'] for path in (private / 'pending').glob('*.json')
                           if path.stem not in finalized]
            pending_ids.extend(record['task']['task_id'] for identity, record in state['tasks'].items()
                               if identity not in finalized)
            repository = TaskMonitorRepository()
            heartbeat = repository.runtime_dir / 'scheduling/auxiliary/remote_monitor.json'
            health = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
            health.update(status='running', last_attempt=time.time())
            atomic_json(heartbeat, health)
            try:
                sampled_at = time.time()
                tasks = collect(repository, now=sampled_at, pending_ids=set(pending_ids),
                                since=state.get('collected_at', sampled_at - RETENTION) - 60)
                limits = {}
                deadline = time.monotonic() + 20
                packet = publisher.tick(tasks, limits=limits)
                while any(task['offset'] < task['size'] for task in packet['tasks']):
                    from src.automation.scheduling.autofix_store import state_dir
                    from src.automation.scheduling.worker_bootstrap import wait_for_code
                    if time.monotonic() >= deadline or wait_for_code(state_dir(), 'remote_monitor'):
                        break
                    offsets = [(task['key'], task['generation'], task['offset']) for task in packet['tasks']]
                    packet = publisher.tick(tasks, limits=limits)
                    if offsets == [(task['key'], task['generation'], task['offset']) for task in packet['tasks']]:
                        break
            except Exception as exc:
                health.update(status='error', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
                atomic_json(heartbeat, health)
                raise
            health.update(status='complete', error=None, last_success_at=packet['sampled_at'], finished_at=time.time())
            atomic_json(heartbeat, health)
    except portalocker.LockException:
        return 0
    return 0


def read_volume(root: Path, task: dict, number: int) -> str | None:
    """Read exactly one advertised volume, tolerating out-of-order delivery."""
    if number < 0 or number * CHUNK_BYTES >= task['offset']:
        return None
    if 'segments' in task:
        from .remote_reader import RemoteCache
        return RemoteCache().text(root, task, number * CHUNK_BYTES,
                                  min((number + 1) * CHUNK_BYTES, task['offset']))
    path = root / 'tasks' / task_key(task['task_id']) / f"{int(task['generation'])}-{number:08d}.bin"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    expected = min(CHUNK_BYTES, task['offset'] - number * CHUNK_BYTES)
    if len(data) < expected:
        return None
    # Decode with adjacent bytes so UTF-8 spanning a volume boundary remains intact.
    prefix = b''
    if number:
        previous = path.with_name(f"{int(task['generation'])}-{number - 1:08d}.bin")
        try:
            prefix = previous.read_bytes()[-3:]
        except FileNotFoundError:
            pass
    import codecs
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    if prefix:
        decoder.decode(prefix, final=False)
    return decoder.decode(data[:expected], final=False)
