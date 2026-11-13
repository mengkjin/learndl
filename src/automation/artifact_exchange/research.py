"""Explicit research result adapters. Failed delivery never masks research outcomes."""
from __future__ import annotations
import contextvars
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import uuid
from typing import Any
import json
import logging
from pathlib import Path
from .core import ArtifactSpec, Config, publish_artifacts, sandboxed

MODEL_ARTIFACTS: contextvars.ContextVar[list[ArtifactSpec] | None] = contextvars.ContextVar('artifact_model', default=None)
CURRENT_TASK: contextvars.ContextVar[Any] = contextvars.ContextVar('artifact_task', default=None)
ANALYSIS_SUFFIXES = {'.pdf', '.xlsx', '.json', '.csv', '.html', '.md', '.txt', '.log'}
EXCLUDED_DIRS = {'models', 'model', 'checkpoints', 'checkpoint', 'snapshots', 'data', 'dataset', 'prepared', 'mail', 'training_mail', 'bundles', 'tensorboard', 'recovery'}


class TaskDelivery:
    def __init__(self):
        self.pending = []
        self.hooks = []
        self.bundles = {}
        self.mail = []
        self.flushing = False
        self.mailing = False


def task_delivery(owner=None):
    owner = owner if owner is not None else CURRENT_TASK.get()
    if owner is None:
        return None
    if getattr(owner, '_artifact_delivery', None) is None:
        owner._artifact_delivery = TaskDelivery()
    return owner._artifact_delivery


def delivery_log(message, *, failed=False, owner=None):
    owner = owner if owner is not None else CURRENT_TASK.get()
    logger = getattr(owner, 'logger', None)
    if logger is not None:
        (logger.warning if failed else logger.note)('[artifacts] ' + message, vb_level=0)
    else:
        logger = logging.getLogger(__name__)
        (logger.warning if failed else logger.info)('[artifacts] ' + message)


def log_results(records, owner=None):
    for record in records:
        manifest = record.get('manifest', {})
        text = f"{record.get('status', 'unknown')} id={record.get('id', '-')}"
        if manifest:
            text += f" experiment={manifest.get('category')}/{manifest.get('experiment')} run={manifest.get('run_id')}"
        if record.get('route'):
            text += f" destination=artifact_exchange/{record['route']}/{record['filename']}"
        if record.get('error') or record.get('reason'):
            text += ' reason=' + str(record.get('error') or record.get('reason'))
        delivery_log(text, failed=record.get('status') == 'failed', owner=owner)


def publish_safe(task, run_id, artifacts, status='success'):
    owner = CURRENT_TASK.get()
    queue = task_delivery(owner)
    config = None
    try:
        if sandboxed():
            records = [{'status': 'skipped', 'reason': 'autofix'}]
        else:
            config = Config.load()
            records = publish_artifacts(task, run_id, artifacts, status, config=config,
                                        prepare_only=queue is not None and not queue.flushing)
    except Exception as exc:
        records = [{'status': 'failed', 'error': str(exc)}]
    fresh = records
    if owner is not None:
        known = {r['id']: r for r in owner._artifact_results if r.get('id')}
        fresh = [r for r in records if not r.get('id') or r['id'] not in known]
        records = [known.get(r.get('id'), r) for r in records]
        owner._artifact_results.extend(fresh)
    if queue is not None and not queue.flushing:
        if fresh:
            queue.pending.append((config, fresh))
    else:
        log_results(fresh, owner)
    return records


def finalize_task_artifacts(owner, saves_complete=True):
    """Run while the task's HTML/Markdown capture is still open, before any mail."""
    from .core import send_prepared
    queue = task_delivery(owner)
    assert queue is not None
    queue.flushing = True
    delivery_log('开始投递研究结果', owner=owner)
    if owner._artifacts:
        if saves_complete:
            publish_safe(owner.task_name, owner.init_time.isoformat(), owner._artifacts, owner.execution_status)
        else:
            record = {'status': 'skipped', 'reason': 'asynchronous saves incomplete'}
            owner._artifact_results.append(record)
            log_results([record], owner)
    for config, records in queue.pending:
        try:
            if config is not None:
                send_prepared(records, config)
        except Exception as exc:
            for record in records:
                if record.get('status') == 'prepared':
                    record.update(status='failed', error=str(exc))
        log_results(records, owner)
    for hook in queue.hooks:
        try:
            hook()
        except Exception as exc:
            delivery_log(f'attachment handoff failed: {exc}', failed=True, owner=owner)
    delivery_log('投递阶段结束；published 表示已放入本机共享目录，不代表远端已归档', owner=owner)


def defer_mail(callback):
    queue = task_delivery()
    if queue is None or queue.mailing:
        return None
    state = {'status': 'queued', 'reason': 'waiting for task artifact delivery'}
    queue.mail.append((callback, state))
    return state


def run_deferred_mail(owner):
    queue = task_delivery(owner)
    assert queue is not None
    queue.mailing = True
    for callback, state in queue.mail:
        try:
            result = callback()
            state.clear()
            state.update(result)
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            delivery_log(f'deferred mail failed: {exc}', failed=True, owner=owner)


def register(spec, task, run_id, status='success'):
    model = MODEL_ARTIFACTS.get()
    if model is not None:
        model.append(spec)
        return []
    owner = CURRENT_TASK.get()
    if owner is not None:
        owner.register_artifact(spec)
        return []
    return publish_safe(task, run_id, [spec], status)


def analysis_specs(folder, category):
    folder = Path(folder)
    members = []
    for p in sorted(folder.rglob('*')):
        rel = p.relative_to(folder)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in ANALYSIS_SUFFIXES:
            # Stock-level data and delivery metadata are not analysis summaries.
            if any(x in p.name.lower() for x in ('prediction', 'package_manifest', 'parts.json', 'mail_status')):
                continue
            members.append(rel.as_posix())
    specs = [ArtifactSpec(folder, category=category, members=tuple(members), experiment=folder.name)] if members else []
    for tb in folder.rglob('tensorboard'):
        if tb.is_dir() and not any(x in EXCLUDED_DIRS - {'tensorboard'} for x in tb.relative_to(folder).parts):
            if any(tb.rglob('*tfevents*')):
                specs.append(ArtifactSpec(tb, kind='tensorboard', category=category, name=tb.relative_to(folder).as_posix(), experiment=folder.name))
    return specs


def publish_analysis(folder, category, status='success'):
    if sandboxed():
        return []
    try:
        config = Config.load()
        if not config.enabled or config.role != 'publisher':
            return []
        folder = Path(folder)
        from .core import digest
        import hashlib
        specs = analysis_specs(folder, category)
        fingerprint = hashlib.sha256()
        for spec in specs:
            paths = [spec.source / n for n in spec.members] if spec.members else sorted(p for p in spec.source.rglob('*') if p.is_file())
            for path in paths:
                fingerprint.update(str(path.relative_to(folder)).encode())
                fingerprint.update(digest(path).encode())
        records = publish_safe(category, folder.name + '__' + status + '__' + fingerprint.hexdigest()[:16], specs, status)
        return records
    except Exception as exc:
        delivery_log(f'analysis preparation failed: {exc}', failed=True)
        return [{'status': 'failed', 'error': str(exc)}]


def delivered(records):
    return bool(records) and all(r.get('status') == 'published' for r in records)


def notice(records):
    return '\nArtifact delivery: ' + json.dumps([{k: r[k] for k in ('id', 'status', 'error') if k in r} for r in records]) if records else ''


def handoff_bundle(metadata, folder, category, status='success'):
    try:
        records = publish_analysis(folder, category, status)
        def finish():
            if not delivered(records):
                return
            from .core import atomic_json
            logs = [str(p) for p in Path(folder).glob('*.log') if p.is_file()]
            atomic_json(Path(metadata).with_suffix('.artifacts.json'), {'records': records, 'logs': logs})
            if CURRENT_TASK.get() is not None:
                from src.proj import Proj
                from src.res.rl_experimental.bundle import _parts, bundle_attachments
                _, parts = _parts(Path(metadata))
                old = {p.resolve() for p in [Path(metadata), *parts]}
                for files in (Proj.email_attachments, Proj.exit_files):
                    original = files.pop_all()
                    matched = any(Path(p).resolve() in old for p in original)
                    for p in original:
                        if Path(p).resolve() not in old:
                            files.append(p.path, filename=p.filename)
                    if matched:
                        for p in bundle_attachments(metadata):
                            files.append(p)
        queue = task_delivery()
        if queue is not None and not queue.flushing:
            queue.bundles[str(Path(metadata).resolve())] = records
            queue.hooks.append(finish)
        else:
            finish()
        return records
    except Exception as exc:
        delivery_log(f'bundle handoff failed: {exc}', failed=True)
        return [{'status': 'failed', 'error': str(exc)}]


def bundle_delivery(metadata):
    if sandboxed():
        return None
    queue = task_delivery()
    if queue is not None:
        pending = queue.bundles.get(str(Path(metadata).resolve()))
        if pending is not None and not delivered(pending):
            return None
    sidecar = Path(metadata).with_suffix('.artifacts.json')
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return None


def register_report(source, run_id):
    try:
        spec = ArtifactSpec(source, category='model_training')
        if CURRENT_TASK.get() is None:
            config = Config.load()
            if sandboxed() or not config.enabled or config.role != 'publisher':
                return
            from src.proj import Save
            Save.async_wait_all(caller_name='artifact_exchange')
        register(spec, 'model_training', run_id)
    except Exception as exc:
        delivery_log(f'report registration failed: {exc}', failed=True)


@contextmanager
def model_run(trainer):
    """Collect one model invocation, including standalone and resumed trainers."""
    specs = []
    token = MODEL_ARTIFACTS.set(specs)
    status = 'success'
    run_id = datetime.now(timezone.utc).isoformat() + '__' + uuid.uuid4().hex
    try:
        yield
    except BaseException:
        status = 'failed'
        raise
    finally:
        MODEL_ARTIFACTS.reset(token)
        try:
            config = None if sandboxed() else Config.load()
            if config is not None and config.enabled and config.role == 'publisher':
                # On failure, close still-open writers before considering diagnostics.
                for callback in trainer.callback.callbacks:
                    close = getattr(callback, 'reset_writer', None)
                    if close is not None:
                        close()
                from src.proj import Save
                Save.async_wait_all(caller_name='artifact_exchange')
                base = trainer.config.base_path
                results = base.rslt()
                if results.exists() and any(p.is_file() for p in results.rglob('*')):
                    specs.append(ArtifactSpec(results, category='model_training'))
                events = base.snapshot('tensorboard')
                if events.exists() and any(events.rglob('*tfevents*')):
                    specs.append(ArtifactSpec(events, kind='tensorboard', category='model_training', name=base.full_name))
                specs = [replace(s, category='model_training', experiment=base.full_name, run_id=run_id) for s in specs]
                # Studies may be registered once per date; one final SQLite snapshot suffices.
                specs = list({(s.source, s.kind, s.destination): s for s in specs}.values())
                if specs:
                    publish_safe('model_training', run_id, specs, status)
        except Exception as exc:
            delivery_log(f'model preparation failed: {exc}', failed=True)


def delivered_report_paths(records):
    """Remove a mail attachment only when its current bytes were actually published."""
    from .core import digest, real_tree
    paths = set()
    for record in records:
        if record.get('status') != 'published':
            continue
        for item in record.get('source_files', []):
            try:
                path = real_tree(Path(item['source']))
                if path.suffix.lower() not in ('.log', '.html') and digest(path) == item['sha256']:
                    paths.add(str(path.resolve()))
            except (ValueError, OSError):
                continue
    return paths
