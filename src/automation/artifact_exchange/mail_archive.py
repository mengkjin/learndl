"""Archive one logical notification before SMTP splitting or retry."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

from .core import ArtifactSpec, Config, atomic_json, component, publish_artifacts, sandboxed

__all__ = ['DEFAULT_TASKS', 'freeze_task_attachments', 'archive_message', 'publish_archive']

# The published 0.7.2 patch uses this allowlist to recognize historical mail.
DEFAULT_TASKS = {
    'daily_update', 'weekly_update', 'tracking_update', 'rcquant_sec_backfill',
    'train_model', 'train_schedule_model', 'resume_train', 'resume_training',
    'update_models', 'resume_testing', 'test_model', 'test_linear',
}


def enabled(owner, override=None):
    explicit = override if override is not None else getattr(owner, 'kwargs', {}).get('mail_sync')
    if explicit is None:
        return True
    if isinstance(explicit, str):
        return explicit.strip().lower() in ('true', 'yes', 'on', '1')
    return bool(explicit)


def publisher(owner, override=None):
    if sandboxed() or not enabled(owner, override):
        return None
    config = Config.load()
    return config if config.enabled and config.role == 'publisher' else None


@contextmanager
def freeze_task_attachments(owner):
    """Keep the original bytes available after artifact handoff edits the mail list."""
    from src.proj import Proj

    owner._mail_original = []
    try:
        config = publisher(owner) if getattr(owner, 'email', False) and not owner.should_forfeit else None
    except Exception as exc:
        owner.logger.warning(f'Mail archive configuration failed: {exc}', vb_level=0)
        config = None
    if config is None:
        yield
        return
    try:
        temporary = TemporaryDirectory(prefix='learndl-mail-original-')
    except OSError as exc:
        owner.logger.warning(f'Mail attachment snapshot unavailable: {exc}', vb_level=0)
        yield
        return
    with temporary as work:
        originals = Proj.exit_files.snapshot() + Proj.email_attachments.snapshot()
        for entry in dict.fromkeys(originals):
            try:
                frozen = Path(work) / str(len(owner._mail_original))
                shutil.copyfile(entry.path, frozen)
                owner._mail_original.append((entry.attachment_name, frozen))
            except OSError as exc:
                owner._mail_original.append((entry.attachment_name, str(exc)))
        try:
            yield
        finally:
            owner._mail_original = []


def archive_message(message, *, override=None, main_task=False, notification_id=None):
    from .research import CURRENT_TASK, delivery_log

    owner = CURRENT_TASK.get()
    try:
        config = publisher(owner, override)
        if config is None:
            return
        from src.proj.util.cli.session import git_head

        metadata = {
            'notification_id': notification_id or uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
            'task': owner.task_name if owner is not None else 'script_mail',
            'status': owner.execution_status if owner is not None else 'unknown',
            'task_run_id': getattr(owner, 'task_run_id', None),
            'model_runs': getattr(owner, '_model_run_ids', []),
            'script': getattr(owner, '_script_entry', None), 'revision': git_head(),
            'started_at': owner.init_time.isoformat() if owner is not None else None,
            'finished_at': owner.end_time.isoformat() if owner is not None and hasattr(owner, 'end_time') else None,
            'source': owner.source if owner is not None else None,
            'artifact_ids': [r['id'] for r in getattr(owner, '_artifact_results', []) if r.get('id')],
            'smtp_status': 'unknown',
        }
        originals = getattr(owner, '_mail_original', ()) if main_task else ()
        records = publish_archive(config, message, metadata, originals=originals)
        for record in records:
            if record.get('status') != 'published':
                reason = record.get('error') or record.get('reason') or record['status']
                delivery_log(f'Mail archive {record["status"]}: {reason}', failed=True, owner=owner)
    except Exception as exc:
        delivery_log(f'Mail archive failed: {exc}', failed=True, owner=owner)



def publish_archive(config, message, metadata, *, originals=(), historical_parts=(), prepare_only=False):
    """Publish a live notification or a historical receipt through the same archive format."""
    from email import message_from_bytes
    from hashlib import sha256
    from src.proj.util.web.mail_package import wire

    with TemporaryDirectory(prefix='learndl-mail-archive-') as work:
        folder = Path(work).resolve()
        raw = historical_parts[0] if historical_parts else wire(message)
        (folder / 'message.eml').write_bytes(raw)
        attachments = []
        bodies = []
        included = set()

        def attachment(name, data, *, smtp):
            key = (name, sha256(data).digest())
            if key in included and (not smtp or historical_parts):
                return
            included.add(key)
            display = Path(name.replace(chr(92), '/')).name
            prefix = f'{len(attachments) + 1:03d}-'
            suffix = Path(display).suffix
            safe_name = prefix + component(Path(display).stem)[:100] + (
                '.' + component(suffix[1:])[:15] if suffix else '')
            relative = (Path('attachments') / safe_name).as_posix()
            output = folder / relative
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(data)
            attachments.append({'name': name, 'file': relative, 'smtp': smtp})

        for index, part_message in enumerate((message, *(message_from_bytes(raw) for raw in historical_parts[1:]))):
            if index:
                attachment(f'原始邮件-{index + 1:03d}.eml', historical_parts[index], smtp=False)
            for part in part_message.walk():
                if part.is_multipart():
                    continue
                decoded = part.get_payload(decode=True)
                data = decoded if isinstance(decoded, bytes) else b''
                if name := part.get_filename():
                    attachment(name, data, smtp=True)
                elif part.get_content_maintype() == 'text':
                    body = data.decode(part.get_content_charset() or 'utf-8', errors='replace')
                    if body not in bodies:
                        bodies.append(body)

        missing = []
        for name, frozen in originals:
            if isinstance(frozen, Path):
                attachment(name, frozen.read_bytes(), smtp=False)
            else:
                missing.append(name)
        body = '\n\n'.join(bodies)
        (folder / 'body.txt').write_text(body, encoding='utf-8')
        if 'Warning: Attachment skipped:' in body:
            missing.append('See attachment warnings in body.txt')
        metadata = metadata | {
            'subject': str(message.get('Subject', '')), 'host': config.source_host,
            'attachments': attachments, 'missing_attachments': missing,
        }
        atomic_json(folder / 'mail.json', metadata)
        return publish_artifacts(metadata['task'], metadata['notification_id'], [
            ArtifactSpec(folder, category='mail_archive', experiment=metadata['task'], destination='mail'),
        ], metadata['status'], config=config, prepare_only=prepare_only)
