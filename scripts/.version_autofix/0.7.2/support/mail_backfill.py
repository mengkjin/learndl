"""Inspect or publish recent SMTP receipts without invoking SMTP."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email import message_from_bytes, policy
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import re

from src.automation.artifact_exchange.core import Config, atomic_json, digest, real_tree, safe_path, send_prepared
from src.automation.artifact_exchange.mail_archive import DEFAULT_TASKS, publish_archive

WINDOW_FILE = '0.7.2-mail-window.json'


def _date(value):
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        try:
            stamp = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def _identity(message):
    subject = re.sub(r'^Learndl\s*\[[^\]]+\]:\s*', '', str(message.get('Subject', '')), flags=re.I)
    for task in sorted(DEFAULT_TASKS):
        label = re.escape(task.replace('_', ' '))
        match = re.match(rf'^(Success|Error|Failed|Warning|Skipped)\s+-\s+{label}\s+-(?:\s|$)',
                         subject, re.I)
        if match:
            return task, match[1].capitalize()
    return None


def _read(path):
    return json.loads(real_tree(path).read_text())


def _receipt(path):
    record = _read(path)
    entries = record['messages']
    if not entries or not all(entry.get('sent') is True for entry in entries):
        raise ValueError('SMTP acceptance is incomplete')
    return entries


def _sources(outbox, task_roots):
    for folder in sorted((outbox / '.archive').glob('*')):
        if folder.is_dir() and not folder.name.startswith('.'):
            yield 'archive', folder, folder / 'payload/manifest.json'
    ledgers = {ledger for root in task_roots if root.is_dir() for ledger in root.rglob('delivery.json')}
    for ledger in sorted(ledgers):
        yield 'task', ledger.parent, ledger
    for folder in sorted((outbox / '.external').glob('*')):
        if folder.is_dir() and not folder.name.startswith('.'):
            yield 'external', folder, folder / 'payload/manifest.json'


def _payload(kind, folder, entries, cutoff):
    if kind != 'task':
        meta = _read(folder / 'meta.json')
        stamp = _date(meta.get('created_at'))
        path = real_tree(folder / 'message.eml')
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if stamp is not None and stamp < cutoff:
            return None
        data = path.read_bytes()
        message = message_from_bytes(data, policy=policy.default)
        header = _date(str(message.get('Date', '')))
        return message, (data,), (stamp or header or mtime), (
            'queue-created_at' if stamp else 'mail-date' if header else 'eml-mtime')
    # A task ledger can authenticate multiple prepared pieces of one logical
    # message. Its on-disk parts are the available historical originals.
    parts = []
    times = []
    for item in entries:
        relative = safe_path(item['file'])
        if len(relative.parts) != 1:
            raise ValueError('Mail part must be inside the task ledger directory')
        path = real_tree(folder / relative)
        if digest(path) != item['sha256']:
            raise ValueError('Mail part differs from its receipt')
        times.append(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
        parts.append(path.read_bytes())
    message = message_from_bytes(parts[0], policy=policy.default)
    header = _date(str(message.get('Date', '')))
    return message, tuple(parts), header or min(times), 'mail-date' if header else 'eml-mtime'


def run(config, outbox, task_roots=(), *, report=None, now=None):
    end = now or datetime.now(timezone.utc)
    rows = []
    if config.role != 'publisher':
        return {'role': config.role, 'rows': rows}
    if report is not None:
        if not config.enabled:
            raise ValueError('Enable Artifact Exchange on the publisher before applying the mail patch')
        window = real_tree(config.state / WINDOW_FILE)
        if window.exists():
            end = datetime.fromisoformat(_read(window)['end'])
        else:
            atomic_json(window, {'end': end.isoformat()})
    start = end - timedelta(days=7)
    result = {'role': config.role, 'start': start.isoformat(), 'end': end.isoformat(), 'rows': rows}
    seen = set()
    for kind, folder, receipt in _sources(real_tree(outbox), task_roots):
        row = {'source': str(folder), 'type': kind}
        rows.append(row)
        try:
            entries = _receipt(receipt)
            payload = _payload(kind, folder, entries, start)
            if payload is None or not start <= payload[2] <= end:
                row.update(state='outside-window')
                continue
            message, parts, stamp, stamp_source = payload
            if (identity := _identity(message)) is None:
                row.update(state='skipped', reason='Not a recognized Mail2Syncthing task notification')
                continue
            from src.proj.util.web.mail_package import fingerprint

            fingerprints = {fingerprint(message_from_bytes(raw), normalize_body_newlines=True,
                                        normalize_header_folding=True) for raw in parts}
            if fingerprints <= seen:
                row.update(state='duplicate')
                continue
            seen.update(fingerprints)
            task, status = identity
            notification = sha256(f'0.7.2:{kind}:{folder.relative_to(outbox) if folder.is_relative_to(outbox) else folder}'.encode()).hexdigest()
            row.update(state='matched', task=task, created_at=stamp.isoformat(), time_source=stamp_source,
                       notification_id=notification)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(state='skipped', reason=str(exc))
            continue
        if report is None:
            continue
        try:
            receipt_rows = publish_archive(config, message, {
                'notification_id': notification, 'task': task, 'status': status,
                'created_at': stamp.isoformat(), 'time_source': stamp_source, 'smtp_status': 'sent',
                'source': f'historical-{kind}', 'historical_parts': len(parts),
            }, historical_parts=parts, prepare_only=True)
            if len(receipt_rows) != 1:
                raise ValueError('Expected one mail archive package')
            record = receipt_rows[0]
            if record['status'] == 'prepared':
                ledger = real_tree(config.state / 'outbox' / record['outbox_key'] / 'delivery.json')
                if _read(ledger)['id'] != record['id'] or digest(ledger.with_name('package.zip')) != record['sha256']:
                    raise ValueError('Prepared package could not be verified')
                try:
                    send_prepared(receipt_rows, config)
                except OSError as exc:
                    record['error'] = str(exc)
                if record['status'] != 'published':
                    if digest(ledger.with_name('package.zip')) != record['sha256']:
                        raise ValueError('Pending mail package failed verification')
                    row.update(state='pending', package_id=record['id'], reason=str(record.get('error') or 'Awaiting Syncthing publication'))
                else:
                    row.update(state='published', package_id=record['id'])
            elif record['status'] == 'published':
                row.update(state='published', package_id=record['id'])
            else:
                raise ValueError(record.get('error') or record.get('reason') or 'Package was not prepared')
        except (OSError, ValueError, KeyError) as exc:
            row.update(state='failed', reason=str(exc))
    result['counts'] = {key: sum(r['state'] == key for r in rows) for key in (
        'matched', 'published', 'pending', 'outside-window', 'skipped', 'duplicate', 'failed')}
    result['counts']['scanned'] = len(rows)
    result['counts']['matched'] += sum(result['counts'][key] for key in ('published', 'pending', 'failed'))
    if report is not None:
        atomic_json(report, result)
        if result['counts']['failed']:
            raise ValueError(f'Historical mail backfill incomplete; see {report}')
    return result


def host_inputs():
    from src.proj import PATH

    return PATH.email_outbox, (PATH.main / 'results', PATH.logs)


def main():
    from src.proj import Logger

    Logger.stdout(json.dumps(run(Config.load(), *host_inputs()), ensure_ascii=False, indent=2), vb_level=0)


if __name__ == '__main__':
    main()
