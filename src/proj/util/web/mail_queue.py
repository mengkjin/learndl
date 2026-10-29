"""Durable mail batches, bounded retries and non-recursive delivery notices.

All mutations require the caller's outbox lock. SMTP acceptance is at-least-once:
process death between acceptance and receipt persistence can cause a duplicate.
"""
from __future__ import annotations

import json
import shutil
import smtplib
import time
import uuid
from email import message_from_bytes, policy
from pathlib import Path

from .mail_package import atomic_bytes, atomic_json, digest, fingerprint, fsync_dir, prepare

DELAYS = (60, 300, 900, 3600)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def delay(attempt):
    return DELAYS[min(max(attempt - 1, 0), len(DELAYS) - 1)]


class MailQueue:
    def __init__(self, api, root: Path):
        self.api, self.root = api, Path(root)

    def state(self):
        return read_json(self.root / '.transport.json')

    def cooling(self):
        return self.state().get('next_retry_at', 0) > time.time()

    def connected(self):
        state = self.state()
        state.update(consecutive=0, cooldowns=0, next_retry_at=0)
        atomic_json(self.root / '.transport.json', state)

    def connect_failure(self, error):
        state = self.state()
        count = state.get('consecutive', 0) + 1
        state.update(consecutive=count, last_error=f'{type(error).__name__}: {error}')
        if count >= 2 or isinstance(error, smtplib.SMTPAuthenticationError):
            rounds = state.get('cooldowns', 0) + 1
            state.update(cooldowns=rounds, next_retry_at=time.time() + delay(rounds))
        atomic_json(self.root / '.transport.json', state)

    def notice(self, key, *, subject, size, status, error='', attempts=0, next_retry_at=0, progress=''):
        path = self.root / '.notices.json'
        notices = read_json(path)
        old = notices.get(key, {})
        notices[key] = dict(old, subject=str(subject), size=size, status=status, error=error[:2000],
                            attempts=attempts, next_retry_at=next_retry_at, progress=progress,
                            first_failed_at=old.get('first_failed_at', time.time()))
        atomic_json(path, notices)

    def resolved(self, key, progress=""):
        notices = read_json(self.root / '.notices.json')
        if key in notices and notices[key]['status'] != 'recovered':
            notices[key].update(status='recovered', next_retry_at=0, error='', progress=progress)
            atomic_json(self.root / '.notices.json', notices)

    @staticmethod
    def signature(row):
        # Repeated attempts for the same problem must not generate repeated warnings.
        return digest(json.dumps([row['status'], row['error']], ensure_ascii=False).encode())

    def deliver_notices(self, budget):
        if self.cooling() or not budget.available():
            return
        notices_path = self.root / '.notices.json'
        notices = read_json(notices_path)
        receipt_path = self.root / '.notice_delivery.json'
        receipt = read_json(receipt_path)
        if receipt.get('next_retry_at', 0) > time.time():
            return
        if receipt.get('complete', True):
            changed = {}
            for key, row in notices.items():
                signature = self.signature(row)
                if signature == row.get('notified_signature'):
                    continue
                if row['status'] == 'recovered' and not row.get('notified_signature'):
                    continue
                if row['status'] == 'pending' and time.time() < row['first_failed_at'] + 300:
                    continue
                changed[key] = dict(row, signature=signature)
            if not changed or (receipt.get('finished_at', 0) + 1800 > time.time()):
                return
            sections = []
            for key, row in changed.items():
                sections.append(f'ID: {key}\nSubject: {row["subject"][:500]}\nBytes: {row["size"]}\n'
                                f'Status: {row["status"]}\nAttempts: {row["attempts"]}\n'
                                f'Next retry (UTC epoch): {row["next_retry_at"]}\n'
                                f'Progress: {row["progress"]}\nError: {row["error"]}\n')
            pages, body = [], ''
            for section in sections:
                if len((body + section).encode()) > 40_000 and body:
                    pages.append(body)
                    body = ''
                body += section + '\n'
            if body:
                pages.append(body)
            receipt = dict(complete=False, changed=changed, pages=[dict(body=p, sent=False) for p in pages],
                           id=uuid.uuid4().hex)
            atomic_json(receipt_path, receipt)
        for index, page in enumerate(receipt['pages'], 1):
            if page['sent'] or not budget.available():
                continue
            message = self.api.message(f'Mail delivery status [{index}/{len(receipt["pages"])}]',
                                       '邮件发送异常 / 恢复摘要（无原始附件）\n\n' + page['body'])
            message['Message-ID'] = f'<notice.{receipt["id"]}.{index}@learndl.local>'
            budget.attempt()
            try:
                self.api._deliver(message)
            except Exception as error:
                # No notification about a failed notification. Persist and retry after cooling.
                receipt.update(last_error=f'{type(error).__name__}: {error}', next_retry_at=time.time()+300)
                atomic_json(receipt_path, receipt)
                if getattr(error, 'mail_stage', 'connect') in ('connect', 'tls', 'auth'):
                    self.connect_failure(error)
                return
            self.connected()
            page['sent'] = True
            atomic_json(receipt_path, receipt)
        if all(p['sent'] for p in receipt['pages']):
            current = read_json(notices_path)
            for key, row in receipt['changed'].items():
                if key in current:
                    current[key]['notified_signature'] = row['signature']
            atomic_json(notices_path, current)
            receipt.update(complete=True, finished_at=time.time(), next_retry_at=0)
            atomic_json(receipt_path, receipt)

    def prepare_folder(self, folder):
        target = folder / 'payload'
        if target.is_dir():
            manifest = read_json(target / 'manifest.json')
            if (manifest.get('schema_version') != 2 or not isinstance(manifest.get('messages'), list)
                    or not manifest['messages'] or not isinstance(manifest.get('recipient'), str)
                    or '@' not in manifest['recipient'] or not isinstance(manifest.get('subject'), str)):
                raise ValueError('Invalid mail batch manifest')
            for item in manifest['messages']:
                if (not isinstance(item, dict) or type(item.get('sent')) is not bool
                        or type(item.get('attempts')) is not int or item['attempts'] < 0
                        or not isinstance(item.get('file'), str) or Path(item['file']).name != item['file']
                        or not isinstance(item.get('sha256'), str) or type(item.get('mime_bytes')) is not int
                        or not isinstance(item.get('next_retry_at', 0), (int, float))):
                    raise ValueError('Invalid mail batch item')
            return manifest
        message, _ = self.api._read_outbox(folder)
        for old in folder.glob('.prepare-*'):
            if old.is_dir():
                shutil.rmtree(old)
        staging = folder / f'.prepare-{uuid.uuid4().hex}'
        try:
            manifest = prepare(message, staging, folder.name, self.api.max_message_bytes,
                               source_size=(folder/'message.eml').stat().st_size)
            staging.replace(target)
            fsync_dir(folder)
            return manifest
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def archive(self, folder):
        self.resolved(folder.name, 'all messages accepted')
        meta = read_json(folder / 'meta.json')
        if meta.get('confirmation_message'):
            self.api.logger.success(f'Send email {meta["confirmation_message"]}')
        dest = self.root / '.archive' / folder.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f'Existing mail backup: {dest}')
        folder.replace(dest)
        fsync_dir(dest.parent)
        fsync_dir(self.root)

    def fail_preparation(self, folder, error):
        try:
            meta = read_json(folder / 'meta.json')
        except (ValueError, OSError):
            meta = {}
        key = folder.name
        meta.update(last_error=f'prepare: {type(error).__name__}: {error}', status='isolated')
        self.api._move_dead(folder, meta)
        self.notice(key, subject=meta.get('subject', key), size=meta.get('source_bytes', 0),
                    status='isolated', error=meta['last_error'])
        self.api.logger.error(f'Email isolated; original preserved: {key}: {meta["last_error"]}')

    def process(self, folder, manifest, budget):
        """Attempt due items; each failure belongs to an item, not the whole queue."""
        ledger = folder / 'payload' / 'manifest.json'
        for index, item in enumerate(manifest['messages'], 1):
            key = f'{folder.name}/{index}'
            if item['sent']:
                self.resolved(key, f'{sum(m["sent"] for m in manifest["messages"])}/{len(manifest["messages"])} accepted')  # recover crash after receipt but before notice update
                continue
            if item.get('status') == 'isolated' or item.get('next_retry_at', 0) > time.time():
                continue
            if not budget.available() or self.cooling():
                break
            path = folder / 'payload' / item['file']
            try:
                if Path(item['file']).name != item['file']:
                    raise ValueError('Unsafe payload filename')
                raw = path.read_bytes()
                if digest(raw) != item['sha256']:
                    raise ValueError('Mail payload checksum mismatch')
                budget.attempt()
                self.api._deliver(message_from_bytes(raw, policy=policy.default), recipient=manifest['recipient'], timeout=budget.timeout)
            except Exception as error:
                stage = getattr(error, 'mail_stage', 'connect' if (isinstance(error, OSError)
                                and not isinstance(error, smtplib.SMTPException) or
                                isinstance(error, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError))) else 'submit')
                text = f'{stage}: {type(error).__name__}: {error}'
                connection = stage in ('connect', 'tls', 'auth')
                item.update(last_error=text, error_stage=stage)
                if connection:
                    self.connect_failure(error)
                    item['next_retry_at'] = time.time() + 60
                else:
                    self.connected()
                    item['attempts'] += 1
                    permanent = (isinstance(error, (ValueError, FileNotFoundError)) or
                                 isinstance(error, smtplib.SMTPRecipientsRefused) or
                                 isinstance(error, smtplib.SMTPResponseException) and error.smtp_code >= 500)
                    item['status'] = 'isolated' if permanent or item['attempts'] >= 5 else 'pending'
                    item['next_retry_at'] = time.time() + delay(item['attempts'])
                atomic_json(ledger, manifest)
                progress = f'{sum(m["sent"] for m in manifest["messages"])}/{len(manifest["messages"])} accepted'
                self.notice(key, subject=manifest['subject'], size=item['mime_bytes'],
                            status=item.get('status', 'pending'), error=text, attempts=item['attempts'],
                            next_retry_at=item['next_retry_at'], progress=progress)
                budget.stats['error'] = text
                self.api.logger.error(f'Email {key} ({item["mime_bytes"]} bytes): {text}; '
                                      f'{item.get("status", "pending")}, next retry {item["next_retry_at"]}')
                continue
            self.connected()
            item.update(sent=True, status='sent', next_retry_at=0, last_error='')
            atomic_json(ledger, manifest)
            self.resolved(key, f'{sum(m["sent"] for m in manifest["messages"])}/{len(manifest["messages"])} accepted')
            budget.stats['sent'] += 1
        return all(item['sent'] for item in manifest['messages'])

    def flush(self, *, timeout=20, max_messages=20, deadline=None, stats=None):
        budget = Budget(timeout, max_messages, deadline, stats)
        self.deliver_notices(budget)
        folders = self.api._pending_outbox()
        # Preparing old oversized entries is deferred until existing small mail is attempted.
        folders.sort(key=lambda p: ((p / 'message.eml').stat().st_size > self.api.max_message_bytes
                                   if (p / 'message.eml').exists() else False, p.name))
        for folder in folders:
            if not budget.available() or self.cooling():
                break
            try:
                manifest = self.prepare_folder(folder)
            except Exception as error:
                self.fail_preparation(folder, error)
                budget.stats['error'] = f'prepare: {error}'
                continue
            if self.process(folder, manifest, budget):
                self.archive(folder)
            elif all(item['sent'] or item.get('status') == 'isolated' for item in manifest['messages']):
                self.api._move_dead(folder)
        return budget.stats

    def external(self, message, recipient=None, delivery_id=None, timeout=20):
        """Caller owns scheduling; this receipt is never scanned by the watchdog."""
        if recipient:
            if message.get('To'):
                message.replace_header('To', recipient)
            else:
                message['To'] = recipient
        content = fingerprint(message)
        key = digest((delivery_id or content).encode())
        folder = self.root / '.external' / key
        if not folder.exists():
            staging = folder.with_name(f'.{key}.{uuid.uuid4().hex}.tmp')
            staging.mkdir(parents=True)
            try:
                atomic_bytes(staging / 'message.eml', message.as_bytes())
                atomic_json(staging / 'meta.json', {'attempts': 0, 'subject': str(message.get('Subject', ''))})
                staging.replace(folder)
                fsync_dir(folder.parent)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        original, _ = self.api._read_outbox(folder)
        if fingerprint(original) != content:
            raise ValueError('delivery_id reused with changed email content')
        del original
        try:
            manifest = self.prepare_folder(folder)
        except Exception as error:
            self.notice(key, subject=str(message.get('Subject', '')), size=len(message.as_bytes()),
                        status='isolated', error=f'prepare: {error}')
            raise
        budget = Budget(timeout, None, None, None)
        accepted = self.process(folder, manifest, budget)
        if accepted:
            self.resolved(key, 'all messages accepted')
        return accepted


class Budget:
    def __init__(self, timeout, maximum, deadline, stats):
        self.timeout, self.maximum, self.deadline = timeout, maximum, deadline
        self.stats = stats if stats is not None else {}
        self.stats.update(attempted=0, sent=0, processed=0)

    def available(self):
        return ((self.maximum is None or self.stats['attempted'] < self.maximum)
                and (self.deadline is None or time.monotonic() < self.deadline))

    def attempt(self):
        self.stats['attempted'] += 1
        self.stats['processed'] += 1


def inspect_queue(api):
    """Read-only view: no locks/files created, no SMTP connections."""
    root = api._queue().root
    rows = []
    for base in (root, root / 'dead', root / '.external', root / '.archive'):
        if not base.is_dir():
            continue
        for folder in sorted(base.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or (base == root and folder.name == 'dead'):
                continue
            try:
                source = folder / 'message.eml'
                manifest = read_json(folder / 'payload' / 'manifest.json')
                meta = read_json(folder / 'meta.json')
                rows.append(dict(id=folder.name, location=str(folder), source_bytes=source.stat().st_size,
                                 needs_conversion=source.stat().st_size > api.max_message_bytes and not manifest,
                                 subject=manifest.get('subject', meta.get('subject')),
                                 messages=manifest.get('messages', []), last_error=meta.get('last_error')))
            except Exception as error:
                rows.append(dict(id=folder.name, location=str(folder), error=str(error)))
    return dict(max_mime_bytes=api.max_message_bytes, seven_zip=shutil.which('7zz') or shutil.which('7z'),
                transport=read_json(root / '.transport.json'), notices=read_json(root / '.notices.json'),
                notice_delivery=read_json(root / '.notice_delivery.json'), entries=rows)


def main():
    import argparse
    from .emailer import Email
    parser = argparse.ArgumentParser(description='Inspect mail (default) or explicitly requeue an isolated batch.')
    parser.add_argument('action', choices=['inspect', 'retry'], nargs='?', default='inspect')
    parser.add_argument('batch_id', nargs='?')
    parser.add_argument('--external', action='store_true', help='Reset a caller-owned receipt; never enqueue it')
    args = parser.parse_args()
    if args.action == 'inspect':
        print(json.dumps(inspect_queue(Email), ensure_ascii=False, indent=2))
        return
    if not args.batch_id or Path(args.batch_id).name != args.batch_id or args.batch_id.startswith('.'):
        parser.error('retry requires a valid batch ID')
    with Email._outbox_lock() as acquired:
        if not acquired:
            raise RuntimeError('Email outbox busy')
        root = Email._queue().root
        source = root / ('.external' if args.external else 'dead') / args.batch_id
        dest = source if args.external else root / args.batch_id
        if not source.is_dir() or (not args.external and dest.exists()):
            raise ValueError('Batch must exist only in dead/; inspect before retry')
        ledger = source / 'payload' / 'manifest.json'
        if ledger.exists():
            manifest = read_json(ledger)
            for item in manifest['messages']:
                if not item['sent']:
                    item.update(status='pending', attempts=0, next_retry_at=0)
            atomic_json(ledger, manifest)
        if not args.external:
            source.replace(dest)
        fsync_dir(root)
        print(f'Requeued {args.batch_id}; no mail sent by this command.')


if __name__ == '__main__':
    main()
