"""Size-bounded task result mail, including closed HtmlCatcher attachments.

Small results stay in one message. Large results use numbered messages with a
durable per-message receipt; never enqueue a second copy in the shared outbox.
No training or database imports. Retry: python -m ...task_mail DELIVERY_DIRECTORY
"""
from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime, timezone
from email import message_from_bytes
from pathlib import Path

import portalocker

from .bundle import _sha256


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class TaskResultMail:
    """Opt-in AutoRunTask sender; max_bytes bounds serialized MIME, not ZIPs."""

    def __init__(self, directory, *, max_bytes=18*1024**2, email_api=None, transport=None):
        self.directory = Path(directory).resolve()
        self.max_bytes = max_bytes
        self.email_api = email_api
        self.transport = transport

    def __call__(self, title, body, recipient=None, *, attachments=None,
                 project_attachments=False, confirmation_message=''):
        from src.proj.env.variable.files import EmailAttachment
        if self.email_api is None:
            from src.proj.util.web.emailer import Email
            self.email_api = Email
        api = self.email_api
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        ledger = directory/'delivery.json'
        if ledger.exists():
            raise FileExistsError(f'Mail already prepared; retry only failed parts: {directory}')
        paths = list(attachments or [])
        if project_attachments:
            from src.proj import Proj
            paths.extend(Proj.email_attachments.pop_all())
        # Conservative raw part cap; actual serialized size is checked below.
        cap = int(self.max_bytes*.65) - 4096
        if cap < 4096:
            raise ValueError('email size budget too small')
        prepared, inventory = [], []
        for index, item in enumerate(paths):
            path = item.path if isinstance(item, EmailAttachment) else Path(item)
            name = item.attachment_name if isinstance(item, EmailAttachment) else path.name
            if Path(name).name != name or name in ('', '.', '..'):
                raise ValueError(f'Unsafe attachment filename: {name}')
            if not path.is_file():
                raise FileNotFoundError(f'Mail attachment missing: {path}')
            destination = directory/f'attachment-{index:03d}'
            destination.mkdir()
            # Keep source immutable; compression sharply reduces large HTML logs.
            if path.suffix.lower() == '.html' and path.stat().st_size > 512*1024:
                name += '.gz'
                target = destination/name
                with path.open('rb') as source, gzip.open(target, 'wb') as out:
                    shutil.copyfileobj(source, out)
            else:
                target = destination/name
                shutil.copy2(path, target)
            inventory.append({'source': str(path), 'source_bytes': path.stat().st_size,
                              'attachment': name, 'bytes': target.stat().st_size})
            if target.stat().st_size <= cap:
                prepared.append(target)
            else:
                # Handles both unusually large logs and legacy oversized ZIP parts.
                parts = []
                with target.open('rb') as source:
                    while chunk := source.read(cap):
                        part = destination/f'{name}.mailpart{len(parts)+1:03d}'
                        part.write_bytes(chunk)
                        parts.append({'name': part.name, 'sha256': _sha256(part), 'size': len(chunk)})
                        prepared.append(part)
                definition = destination/f'{name}.mailparts.json'
                save_json(definition, {'filename': name, 'sha256': _sha256(target), 'parts': parts,
                                      'instructions': 'Concatenate mailpart files in listed order before analyze-bundle. For .gz then gunzip.'})
                prepared.append(definition)
        address = api.recipient(recipient)
        note = '\nAll numbered messages belong to this task. Keep all result parts and JSON together. Large HTML logs use gzip.\n'
        groups = [[]]
        # Reserve space for numbering (also verified on the final MIME bytes).
        for path in prepared:
            candidate = groups[-1] + [path]
            raw = api.message(title+' [99999/99999]', body+note, address, attachments=candidate).as_bytes()
            if len(raw) > self.max_bytes:
                if not groups[-1]:
                    raise ValueError(f'Attachment cannot fit mail size budget: {path}')
                groups.append([path])
            else:
                groups[-1] = candidate
        state = {'schema_version': 1, 'recipient': address, 'max_mime_bytes': self.max_bytes,
                 'attachments': inventory, 'messages': [], 'status': 'prepared'}
        for index, group in enumerate(groups, 1):
            subject = title if len(groups) == 1 else f'{title} [{index}/{len(groups)}]'
            message = api.message(subject, body+note, address, attachments=group)
            raw = message.as_bytes()
            if len(raw) > self.max_bytes:
                raise ValueError('Final MIME exceeds configured limit')
            path = directory/f'message-{index:03d}.eml'
            path.write_bytes(raw)
            state['messages'].append({'file': path.name, 'sha256': _sha256(path),
                                      'mime_bytes': len(raw), 'sent': False, 'attempts': 0})
        save_json(ledger, state)
        return retry_delivery(directory, transport=self.transport)


def retry_delivery(directory, *, transport=None):
    """Send only messages without SMTP acceptance; preserve all receipts/errors."""
    directory = Path(directory)
    with portalocker.Lock(directory/'.send.lock', timeout=0):
        return _retry_delivery(directory, transport=transport)


def _retry_delivery(directory, *, transport=None):
    directory = Path(directory)
    ledger = directory/'delivery.json'
    state = json.loads(ledger.read_text())
    if transport is None:
        from src.proj.util.web.emailer import Email
        from src.proj.env import MACHINE

        def transport(message, recipient):
            if not MACHINE.emailable:
                raise RuntimeError('Email disabled on this machine')
            Email._deliver(message, recipient)
            return True
    for item in state['messages']:
        if item['sent']:
            continue
        path = directory/item['file']
        if path.parent.resolve() != directory.resolve() or _sha256(path) != item['sha256']:
            raise ValueError(f'Mail payload checksum/path invalid: {path}')
        item['attempts'] += 1
        item['attempted_at'] = datetime.now(timezone.utc).isoformat()
        # Persist an in-flight marker: interruption after SMTP acceptance is
        # ambiguous, never silently resend it on the next invocation.
        if item.get('in_flight'):
            raise RuntimeError(f'Unknown prior SMTP outcome; inspect before retry: {path}')
        item['in_flight'] = True
        save_json(ledger, state)
        try:
            accepted = transport(message_from_bytes(path.read_bytes()), state['recipient'])
            if not accepted:
                raise RuntimeError('SMTP transport did not confirm acceptance')
            item.update(sent=True, error=None)
        except Exception as exc:
            item.update(sent=False, error=f'{type(exc).__name__}: {exc}')
        item['in_flight'] = False
        save_json(ledger, state)
    state['status'] = 'sent' if all(item['sent'] for item in state['messages']) else 'failed'
    save_json(ledger, state)
    print(f'[RL/mail] {sum(item["sent"] for item in state["messages"])}/{len(state["messages"])} messages accepted; receipt: {ledger}', flush=True)
    return state['status'] == 'sent'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retry only unsent RL task result messages (no training).')
    parser.add_argument('directory')
    args = parser.parse_args()
    raise SystemExit(0 if retry_delivery(args.directory) else 1)
