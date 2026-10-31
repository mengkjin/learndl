"""Size-bounded task result mail, including closed HtmlCatcher attachments.

Small results stay in one message. Large results use numbered messages with a
durable per-message receipt; never enqueue a second copy in the shared outbox.
Retry: python -m src.proj.util.web.task_mail DELIVERY_DIRECTORY
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from email import message_from_bytes
from pathlib import Path

import portalocker

from src.proj.util.web.mail_package import MAX_BYTES, atomic_json, prepare


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    atomic_json(path, value)


def _skip_disabled() -> bool:
    from src.proj.env import MACHINE
    if MACHINE.emailable:
        return False
    print('[Task/mail] skipped: email disabled on this machine', flush=True)
    return True


class TaskResultMail:
    """Opt-in sender; True means delivered or disabled by machine policy.

    Disabled delivery is a successful no-op, without creating SMTP receipts.
    max_bytes bounds serialized MIME, not ZIPs.
    """

    def __init__(self, directory, *, max_bytes=MAX_BYTES, email_api=None, transport=None):
        self.directory = Path(directory).resolve()
        self.max_bytes = max_bytes
        self.email_api = email_api
        self.transport = transport

    def __call__(self, title, body, recipient=None, *, attachments=None,
                 project_attachments=False, confirmation_message=''):
        if _skip_disabled():
            return True
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
        message = api.message(title, body, api.recipient(recipient), attachments=paths)
        prepared = prepare(message, directory, str(directory), limit=self.max_bytes)
        state = {'schema_version': 2, 'recipient': api.recipient(recipient),
                 'max_mime_bytes': self.max_bytes, 'attachments': prepared['attachments'],
                 'messages': prepared['messages'], 'status': 'prepared'}
        save_json(ledger, state)
        return retry_delivery(directory, transport=self.transport)


def retry_delivery(directory, *, transport=None):
    """Retry unsent mail; disabled machines return True without touching receipts."""
    if _skip_disabled():
        return True
    directory = Path(directory)
    with portalocker.Lock(directory/'.send.lock', timeout=0):
        return _retry_delivery(directory, transport=transport)


def _retry_delivery(directory, *, transport=None):
    directory = Path(directory)
    ledger = directory/'delivery.json'
    state = json.loads(ledger.read_text())
    if transport is None:
        from src.proj.util.web.emailer import Email

        def default_transport(message, recipient):
            # Existing ledgers can still contain oversized legacy MIME messages.
            # The external receipt records any sub-parts without a second queue owner.
            from src.proj.util.web.mail_package import fingerprint
            return Email.send_with_smtplib(message, recipient,
                delivery_id=f'task-result:{directory}:{fingerprint(message)}')

        transport = default_transport
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
    print(f'[Task/mail] {sum(item["sent"] for item in state["messages"])}/{len(state["messages"])} messages accepted; receipt: {ledger}', flush=True)
    return state['status'] == 'sent'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Retry only unsent task result messages (no task rerun).')
    parser.add_argument('directory')
    args = parser.parse_args()
    return 0 if retry_delivery(args.directory) else 1


if __name__ == '__main__':
    raise SystemExit(main())
