"""One durable delivery per training attempt, including failures; no SMTP in offline tests."""
from __future__ import annotations

__all__ = [
    'Tee',
    'capture_log',
    'send_training_result',
]
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import json
import sys
import traceback
from .bundle import write_json


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def capture_log(path):
    with Path(path).open('x', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def send_training_result(folder, result, recipient=None, fallback_log=None, delivery_root=None, *, _transfers=None):
    """Mail errors are recorded, never replace a training exception or abort later models."""
    from src.proj import MACHINE
    from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
    folder = Path(folder)
    destination = Path(delivery_root or folder/'training_mail') / result['name']
    from src.automation.artifact_exchange.research import publish_safe, notice, delivered
    from src.automation.artifact_exchange import ArtifactSpec
    # Persist a completed per-training summary even when SMTP is disabled.
    transfers = []

    try:
        destination.mkdir(parents=True, exist_ok=True)
        snapshot = destination/'training_result.json'
        if not snapshot.exists():
            snapshot.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        transfers = (_transfers if _transfers is not None else publish_safe('factor_sequence', str(destination),
            [ArtifactSpec(snapshot, category='factor_sequence', experiment=snapshot.parent.name)], result['status']))
        from src.automation.artifact_exchange.research import defer_mail
        queued = defer_mail(lambda: send_training_result(folder, dict(result), recipient, fallback_log, delivery_root, _transfers=transfers))
        if queued is not None:
            queued['receipt'] = str(destination/'delivery.json')
            write_json(destination/'status.json', queued)
            return queued
        if not MACHINE.emailable:
            state = {'status': 'skipped', 'reason': 'Email disabled on this machine'}
        elif (destination/'delivery.json').exists():
            state = {'status': 'sent' if retry_delivery(destination) else 'failed'}
        else:
            log = folder/f"{result['name']}.log"
            shared = not log.exists()
            if shared and fallback_log:
                log = Path(fallback_log)
            attachments = ([] if delivered(transfers) else [snapshot]) + ([log] if log.is_file() else [])
            body = (f"Training: {result['name']}\nStatus: {result['status']}\n"
                    f"Model: {result['model']}; seed: {result['seed']}\n"
                    f"Results: {folder}\n"
                    + ('Legacy attempt: only shared fold log available.\n' if shared else 'Individual training console log attached.\n')
                    + str(result.get('error', '')))
            ok = TaskResultMail(destination)(f"Factor training [{result['status']}] {result['name']}",
                body + notice(transfers), recipient=recipient, attachments=attachments)
            state = {'status': 'sent' if ok else 'failed'}
    except Exception:
        state = {'status': 'failed', 'error': traceback.format_exc()}
    state['receipt'] = str(destination/'delivery.json')
    try:
        write_json(destination/'status.json', state)
    except OSError:
        state={'status':'failed','error':traceback.format_exc(),'receipt':state['receipt']}
    return state
