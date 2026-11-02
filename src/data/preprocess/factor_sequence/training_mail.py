"""One durable delivery per training attempt, including failures; no SMTP in offline tests."""
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


def send_training_result(folder, result, recipient=None, fallback_log=None, delivery_root=None):
    """Mail errors are recorded, never replace a training exception or abort later models."""
    from src.proj import MACHINE
    from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
    folder = Path(folder)
    destination = Path(delivery_root or folder/'training_mail') / result['name']
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if not MACHINE.emailable:
            state = {'status': 'skipped', 'reason': 'Email disabled on this machine'}
        elif (destination/'delivery.json').exists():
            state = {'status': 'sent' if retry_delivery(destination) else 'failed'}
        else:
            snapshot = destination/'training_result.json'
            snapshot.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            log = folder/f"{result['name']}.log"
            shared = not log.exists()
            if shared and fallback_log:
                log = Path(fallback_log)
            attachments = [snapshot] + ([log] if log.is_file() else [])
            body = (f"Training: {result['name']}\nStatus: {result['status']}\n"
                    f"Model: {result['model']}; seed: {result['seed']}\n"
                    f"Results: {folder}\n"
                    + ('Legacy attempt: only shared fold log available.\n' if shared else 'Individual training console log attached.\n')
                    + str(result.get('error', '')))
            ok = TaskResultMail(destination)(f"Factor training [{result['status']}] {result['name']}",
                body, recipient=recipient, attachments=attachments)
            state = {'status': 'sent' if ok else 'failed'}
    except Exception:
        state = {'status': 'failed', 'error': traceback.format_exc()}
    state['receipt'] = str(destination/'delivery.json')
    try:
        write_json(destination/'status.json', state)
    except OSError:
        state={'status':'failed','error':traceback.format_exc(),'receipt':state['receipt']}
    return state
