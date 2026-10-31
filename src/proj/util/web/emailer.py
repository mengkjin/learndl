"""SMTP email sending using credentials from machine secrets and ``Proj`` attachments."""
from __future__ import annotations

import json , os , shutil , ssl , smtplib , time , uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager , suppress
from datetime import datetime , timezone
from email import encoders, policy
from email.message import Message
from email.parser import BytesParser
from .mail_package import MAX_BYTES, needs_preparation, wire
from .mail_queue import MailQueue
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any , Literal , TypeAlias

import portalocker

from src.proj.env import MACHINE , PATH , Proj
from src.proj.env.variable.files import EmailAttachment
from src.proj.core import strPath
from src.proj.bases import BoundLogger , NoInstance

__all__ = ['Email' , 'EmailAttachment']

AttachmentSpec : TypeAlias = strPath | EmailAttachment

ServerType : TypeAlias = Literal['netease']
_EmailSettings : dict = {}
_DEAD_LETTER_ATTEMPTS = 5
_OUTBOX_LOCK_TIMEOUT = 5.0

def _get_email_setting(name : str , server : ServerType = 'netease') -> Any:
    if not _EmailSettings:
        email_accounts = MACHINE.secret.get('accounts' , 'email')
        server_settings = email_accounts.get(server , {})
        machine_settings = email_accounts.get(server , {}).get(MACHINE.name.lower() , {})
        settings = server_settings | machine_settings
        _EmailSettings.update(settings)
    if name == 'smtp_port':
        return 25 if MACHINE.platform_server else 465
    assert name in _EmailSettings , f'{name} is not set in email settings'
    return _EmailSettings.get(name.lower() , None)

def _atomic_write(path : Path , data : bytes) -> None:
    """Replace *path* only after the new bytes are flushed."""
    path.parent.mkdir(parents = True , exist_ok = True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp , path)
    finally:
        temp.unlink(missing_ok = True)

def _resolve_attachment(item : AttachmentSpec) -> tuple[Path , str]:
    """Return the local path and the name written on the MIME part."""
    if isinstance(item , EmailAttachment):
        return item.path , item.attachment_name
    path = Path(item)
    return path , path.name

def _fsync_dir(path : Path) -> None:
    """Persist a directory rename on POSIX. Windows commits ``os.replace`` itself."""
    if os.name != 'posix':
        return
    fd = os.open(path , os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

class Email(BoundLogger, metaclass=NoInstance):
    """
    Email class for sending email with attachment
    example:
        from src.proj.util.web.emailer import Email , EmailAttachment
        Email.send('Test Email' , 'This is a test email' , 'test@example.com' ,
                   project_attachments = True , attachments = ['path/to/additional.txt'])
        Email.send('Report' , 'body' , attachments = EmailAttachment(
            'results/detailed_alpha_data.xlsx' , filename = 'detailed_alpha_data_gru.xlsx'))
    """
    max_message_bytes: int = MAX_BYTES
    smtp_server : str = _get_email_setting('smtp_server')
    smtp_port : int | Literal['auto'] = _get_email_setting('smtp_port')
    sender : str = _get_email_setting('sender')
    password : str = _get_email_setting('password')

    @classmethod
    def recipient(cls , recipient : str | None = None) -> str:
        if recipient is None: 
            recipient = str(cls.sender)
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'
        return recipient
    
    @classmethod
    def message(cls , title : str  , body : str | None = None , recipient : str | None = None , * ,
                attachments : AttachmentSpec | Sequence[AttachmentSpec] | None = None ,
                project_attachments : bool = False ,
                title_prefix : str | None = f'Learndl [{MACHINE.nickname}]:') -> Message:
        message = MIMEMultipart()
        message['From'] = cls.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}' if title_prefix else title

        if attachments is None:
            attachment_items : list[AttachmentSpec] = []
        elif isinstance(attachments , (str , Path)):
            attachment_items = [attachments]
        elif isinstance(attachments , EmailAttachment):
            attachment_items = [attachments]
        else:
            attachment_items = list(attachments)

        if project_attachments:
            attachment_items.extend(Proj.email_attachments.pop_all())

        body_text = body if body is not None else ''
        for item in attachment_items:
            path , filename = _resolve_attachment(item)
            if not path.exists():
                body_text += f'\nAttachment not found: {path}'
            else:
                with open(path, 'rb') as attach_file:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attach_file.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition' , 'attachment' , filename = filename)
                    message.attach(part)
        message.attach(MIMEText(body_text , 'plain' , 'utf-8'))
        return message

    @classmethod
    def _resolved_smtp_port(cls) -> int:
        if cls.smtp_port == 'auto':
            return 25 if MACHINE.platform_server else 465
        return cls.smtp_port

    @classmethod
    def _queue(cls):
        return MailQueue(cls, PATH.email_outbox)

    @classmethod
    def _deliver(cls , message : Message , recipient : str | None = None , timeout : int = 20) -> None:
        """Submit one bounded MIME message, preserving the exact failure stage."""
        raw = wire(message)
        if len(raw) > cls.max_message_bytes:
            error = ValueError(f'MIME size {len(raw)} exceeds {cls.max_message_bytes}; prepare a batch first')
            setattr(error, 'mail_stage', 'prepare')
            raise error
        smtp_port = cls._resolved_smtp_port()
        if smtp_port not in (465, 587, 25):
            raise ValueError(f'Unsupported SMTP port: {smtp_port}')
        stage, server = 'connect', None
        try:
            context = ssl.create_default_context()
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(cls.smtp_server, smtp_port, context=context, timeout=timeout)
            else:
                server = smtplib.SMTP(cls.smtp_server, smtp_port, timeout=timeout)
                stage = 'tls'
                server.starttls(context=context)
            stage = 'auth'
            server.login(cls.sender, cls.password)
            stage = 'submit'
            # sendmail accepts bytes without regenerating MIME or line endings.
            refused = server.sendmail(cls.sender, cls.recipient(recipient), raw)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
        except Exception as error:
            setattr(error, 'mail_stage', stage)
            raise
        finally:
            if server is not None:
                try:
                    with suppress(OSError):
                        server.quit()
                finally:
                    with suppress(OSError):
                        server.close()

    @classmethod
    def send_with_smtplib(cls, message, recipient=None, confirmation_message=None,
                          timeout: int = 20, *, delivery_id: str | None = None) -> bool:
        """Caller-owned retry with per-part receipts; never enqueue an automatic copy.

        Supply a stable delivery_id for an external event. Large legacy calls without
        an ID use a content fingerprint; small unidentified calls remain independent.
        """
        if not MACHINE.emailable:
            return False
        if delivery_id is None and not needs_preparation(message, cls.max_message_bytes):
            delivery_id = uuid.uuid4().hex
        try:
            with cls._outbox_lock() as locked:
                if not locked:
                    return False
                sent = cls._queue().external(message, recipient, delivery_id, timeout)
        except Exception as error:
            cls.logger.error(f'Email external delivery failed: {type(error).__name__}: {error}')
            return False
        if sent and confirmation_message:
            cls.logger.success(f'Send email {confirmation_message}')
        return sent

    @classmethod
    @contextmanager
    def _outbox_lock(cls, wait_seconds: float = _OUTBOX_LOCK_TIMEOUT) -> Iterator[bool]:
        """Yield whether this process holds the outbox lock.

        The wait is bounded so a busy flush cannot stall the caller indefinitely.
        Existing mail is drained only while the lock is held. New entries use
        unique staging directories and can be published without this lock.
        """
        root = PATH.email_outbox
        root.mkdir(parents = True , exist_ok = True)
        handle = open(root / '.lock' , 'a+')
        acquired = False
        deadline = time.monotonic() + wait_seconds
        try:
            while True:
                try:
                    portalocker.lock(handle , portalocker.LOCK_EX | portalocker.LOCK_NB)
                    acquired = True
                    break
                except (portalocker.AlreadyLocked , portalocker.LockException , BlockingIOError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                portalocker.unlock(handle)
            handle.close()

    @classmethod
    def _pending_outbox(cls) -> list[Path]:
        """Return queued messages in FIFO order. Incomplete staging dirs stay hidden."""
        root = PATH.email_outbox
        if not root.is_dir():
            return []
        folders = [
            path for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith('.')
            and path.name != 'dead'
        ]
        return sorted(folders , key = lambda path : path.name)

    @classmethod
    def _read_outbox(cls , folder : Path) -> tuple[Message , dict[str , Any]]:
        with (folder / 'message.eml').open('rb') as stream:
            message = BytesParser(policy=policy.default).parse(stream)
        meta = json.loads((folder / 'meta.json').read_text(encoding = 'utf-8'))
        if not isinstance(meta , dict):
            raise ValueError('Email metadata must be an object')
        attempts = meta.get('attempts' , 0)
        if type(attempts) is not int or attempts < 0:
            raise ValueError('Email attempts must be a nonnegative integer')
        if not message['To'] or '@' not in str(message['To']):
            raise ValueError('Queued email has no valid recipient')
        if any(part.defects for part in message.walk()):
            raise ValueError('Queued email has malformed MIME content')
        return message , meta

    @classmethod
    def _write_meta(cls , folder : Path , meta : dict[str , Any]) -> None:
        _atomic_write(folder / 'meta.json' , json.dumps(meta , ensure_ascii = False , indent = 2).encode('utf-8'))

    @classmethod
    def _enqueue(cls , message : Message , confirmation_message : str , last_error : str , attempts : int = 0) -> Path:
        """Publish one assembled message. The directory appears only after both files exist."""
        root = PATH.email_outbox
        root.mkdir(parents = True , exist_ok = True)
        name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
        staging = root / f'.{name}.tmp'
        staging.mkdir(parents = True , exist_ok = False)
        try:
            _atomic_write(staging / 'message.eml' , message.as_bytes())
            meta = {
                'confirmation_message' : confirmation_message or '' ,
                'attempts' : attempts ,
                'last_error' : last_error ,
                'created_at' : datetime.now(timezone.utc).isoformat() ,
                'subject': str(message.get('Subject', '')),
                'source_bytes': (staging / 'message.eml').stat().st_size,
            }
            cls._write_meta(staging , meta)
            _fsync_dir(staging)
            dest = root / name
            os.replace(staging , dest)
        except Exception:
            shutil.rmtree(staging , ignore_errors = True)
            raise
        _fsync_dir(root)
        cls.logger.stdout(f'Email queued: {dest.name}' + (f': {last_error}' if last_error else ''))
        return dest

    @classmethod
    def _move_dead(cls , folder : Path , meta : dict[str , Any] | None = None) -> None:
        """Isolate rejected/corrupt mail, preserving corrupt metadata for inspection."""
        if meta is not None:
            cls._write_meta(folder , meta)
        dead = PATH.email_outbox / 'dead'
        dead.mkdir(parents = True , exist_ok = True)
        dest = dead / folder.name
        if dest.exists():
            dest = dead / f'{folder.name}_{uuid.uuid4().hex[:8]}'
        os.replace(folder , dest)

    @classmethod
    def _flush_outbox(cls, timeout: int = 20, *, max_messages: int | None = 20,
                      deadline: float | None = None, stats: dict[str, Any] | None = None) -> str | None:
        cls._queue().flush(timeout=timeout, max_messages=max_messages, deadline=deadline, stats=stats)
        return 'SMTP connection cooling down' if cls._queue().cooling() else None

    @classmethod
    def retry_pending(cls, *, max_messages: int = 20, max_seconds: float = 20) -> dict[str, Any]:
        """Retry published outbox entries, never competing with an active sender.

        The budget is checked between messages; an in-flight SMTP transaction is
        allowed to finish. SMTP acceptance followed by process death can still
        cause a later retry: this is not an exactly-once transport.
        """
        if max_messages <= 0 or max_seconds <= 0:
            raise ValueError('Email retry budgets must be positive')
        stats: dict[str, Any] = dict(status='idle', lock_busy=False, attempted=0,
                                    sent=0, processed=0, remaining=0, error=None)
        if not MACHINE.emailable:
            stats['status'] = 'disabled'
            return stats
        # Do not create an empty outbox just to inspect it.
        if not PATH.email_outbox.is_dir():
            return stats
        with cls._outbox_lock(wait_seconds=0) as locked:
            if not locked:
                stats.update(status='busy', lock_busy=True)
                stats['remaining'] = len(cls._pending_outbox())
                return stats
            try:
                cls._flush_outbox(max_messages=max_messages,
                                  deadline=time.monotonic() + max_seconds, stats=stats)
                stats['status'] = 'cooldown' if cls._queue().cooling() else ('error' if stats['error'] else 'complete')
                stats['next_retry_at'] = cls._queue().state().get('next_retry_at', 0)
            except Exception as exc:
                stats.update(status='error', error=f'{type(exc).__name__}: {exc}')
                cls.logger.error(f'Email outbox retry failed: {exc}')
            stats['remaining'] = len(cls._pending_outbox())
        return stats

    @classmethod
    def send(cls, title: str, body: str = 'This is test! Hello, World!', recipient: str | None = None, *,
             attachments: AttachmentSpec | Sequence[AttachmentSpec] | None = None,
             project_attachments: bool = False, title_prefix: str | None = f'Learndl [{MACHINE.nickname}]:',
             confirmation_message: str = '', queue_on_failure: bool = True,
             delivery_id: str | None = None) -> bool:
        """Prepare bounded mail; True only when all parts have SMTP acceptance.

        Automatic mail has a single durable outbox owner. External callers use
        queue_on_failure=False and a stable delivery_id to retry only unsent parts.
        """
        if not MACHINE.emailable:
            cls.logger.alert1(f'{MACHINE.name} is not available for email, skip sending email')
            return False
        message = cls.message(title, body, recipient, attachments=attachments,
                              project_attachments=project_attachments, title_prefix=title_prefix)
        if not queue_on_failure:
            return cls.send_with_smtplib(message, recipient, confirmation_message, delivery_id=delivery_id)
        # Publish before network work. A competing worker can safely deliver this entry.
        folder = cls._enqueue(message, confirmation_message, '')
        del message
        with cls._outbox_lock() as locked:
            if not locked:
                return False
            cls._flush_outbox(deadline=time.monotonic()+20)
            backup = PATH.email_outbox / '.archive' / folder.name / 'payload' / 'manifest.json'
            if backup.exists():
                return all(item['sent'] for item in json.loads(backup.read_text())['messages'])
        return False

    @classmethod
    def print_info(cls , server : ServerType = 'netease'):
        infos = {'server' : server , 'smtp_server' : cls.smtp_server , 'smtp_port' : cls.smtp_port , 'sender' : cls.sender , 'password' : cls.password}
        cls.logger.stdout_pairs(infos , title = 'Email Settings:')
