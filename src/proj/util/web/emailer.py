"""SMTP email sending using credentials from machine secrets and ``Proj`` attachments."""
from __future__ import annotations

import json , os , shutil , ssl , smtplib , time , uuid
from collections.abc import Iterator
from contextlib import contextmanager , suppress
from datetime import datetime , timezone
from email import encoders , message_from_bytes
from email.message import Message
from email.errors import MessageError
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

def _is_transient(exc : BaseException) -> bool:
    """Return whether *exc* is a connection failure that should stop the drain.

    ``SMTPException`` subclasses ``OSError``. Authentication and recipient
    refusals are therefore excluded before the generic socket check.
    """
    if isinstance(exc , (smtplib.SMTPServerDisconnected , smtplib.SMTPConnectError)):
        return True
    if isinstance(exc , smtplib.SMTPException):
        return False
    return isinstance(exc , (TimeoutError , ConnectionError , OSError , ssl.SSLError))

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
                attachments : AttachmentSpec | list[AttachmentSpec] | None = None ,
                project_attachments : bool = False ,
                title_prefix : str | None = f'Learndl [{MACHINE.nickname}]:') -> Message:
        message = MIMEMultipart()
        message['From'] = cls.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}' if title_prefix else title

        if attachments is None:
            attachment_items : list[AttachmentSpec] = []
        elif isinstance(attachments , list):
            attachment_items = list(attachments)
        else:
            attachment_items = [attachments]

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
    def _deliver(cls , message : Message , recipient : str | None = None , timeout : int = 20) -> None:
        """Hand *message* to SMTP; propagate submission errors, not cleanup errors."""
        smtp_port = cls._resolved_smtp_port()
        assert smtp_port in [465, 587, 25] , f'smtp_port must be 465, 587, or 25, got {smtp_port}'
        default_context = ssl.create_default_context()
        if smtp_port == 465:
            smtp_server = smtplib.SMTP_SSL(cls.smtp_server, smtp_port, context=default_context, timeout=timeout)
        else:
            smtp_server = smtplib.SMTP(cls.smtp_server, smtp_port, timeout=timeout)
        try:
            if smtp_port != 465:
                smtp_server.starttls(context = default_context)
            smtp_server.login(cls.sender , cls.password)
            smtp_server.send_message(message , from_addr = cls.sender , to_addrs = cls.recipient(recipient))
        finally:
            # QUIT/close failures must neither hide the submission error nor
            # turn an accepted message into a retry (and a duplicate).
            try:
                with suppress(OSError):
                    smtp_server.quit()
            finally:
                with suppress(OSError):
                    smtp_server.close()

    @classmethod
    def send_with_smtplib(cls , message , recipient : str | None = None , confirmation_message : str | None = None ,
                          timeout : int = 20) -> bool:
        try:
            cls._deliver(message , recipient , timeout)
        except Exception as e:
            cls.logger.error(f'Error : sending email went wrong: {e}')
            return False
        if confirmation_message:
            cls.logger.success(f'Send email {confirmation_message}')
        return True

    @classmethod
    @contextmanager
    def _outbox_lock(cls) -> Iterator[bool]:
        """Yield whether this process holds the outbox lock.

        The wait is bounded so a busy flush cannot stall the caller indefinitely.
        Existing mail is drained only while the lock is held. New entries use
        unique staging directories and can be published without this lock.
        """
        root = PATH.email_outbox
        root.mkdir(parents = True , exist_ok = True)
        handle = open(root / '.lock' , 'a+')
        acquired = False
        deadline = time.monotonic() + _OUTBOX_LOCK_TIMEOUT
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
        message = message_from_bytes((folder / 'message.eml').read_bytes())
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
            }
            cls._write_meta(staging , meta)
            _fsync_dir(staging)
            dest = root / name
            os.replace(staging , dest)
        except Exception:
            shutil.rmtree(staging , ignore_errors = True)
            raise
        _fsync_dir(root)
        cls.logger.error(f'Email queued for later delivery: {dest.name}: {last_error}')
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
    def _flush_outbox(cls , timeout : int = 20) -> str | None:
        """Retry queued mail in order.

        Returns:
            The connection error text when the network is still down.
            ``None`` when the queue is empty, drained, or only produced SMTP rejections.
        """
        for folder in cls._pending_outbox():
            try:
                message , meta = cls._read_outbox(folder)
            except (OSError , ValueError , MessageError) as exc:
                cls.logger.error(f'Invalid email outbox entry {folder.name}: {exc}')
                try:
                    cls._move_dead(folder)
                except OSError as move_error:
                    cls.logger.error(f'Cannot isolate email outbox entry {folder.name}: {move_error}')
                continue
            try:
                cls._deliver(message , recipient = str(message['To']) if message['To'] else None , timeout = timeout)
            except Exception as exc:
                if _is_transient(exc):
                    meta['last_error'] = str(exc)
                    cls._write_meta(folder , meta)
                    cls.logger.error(f'Email outbox still unreachable, keep {folder.name}: {exc}')
                    return str(exc)
                if isinstance(exc , smtplib.SMTPException):
                    attempts = meta.get('attempts' , 0) + 1
                    meta['attempts'] = attempts
                    meta['last_error'] = str(exc)
                    if attempts >= _DEAD_LETTER_ATTEMPTS:
                        cls._move_dead(folder , meta)
                        cls.logger.error(f'Email moved to dead letter after {attempts} rejections: {folder.name}: {exc}')
                    else:
                        cls._write_meta(folder , meta)
                        cls.logger.error(f'Email rejected, will retry ({attempts}/{_DEAD_LETTER_ATTEMPTS}): {exc}')
                    continue
                raise
            confirmation = str(meta.get('confirmation_message') or '')
            if confirmation:
                cls.logger.success(f'Send email {confirmation}')
            shutil.rmtree(folder)
        return None

    @classmethod
    def send(cls , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , * , 
             attachments : AttachmentSpec | list[AttachmentSpec] | None = None ,
             project_attachments : bool = False ,
             title_prefix : str | None = f'Learndl [{MACHINE.nickname}]:' ,
             confirmation_message : str = '' ,
             queue_on_failure : bool = True) -> bool:
        """Send one email and retry mail saved from earlier SMTP failures.

        The return value is whether this call's message reached SMTP.
        A queued message returns ``False`` and is retried by a later ``send``.
        Callers with their own persistent retry queue must pass
        ``queue_on_failure=False`` so only one queue owns each notification.
        Identical content is not deduplicated: separate calls may be intentional.
        """
        if not MACHINE.emailable:
            cls.logger.alert1(f'{MACHINE.name} is not available for email, skip sending email')
            return False

        with cls._outbox_lock() as locked:
            if locked:
                blocked = cls._flush_outbox()
            else:
                blocked = None
                cls.logger.alert1('Email outbox lock busy, skip retry and send the current message only')

            message = cls.message(
                title , body , recipient ,
                attachments = attachments ,
                project_attachments = project_attachments ,
                title_prefix = title_prefix ,
            )
            if blocked is not None:
                cls.logger.error(f'Error : sending email deferred, smtp unreachable: {blocked}')
                if queue_on_failure:
                    cls._enqueue(message , confirmation_message , blocked)
                return False
            try:
                cls._deliver(message , recipient)
            except Exception as exc:
                cls.logger.error(f'Error : sending email went wrong: {exc}')
                if _is_transient(exc) or isinstance(exc , smtplib.SMTPException):
                    if queue_on_failure:
                        attempts = 0 if _is_transient(exc) else 1
                        cls._enqueue(message , confirmation_message , str(exc) , attempts = attempts)
                    return False
                raise
            if confirmation_message:
                cls.logger.success(f'Send email {confirmation_message}')
            return True

    @classmethod
    def print_info(cls , server : ServerType = 'netease'):
        infos = {'server' : server , 'smtp_server' : cls.smtp_server , 'smtp_port' : cls.smtp_port , 'sender' : cls.sender , 'password' : cls.password}
        cls.logger.stdout_pairs(infos , title = 'Email Settings:')
