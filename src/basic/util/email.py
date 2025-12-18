import smtplib , shutil

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Literal

from src.proj import MACHINE , PATH , Logger
class _EmailSettings:
    def __init__(
        self , 
        server : Literal['netease'] = 'netease'
    ):
        self.email_conf = MACHINE.local_settings('email')[server]
        if MACHINE.name in self.email_conf:
            self.email_conf.update(self.email_conf[MACHINE.name])

    @property
    def smtp_server(self) -> str:
        return self.email_conf['smtp_server']
    @property
    def smtp_port(self) -> int:
        return self.email_conf['smtp_port']
    @property
    def sender(self) -> str:
        return self.email_conf['sender']
    @property
    def password(self) -> str:
        return self.email_conf['password']

class Email:
    Attachments : dict[str , list[Path]] = {}
    Attachment_dir = PATH.temp.joinpath('email_attachments')
    _instance = None

    settings = _EmailSettings()

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def Attach(cls , attachment : str | Path | list[str] | list[Path] | None = None , 
               group : str = 'default' , copy = False):
        if attachment is None: 
            return
        if not isinstance(attachment , list): 
            attachment = [Path(attachment)]
        if group not in cls.Attachments:
            cls.Attachments[group] = []
        for f in attachment:
            old_path = Path(f)
            if copy:
                new_path = cls.Attachment_dir.joinpath(old_path.name)
                shutil.copy(old_path , new_path)
            else:
                new_path = old_path
            if new_path not in cls.Attachments[group]:
                cls.Attachments[group].append(new_path)

    @classmethod
    def recipient(cls , recipient : str | None = None):
        if recipient is None: 
            recipient = str(cls.settings.sender)
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'
        return recipient
    
    @classmethod
    def message(cls , title : str  , body : str | None = None , recipient : str | None = None , 
                attachment_group : str | list[str] = 'default' , 
                clear_attachments : bool = True ,
                title_prefix : str | None = 'Learndl:'):
        message = MIMEMultipart()
        message['From'] = cls.settings.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}'
        message.attach(MIMEText(body if body is not None else '', 'plain', 'utf-8'))

        if not isinstance(attachment_group , list): 
            attachment_group = [attachment_group]
        attachments = []
        for group in attachment_group:
            if group in cls.Attachments:
                attachments.extend(cls.Attachments[group])
                if clear_attachments: 
                    cls.Attachments[group] = []
        attachments = list(set(attachments))

        for attachment in attachments:
            with open(attachment, 'rb') as attach_file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attach_file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={attachment.name}')
                message.attach(part)

        return message
    
    @classmethod
    def clear_unused_attachments(cls):
        attachments = [f for group in cls.Attachments.values() for f in group]
        for f in cls.Attachment_dir.iterdir():
            if f not in attachments: 
                f.unlink()
    
    @classmethod
    def connection(cls):
        return smtplib.SMTP(cls.settings.smtp_server, cls.settings.smtp_port)

    @classmethod
    def send(cls , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , 
             confirmation_message = '' , attachment_group : str | list[str] = 'default'):
        
        if not MACHINE.server:
            Logger.warn('not in my server , skip sending email')
            return

        message = cls.message(title , body , recipient , attachment_group = attachment_group)

        try:
            with cls.connection() as smtp:
                smtp.starttls()
                smtp.login(cls.settings.sender, cls.settings.password)
                smtp.sendmail(cls.settings.sender, cls.recipient(recipient), message.as_string())
            Logger.success(f'sending email success {confirmation_message}')
        except Exception as e:
            Logger.fail(f'sending email went wrong: {e}')

def send_email(title : str  , 
               body : str = 'This is test! Hello, World!' ,
               recipient : str | None = None,
               attachments : str | Path | list[str | Path] | None = None,
               server : Literal['netease'] = 'netease' , 
               confirmation_message : str = ''):
    if not MACHINE.server:
        Logger.warn('not in my server , skip sending email')
        return
    
    settings = _EmailSettings(server)

    if recipient is None: 
        recipient = str(settings.sender)
    assert recipient , 'recipient is required'
    assert '@' in recipient , f'recipient address must contain @ , got {recipient}'

    '''
    message = MIMEText(body , 'plain', 'utf-8')
    message['From'] = settings.sender
    message['To'] = recipient 
    message['Subject'] = title 
    '''

    message = MIMEMultipart()
    message['From'] = settings.sender
    message['To'] = recipient
    message['Subject'] = title
    message.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachments:
        if not isinstance(attachments , list): 
            attachments = [attachments]
        for attachment in attachments:
            with open(attachment, 'rb') as attach_file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attach_file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={str(attachment)}')
                message.attach(part)

    try:
        with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as smtp_connection:
            smtp_connection.starttls()
            smtp_connection.login(settings.sender, settings.password)
            smtp_connection.sendmail(settings.sender, recipient, message.as_string())
        Logger.success(f'sending email success {confirmation_message}')
    except Exception as e:
        Logger.fail(f'sending email went wrong: {e}')
