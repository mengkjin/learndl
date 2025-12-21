import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Literal

from src.proj.env import MACHINE , ProjStates
from .logger import Logger

class _EmailSettings:
    Settings : dict[str , dict] = MACHINE.local_settings('email')

    def __init__(self , server : Literal['netease'] = 'netease' , *args , **kwargs):
        self.server = server
        self.email_conf = self.Settings[server]
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
    """
    Email class for sending email with attachment
    example:
        from src.proj import Email , ProjStates
        ProjStates.email_attachments.append('path/to/attachment.txt')
        Email.send(title = 'Test Email' , body = 'This is a test email' , recipient = 'test@example.com' , send_attachments = True , additional_attachments = ['path/to/additional.txt'])
    """
    _instance = None
    _settings = _EmailSettings()

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def setup_settings(cls , server : Literal['netease'] = 'netease'):
        if cls._settings.server != server:
            cls._settings = _EmailSettings(server)
        return cls._settings

    @classmethod
    def recipient(cls , recipient : str | None = None):
        if recipient is None: 
            recipient = str(cls._settings.sender)
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'
        return recipient
    
    @classmethod
    def message(cls , title : str  , body : str | None = None , recipient : str | None = None , 
                send_attachments : bool = True ,
                additional_attachments : str | Path | list[str | Path] | None = None ,
                title_prefix : str | None = 'Learndl:'):
        message = MIMEMultipart()
        message['From'] = cls._settings.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}'
        message.attach(MIMEText(body if body is not None else '', 'plain', 'utf-8'))

        attachments : list[Path] = []
        if send_attachments:
            attachments.extend([Path(f) for f in ProjStates.email_attachments])
            ProjStates.email_attachments.clear()
        if additional_attachments:
            if isinstance(additional_attachments , list):
                attachments.extend([Path(f) for f in additional_attachments])
            else:
                attachments.append(Path(additional_attachments))
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
    def connection(cls):
        return smtplib.SMTP(cls._settings.smtp_server, cls._settings.smtp_port)

    @classmethod
    def send(cls , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , 
             send_attachments : bool = True ,
             additional_attachments : str | Path | list[str | Path] | None = None ,
             title_prefix : str | None = 'Learndl:' ,
             server : Literal['netease'] = 'netease' , 
             confirmation_message = ''):
        
        if not MACHINE.server:
            Logger.warn('not in my server , skip sending email')
            return

        cls.setup_settings(server)
        message = cls.message(title , body , recipient , send_attachments , additional_attachments , title_prefix)

        try:
            with cls.connection() as smtp:
                smtp.starttls()
                smtp.login(cls._settings.sender, cls._settings.password)
                smtp.sendmail(cls._settings.sender, cls.recipient(recipient), message.as_string())
            Logger.success(f'sending email success {confirmation_message}')
        except Exception as e:
            Logger.fail(f'sending email went wrong: {e}')