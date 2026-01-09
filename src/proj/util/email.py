import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Literal , Sequence

from src.proj.env import MACHINE
from src.proj.proj import Proj
from src.proj.log import Logger

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
        from src.proj import Email , Proj
        Proj.States.email_attachments.append('path/to/attachment.txt')
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
    def message(cls , title : str  , body : str | None = None , recipient : str | None = None , * ,
                attachments : str | Path | Sequence[str | Path] | None = None ,
                project_attachments : bool = False ,
                title_prefix : str | None = 'Learndl:'):
        message = MIMEMultipart()
        message['From'] = cls._settings.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}'
        
        
        if attachments is None:
            attachment_paths : list[Path] = []
        elif isinstance(attachments , Sequence):
            attachment_paths = [Path(f) for f in attachments]
        else:
            attachment_paths = [Path(attachments)]
            
        if project_attachments:
            attachment_paths.extend(Proj.email_attachments.pop_all())

        body_text = body if body is not None else ''

        for attachment in attachment_paths:
            if not attachment.exists():
                body_text += f'\nAttachment not found: {attachment}'
            else:
                with open(attachment, 'rb') as attach_file:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attach_file.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={attachment.name}')
                    message.attach(part)
        message.attach(MIMEText(body if body is not None else '', 'plain', 'utf-8'))        
        return message
    
    @classmethod
    def connection(cls):
        return smtplib.SMTP(cls._settings.smtp_server, cls._settings.smtp_port)

    @classmethod
    def send(cls , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , * , 
             attachments : str | Path | Sequence[str | Path] | None = None ,
             project_attachments : bool = False ,
             title_prefix : str | None = 'Learndl:' ,
             server : Literal['netease'] = 'netease' , 
             confirmation_message = ''):
        
        if not MACHINE.server:
            Logger.alert1(f'{MACHINE.name} is not a server, skip sending email')
            return

        cls.setup_settings(server)
        message = cls.message(title , body , recipient , attachments = attachments , project_attachments = project_attachments , title_prefix = title_prefix)

        try:
            with cls.connection() as smtp:
                smtp.starttls()
                smtp.login(cls._settings.sender, cls._settings.password)
                smtp.sendmail(cls._settings.sender, cls.recipient(recipient), message.as_string())
            if confirmation_message:
                Logger.success(f'Send email {confirmation_message}')
        except Exception as e:
            Logger.error(f'Error : sending email went wrong: {e}')