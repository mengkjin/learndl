import smtplib , ssl

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any , Literal

from src.proj.env import MACHINE
from src.proj.proj import Proj
from src.proj.log import Logger

class EmailSetting:
    EmailSettings = MACHINE.local_settings('email')

    def __init__(self , name : str , server : Literal['netease'] = 'netease'):
        server_settings = self.EmailSettings.get(server , {})
        machine_settings = self.EmailSettings.get(server , {}).get(MACHINE.name.lower() , {})
        settings = server_settings | machine_settings

        self.name = name
        self.server = server
        self.value = settings.get(name.lower() , None)

    def __get__(self , instance : Any , owner : Any) -> Any:
        if self.name == 'smtp_port':
            return 25 if MACHINE.platform_server else 465
            raise Exception('smtp_port is not set and will use default port 465/587/25')
        assert self.value is not None , f'{self.name} is not set in {self.server} email settings'
        return self.value
        

class EmailMeta(type):
    smtp_server = EmailSetting('smtp_server')
    smtp_port = EmailSetting('smtp_port')
    sender = EmailSetting('sender')
    password = EmailSetting('password')

    def __call__(cls , *args , **kwargs):
        raise Exception('EmailMeta subclass is not meant to be instantiated')

class Email(metaclass=EmailMeta):
    """
    Email class for sending email with attachment
    example:
        from src.proj import Email , Proj
        Proj.States.email_attachments.append('path/to/attachment.txt')
        Email.send(title = 'Test Email' , body = 'This is a test email' , recipient = 'test@example.com' , send_attachments = True , additional_attachments = ['path/to/additional.txt'])
    """

    @classmethod
    def recipient(cls , recipient : str | None = None):
        if recipient is None: 
            recipient = str(cls.sender)
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'
        return recipient
    
    @classmethod
    def message(cls , title : str  , body : str | None = None , recipient : str | None = None , * ,
                attachments : str | Path | list[Path] | list[str] | None = None ,
                project_attachments : bool = False ,
                title_prefix : str | None = f'Learndl [{MACHINE.nickname}]:'):
        message = MIMEMultipart()
        message['From'] = cls.sender
        message['To'] = cls.recipient(recipient)
        message['Subject'] = f'{title_prefix} {title}'
        
        if attachments is None:
            attachment_paths : list[Path] = []
        elif isinstance(attachments , list):
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
    def send_with_smtplib(cls , message : MIMEMultipart , recipient : str | None = None , confirmation_message : str | None = None ,
                          timeout : int = 20):
        if cls.smtp_port == 'auto':
            smtp_port = 25 if MACHINE.platform_server else 465
        else:
            smtp_port = cls.smtp_port
        assert smtp_port in [465, 587, 25] , f'smtp_port must be 465, 587, or 25, got {smtp_port}'
        
        try:
            default_context = ssl.create_default_context()
            if smtp_port == 465:
                smtp_server = smtplib.SMTP_SSL(cls.smtp_server, smtp_port, context=default_context, timeout=timeout)
            else:
                smtp_server = smtplib.SMTP(cls.smtp_server, smtp_port, timeout=timeout)
            with smtp_server as server:
                if smtp_port != 465:
                    server.starttls(context=default_context)
                server.login(cls.sender, cls.password)
                server.send_message(message , from_addr=cls.sender, to_addrs=cls.recipient(recipient))
            if confirmation_message:
                Logger.success(f'Send email {confirmation_message}')
        except Exception as e:
            Logger.error(f'Error : sending email went wrong: {e}')

    @classmethod
    def send(cls , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , * , 
             attachments : str | Path | list[Path] | list[str] | None = None ,
             project_attachments : bool = False ,
             title_prefix : str | None = f'Learndl [{MACHINE.nickname}]:' ,
             confirmation_message = ''):
        
        if not MACHINE.emailable:
           Logger.alert1(f'{MACHINE.name} is not available for email, skip sending email')
           return

        message = cls.message(title , body , recipient , attachments = attachments , project_attachments = project_attachments , title_prefix = title_prefix)
        cls.send_with_smtplib(message , recipient , confirmation_message)

    @classmethod
    def print_info(cls , server : Literal['netease'] = 'netease'):
        infos = {'server' : server , 'smtp_server' : cls.smtp_server , 'smtp_port' : cls.smtp_port , 'sender' : cls.sender , 'password' : cls.password}
        Logger.stdout_pairs(infos , title = 'Email Settings:')
