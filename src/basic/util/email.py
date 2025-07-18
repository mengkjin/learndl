import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Literal , Any

from src.project_setting import MACHINE
from src.basic import conf as CONF

class Email:
    ATTACHMENTS : list[Path] = []
    _instance = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , server : Literal['netease'] = 'netease'):
        email_conf = CONF.confidential('email')[server]
        self.smtp_server = email_conf['smtp_server']
        self.smtp_port   = email_conf['smtp_port']
        self.sender      = email_conf['sender']  
        self.password    = email_conf['password']  

    @classmethod
    def attach(cls , attachment : str | Path | list[str] | list[Path] | None = None):
        if attachment is None: return
        if not isinstance(attachment , list): attachment = [Path(attachment)]
        cls.ATTACHMENTS.extend([Path(f) for f in attachment])

    def recipient(self , recipient : str | None = None):
        if recipient is None: recipient = str(self.sender)
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'
        return recipient
    
    def message(self , title : str  , body : str | None = None , recipient : str | None = None , 
                title_prefix : str | None = 'Learndl:'):
        message = MIMEMultipart()
        message['From'] = self.sender
        message['To'] = self.recipient(recipient)
        message['Subject'] = title_prefix + title if title_prefix else title
        message.attach(MIMEText(body if body is not None else '', 'plain', 'utf-8'))

        attachments = set(self.ATTACHMENTS)
        for attachment in attachments:
            with open(attachment, 'rb') as attach_file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attach_file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={attachment.name}')
                message.attach(part)

        self.ATTACHMENTS.clear()

        return message
    
    def connection(self):
        return smtplib.SMTP(self.smtp_server, self.smtp_port)

    def send(self , title : str  , 
             body : str = 'This is test! Hello, World!' ,
             recipient : str | None = None , 
             confirmation_message = ''):
        
        if not MACHINE.server:
            print('not in my server , skip sending email')
            return

        message = self.message(title , body , recipient)

        try:
            with self.connection() as smtp:
                smtp.starttls()
                smtp.login(self.sender, self.password)
                smtp.sendmail(self.sender, self.recipient(recipient), message.as_string())
            print(f'sending email success {confirmation_message}')
        except Exception as e:
            print('sending email went wrong:', e)

def send_email(title : str  , 
               body : str = 'This is test! Hello, World!' ,
               attachments : str | Path | list[str | Path] | None = None,
               recipient : str | None = None,
               server : Literal['netease'] = 'netease' , 
               confirmation_message : str = ''):
    if not MACHINE.server:
        print('not in my server , skip sending email')
        return
    
    email_conf = CONF.confidential('email')[server]
    smtp_server = email_conf['smtp_server']
    smtp_port   = email_conf['smtp_port']
    sender      = email_conf['sender']  
    password    = email_conf['password']  

    if recipient is None: recipient = str(sender)
    assert recipient , 'recipient is required'
    assert '@' in recipient , f'recipient address must contain @ , got {recipient}'

    '''
    message = MIMEText(body , 'plain', 'utf-8')
    message['From'] = sender
    message['To'] = recipient 
    message['Subject'] = title 
    '''

    message = MIMEMultipart()
    message['From'] = sender
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
        with smtplib.SMTP(smtp_server, smtp_port) as smtp_connection:
            smtp_connection.starttls()
            smtp_connection.login(sender, password)
            smtp_connection.sendmail(sender, recipient, message.as_string())
        print(f'sending email success {confirmation_message}')
    except Exception as e:
        print('sending email went wrong:', e)
