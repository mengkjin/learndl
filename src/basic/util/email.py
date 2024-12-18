import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Literal

from src.project_setting import MY_SERVER
from src.basic import conf as CONF

def send_email(title : str  , 
               body : str = 'This is test! Hello, World!' ,
               attachment : str | Path | None = None,
               recipient : str | None = None,
               server : Literal['netease'] = 'netease'):
    if not MY_SERVER:
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

    # 添加附件
    if attachment:
        attachment = Path(attachment)
        with open(attachment, 'rb') as attach_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attach_file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={attachment.name}')
            message.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as smtp_connection:
            smtp_connection.starttls()
            smtp_connection.login(sender, password)
            smtp_connection.sendmail(sender, recipient, message.as_string())
        print('sending email success')
    except Exception as e:
        print('sending email went wrong:', e)