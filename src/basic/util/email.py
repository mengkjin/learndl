import smtplib
from email.mime.text import MIMEText
from email.header import Header
from typing import Literal
from .. import PATH

def send_email(title = 'This is test! Hello, World!' , 
               body = 'This is test! Hello, World!' , 
               recipient_email = 'mengkjin@163.com'):
    smtp_server = 'smtp.163.com'
    smtp_port = 25

    confidential = PATH.read_yaml(PATH.conf.joinpath('.confidential.yaml'))
    sender_email = confidential['sender_email']  
    sender_password = confidential['sender_password']  

    message = MIMEText(body , 'plain', 'utf-8')
    message['From'] = Header('发件人昵称 <{}>'.format(sender_email), 'utf-8')  # type: ignore
    message['To'] = Header('收件人昵称 <{}>'.format(recipient_email), 'utf-8')  # type: ignore
    message['Subject'] = Header(title, 'utf-8')  # type: ignore

    try:
        smtp_connection = smtplib.SMTP(smtp_server, smtp_port)
        smtp_connection.login(sender_email, sender_password)
        smtp_connection.sendmail(sender_email, recipient_email, message.as_string())

        smtp_connection.quit()
        print('sending email success')
    except Exception as e:
        print('sending email went wrong:', e)

def email_myself(do_send = False , 
                 title = 'This is test! Hello, World!' , 
                 body_content = '' ,
                 body_file : Literal['print_log.txt'] | None = None , 
                 recipient_email = 'mengkjin@163.com'):
    def wrapper(func):
        if not do_send: return func
        def inner(*args , **kwargs):
            ret = func(*args , **kwargs)
            if body_file is not None:
                fn = str(PATH.logs.joinpath(body_file))
                with open(fn , 'r') as f:
                    body = f.read()
            else:
                body = body_content
            send_email(title , body , recipient_email)
            return ret
        return inner
    return wrapper
