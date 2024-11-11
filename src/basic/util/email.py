import smtplib
from email.mime.text import MIMEText
from email.header import Header
from typing import Literal
from .. import PATH , CONF

def send_email(title = 'This is test! Hello, World!' , 
               body = 'This is test! Hello, World!' , 
               recipient : str | None = None,
               server : Literal['netease'] = 'netease'):
    email_conf = CONF.confidential('email')[server]
    smtp_server = email_conf['smtp_server']
    smtp_port   = email_conf['smtp_port']
    sender      = email_conf['sender']  
    password    = email_conf['password']  

    message = MIMEText(body , 'plain', 'utf-8')
    message['From'] = Header('发件人昵称 <{}>'.format(sender), 'utf-8')  # type: ignore
    message['To'] = Header('收件人昵称 <{}>'.format(recipient_email), 'utf-8')  # type: ignore
    message['Subject'] = Header(title, 'utf-8')  # type: ignore

    if recipient is None:
        recipient = str(sender)
    else:
        assert recipient , 'recipient is required'
        assert '@' in recipient , f'recipient address must contain @ , got {recipient}'

    try:
        smtp_connection = smtplib.SMTP(smtp_server, smtp_port)
        smtp_connection.login(sender, password)
        smtp_connection.sendmail(sender, recipient, message.as_string())

        smtp_connection.quit()
        print('sending email success')
    except Exception as e:
        print('sending email went wrong:', e)

def email_myself(do_send = False , 
                 title : str = '' , 
                 body_content : str = '' ,
                 body_file : Literal['print_log.txt'] | None = None , 
                 server : Literal['netease'] = 'netease'):
    def wrapper(func):
        if not do_send: return func
        assert title , 'title is required'
        def inner(*args , **kwargs):
            ret = func(*args , **kwargs)
            if isinstance(ret , dict) and 'log' in ret:
                body = ret['log']
            elif body_file is not None:
                fn = str(PATH.logs.joinpath(body_file))
                with open(fn , 'r') as f:
                    body = f.read()
            else:
                body = body_content
            assert body , 'body is required , or body_file is not None'
            send_email(title , body , server = server)
            return ret
        return inner
    return wrapper
