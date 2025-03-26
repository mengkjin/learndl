import pandas as pd
import os

from datetime import datetime
from src.basic import send_email , PATH , CALENDAR , MACHINE

class TempFile:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def __enter__(self):
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            os.remove(self.file_name)
        except:
            pass


def email_to_fanghan(test = False):
    today = CALENDAR.updated()
    pred_dates = PATH.pred_dates('gru_day_V1')
    use_date = pred_dates[pred_dates <= today].max()

    title = f'{today} 因子数据更新'
    body = '因子数据更新完成'
    recipient = 'mengkjin@163.com' if test else 'Gladiator9907@hotmail.com'
    
    attachments = f'gru_score_{datetime.now().strftime("%Y%m%d")}.csv'
    with TempFile(attachments) as temp_file:
        df = PATH.pred_load('gru_day_V1' , use_date)
        df.to_csv(temp_file)
        try:
            send_email(title , body , attachments , recipient , confirmation_message='Fanghan')
        except:
            print(f'发送邮件给方晗失败!')

class NotificationAPI:
    @classmethod
    def proceed(cls):
        cls.send_email()

    @staticmethod
    def send_email():
        if not MACHINE.server:
            print('not in my server , skip sending email')
            return
        email_to_fanghan()


    
        
        