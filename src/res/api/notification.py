import pandas as pd
import os
import torch
from datetime import datetime
from src.basic import send_email , PATH , CALENDAR , MACHINE , TaskRecorder

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

def check_cuda_status():
    if not MACHINE.server or torch.cuda.is_available(): return
    
    title = f'Learndl: Server CUDA Failed'
    body = f"""Server {MACHINE.name} CUDA Failed , please check the cuda status, possible solution: 
    sudo apt purge nvidia-*
    sudo apt install nvidia-driver-535
    sudo reboot

    lsmod | grep nvidia
    nvidia-smi
    """
    recipient = 'mengkjin@163.com'
    
    try:
        send_email(title , body , recipient , confirmation_message='CUDA Status')
    except:
        pass


def email_to_fanghan(test = False):
    today = CALENDAR.updated()
    task_recorder = TaskRecorder('notification' , 'email_to_fanghan' , str(today))
    if task_recorder.is_finished():
        print(f'email_to_fanghan at {today} is already done')
        return
    
    pred_dates = PATH.pred_dates('gru_day_V1')
    use_date = pred_dates[pred_dates <= today].max()

    title = f'{today} 因子数据更新'
    body = '因子数据更新完成'
    recipient = 'mengkjin@163.com' if test else 'Gladiator9907@hotmail.com'
    
    attachments = f'gru_score_{datetime.now().strftime("%Y%m%d")}.csv'
    with TempFile(attachments) as temp_file:
        df1 = PATH.pred_load('gru_day_V1' , use_date)

        from src.res.trading.util import CompositeAlpha
        df2 = CompositeAlpha('use_daily' , [
            'sellside@huatai.master_combined@master_combined' ,
            'sellside@dongfang.scores_v0@avg' ,
            'gru_day_V1'
        ]).get(use_date).item().to_dataframe().rename(columns={'alpha' : 'use_daily'})
        df = pd.merge(df1 , df2 , on='secid' , how='left')
        df.to_csv(temp_file)
        try:
            send_email(title , body , recipient , attachments , confirmation_message='Fanghan')
            task_recorder.mark_finished(success = True)
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
        check_cuda_status()
        email_to_fanghan()


    
        
        