import pandas as pd
import os
import torch
from datetime import datetime
from src.proj import MACHINE , Logger , CALENDAR , DB
from src.proj.util import Options , Email
from src.proj.util import TaskRecorder

from .util import wrap_update

class TempFile:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def __enter__(self):
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            os.remove(self.file_name)
        except Exception as e:
            Logger.error(f'Failed to remove temp file: {e}')

def check_cuda_status():
    if not MACHINE.server:
        Logger.skipping(f'{MACHINE.name} is not a server, skip checking cuda status')
    elif torch.cuda.is_available(): 
        Logger.success(f'Server {MACHINE.name} CUDA is available')
    else:
        Logger.error(f'Server {MACHINE.name} CUDA Failed , please check the cuda status, possible solution:')
    
        title = f'Server CUDA Failed'
        body = f"""Server {MACHINE.name} CUDA Failed , please check the cuda status, possible solution: 
        sudo apt purge nvidia-*
        sudo apt install nvidia-driver-535
        sudo reboot

        lsmod | grep nvidia
        nvidia-smi
        """
        recipient = 'mengkjin@163.com'
        
        try:
            Email.send(title , body , recipient , confirmation_message='CUDA Status')
        except Exception as e:
            Logger.error(f'Failed to send email to notify CUDA status: {e}')


def email_to_fanghan(test = False):
    if not MACHINE.server:
        Logger.skipping(f'{MACHINE.name} is not a server, skip emailing to fanghan')
        return
    today = CALENDAR.updated()
    task_recorder = TaskRecorder('notification' , 'email_to_fanghan' , str(today))
    if task_recorder.is_finished():
        Logger.skipping(f'email_to_fanghan at {today} already done')
        return
    
    pred_dates = DB.dates('pred' , 'gru_day_V1')
    use_date = pred_dates[pred_dates <= today].max()

    title = f'{today} 因子数据更新'
    body = '因子数据更新完成'
    recipient = 'mengkjin@163.com' if test else 'Gladiator9907@hotmail.com'
    
    attachments = f'gru_score_{datetime.now().strftime("%Y%m%d")}.csv'
    with TempFile(attachments) as temp_file:
        df1 = DB.load('pred' , 'gru_day_V1' , use_date)

        from src.res.trading.util import CompositeAlpha
        df2 = CompositeAlpha('use_daily' , [
            'sellside@huatai.master_combined@master_combined' ,
            'sellside@dongfang.scores_v0@avg' ,
            'gru_day_V1'
        ]).get(use_date).item().to_dataframe().rename(columns={'alpha' : 'use_daily'})
        df = pd.merge(df1 , df2 , on='secid' , how='left')
        df.to_csv(temp_file)
        try:
            Email.send(title , body , recipient , attachments = attachments)
            task_recorder.mark_finished()
        except Exception as e:
            Logger.error(f'Failed to send email to fanghan: {e}')
    Logger.success(f'Email to Fanghan at {today}')
    return

def reset_options_cache():
    Options.cache.clear()
    Logger.success(f'Reset Options Cache')

class NotificationAPI:
    @classmethod
    def update(cls):
        wrap_update(cls.process , 'Notification')

    @classmethod
    def process(cls):
        check_cuda_status()
        email_to_fanghan()
        reset_options_cache()



    
        
        