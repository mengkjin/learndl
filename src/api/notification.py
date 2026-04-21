import pandas as pd
import torch
from datetime import datetime
from src.proj import MACHINE , Logger , CALENDAR , DB
from src.proj.util import Options , Email , TempFile , TaskRecorder
from .util import wrap_update

def check_cuda_status():
    """
    Log CUDA availability on platform servers and optionally email diagnostics on failure.

    [API Interaction]:
      expose: false
      roles: [developer, admin]
      risk: read_only
      lock_num: -1
      disable_platforms: [windows, macos]
      execution_time: immediate
      memory_usage: low
    """
    if not MACHINE.platform_server:
        Logger.skipping(f'{MACHINE.name} is not platform server, skip checking cuda status')
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
    """
    Send the daily factor-update email with score attachment when not already recorded.

    Args:
        test: When true, send to the developer mailbox instead of production recipient.

    [API Interaction]:
      expose: false
      roles: [admin]
      risk: write
      lock_num: -1
      disable_platforms: [windows, macos]
      execution_time: short
      memory_usage: low
    """
    if not MACHINE.platform_server:
        Logger.skipping(f'{MACHINE.name} is not platform server, skip emailing to fanghan')
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

        from src.res.factor.util import AlphaComposite
        df2 = AlphaComposite('use_daily' , [
            'factor@ht_master_combined' ,
            'factor@df_scores_v0' ,
            'gru_day_V1'
        ]).get(use_date).item().to_dataframe().rename(columns={'alpha' : 'use_daily'})
        df = pd.merge(df1 , df2 , on='secid' , how='left')
        df.to_csv(temp_file)
        try:
            Email.send(title , body , recipient , attachments = attachments)
            task_recorder.mark_finished()
        except Exception as e:
            Logger.error(f'Failed to send email to fanghan: {e}')
            Logger.print_exc(e)
    Logger.success(f'Email to Fanghan at {today}')
    return

def reset_options_cache():
    """
    Clear the in-process ``Options`` cache after notification work completes.

    [API Interaction]:
      expose: false
      roles: [developer, admin]
      risk: write
      lock_num: -1
      disable_platforms: []
      execution_time: immediate
      memory_usage: low
    """
    Options.cache.clear()
    Logger.success(f'Reset Options Cache')

class NotificationAPI:
    @classmethod
    def update(cls):
        """
        Run the wrapped notification pipeline (CUDA check, email, cache reset).

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: short
          memory_usage: low
        """
        wrap_update(cls.process , 'Notification')

    @classmethod
    def process(cls):
        """
        Execute CUDA diagnostics, Fanghan email task, and ``reset_options_cache``.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: short
          memory_usage: low
        """
        check_cuda_status()
        email_to_fanghan()
        reset_options_cache()



    
        
        