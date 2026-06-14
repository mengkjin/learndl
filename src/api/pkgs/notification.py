"""
API for notification operations of this project.
"""

from __future__ import annotations
import pandas as pd
import torch , os
from datetime import datetime
from src.proj import MACHINE , Logger , CALENDAR , DB , Options
from src.proj.util.web.emailer import Email
from src.proj.util.script import TaskRecorder
from src.proj.util.filesys.ttl_cache import DiskTTLCache
from src.api.util import wrap_update

__all__ = ['NotificationAPI']

class TempFile:
    """Context manager that deletes a filesystem path on exit."""

    def __init__(self, file_name: str):
        """Store path to remove in ``__exit__``."""
        self.file_name = file_name

    def __enter__(self):
        """Return the temp path string."""
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Best-effort ``os.remove`` with error logging."""
        try:
            os.remove(self.file_name)
        except Exception as e:
            Logger.error(f'Failed to remove temp file: {e}')

class NotificationAPI:
    @classmethod
    def update(cls):
        """
        Run the wrapped notification pipeline (CUDA check, email, cache reset).

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
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
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: low
        """
        cls.check_cuda_status()
        cls.email_to_fanghan()
        cls.reset_options_cache()

    @classmethod
    def check_cuda_status(cls):
        """
        Log CUDA availability on platform servers and optionally email diagnostics on failure.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 1
          lock_timeout: 1
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

    @classmethod
    def email_to_fanghan(cls , test : bool = False):
        """
        Send the daily factor-update email with score attachment when not already recorded.

        Args:
          test: When true, send to the developer mailbox instead of production recipient.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [windows, macos]
          execution_time: short
          memory_usage: low
        """
        if not MACHINE.platform_server:
            Logger.skipping(f'{MACHINE.name} is not platform server, skip emailing to fanghan')
            return
        today = CALENDAR.updated()
        task_recorder = TaskRecorder('notification' , 'email_to_fanghan' , str(today))
        record_entry = DiskTTLCache.get('daily_update', 'email_to_fanghan')
        if task_recorder.is_finished() or record_entry.valid_value:
            Logger.skipping(f'email_to_fanghan at {today} already done {record_entry.time_str} ...')
            return
        
        pred_dates = DB.dates('pred' , 'gru_day_V1')
        use_date = pred_dates.slice(end = today).max

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
                record_entry.put(True , ttl_hours = 24)
            except Exception as e:
                Logger.error(f'Failed to send email to fanghan: {e}')
                Logger.print_exc(e)
        Logger.success(f'Email to Fanghan at {today}')
        return

    @classmethod
    def reset_options_cache(cls):
        """
        Clear the in-process ``Options`` cache after notification work completes.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        """
        Options.cache.clear()
        Logger.success(f'Reset Options Cache')