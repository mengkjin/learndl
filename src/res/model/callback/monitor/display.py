"""
Callback to display information of trainer
- CallbackTimer : time cost of callback hooks
- StatusLogger : display epoch and event information
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from src.proj import Base , Proj
from src.res.model.util import BaseCallBack , BatchInputLoader

__all__ = ['CallbackTimer' , 'StatusLogger']

class CallbackTimer(BaseCallBack):
    """Time Cost of Callback Hooks"""
    CB_ORDER = 1000
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.recording = Proj.vb.is_max_level
        self.record_hook_durations : dict[str,list[float]]  = {hook:[] for hook in self.implemented_hooks}
        self.record_start_time : dict[str,datetime] = {}
    def __bool__(self):
        """disable callback timer"""
        return self.recording
    @cached_property
    def implemented_hooks(self):
        return self.all_hooks
    def at_enter(self , hook_name , *args , **kwargs):
        super().at_enter(hook_name , *args , **kwargs)
        if self.recording and hook_name != 'on_summarize_model': 
            self.record_start_time[hook_name] = datetime.now()
    def at_exit(self, hook_name , *args , **kwargs):
        if self.recording and hook_name != 'on_summarize_model': 
            self.record_hook_durations[hook_name].append((datetime.now() - self.record_start_time[hook_name]).total_seconds())
        super().at_exit(hook_name , *args , **kwargs)
    def on_summarize_model(self):
        if self.recording: 
            columns = ['hook_name' , 'num_calls', 'total_time' , 'avg_time']
            values  = [[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self.record_hook_durations.items() if v]
            df = pd.DataFrame(values , columns = columns).sort_values(by=['total_time'],ascending=False).head(10)
            if not df.empty:
                self.logger.display(df , title = 'Table: Callback Time Costs:')  
            
class StatusLogger(BaseCallBack):
    """Display Epoch and Event Information"""
    CB_ORDER : int = 100

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.show_info_step = [0,10,5,3,2,1][min(Proj.vb.vb // 2 , 5)]
        self.dataloader_info = Proj.vb.is_max_level
        
    @contextmanager
    def log_phase(self, phase):
        from src.res.model.util.model_identity import build_identity

        trainer = self.trainer
        if trainer.training_run is None:
            yield
            return
        self.log_path = self.config.base_path('logs', f'{phase}.md')
        self.log_available = True
        build = build_identity(self.config.base_path.base)['name']
        run = trainer.training_run.run_id
        operation = getattr(trainer, 'run_operation', 'internal')
        at = datetime.now(timezone.utc).isoformat()
        self._append(f'\n## {at} · {operation} · {build} · {run}\n\n')
        trainer.status_log = self
        try:
            yield
        except BaseException as exc:
            self.write(f'{phase} failed ({type(exc).__name__}): {exc}')
            raise
        else:
            self.write(f'{phase} completed')
        finally:
            trainer.status_log = None

    def _append(self, text):
        if not self.log_available:
            return
        try:
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(text)
        except OSError as exc:
            self.logger.warning(f'Model phase log write failed: {exc}', vb_level=0)
            self.log_available = False

    def write(self, message):
        self._append('- ' + ' '.join(str(message).splitlines()) + '\n')

    @property
    def dataloader(self) -> BatchInputLoader | Any : 
        return self.trainer.dataloader

    @property
    def display_progress(self) -> bool:
        return (
            (self.status.epoch >= 0) and 
            (self.show_info_step > 0) and 
            ((self.status.epoch - self.status.milestone_epoch) % self.show_info_step == 0)
        )
        
    def on_validation_epoch_end(self):
        if self.config.module_type == 'boost' and getattr(self.trainer, 'status_log', None) is self:
            self.write(f'{self.texts.model_key} · {self.texts.progress}')
        if not self.display_progress:
            return
        if Proj.vb.is_max_level or self.status.total_models <= self.config.model_num:
            self.logger.stdout(self.texts.progress , add_prefix = False)
        else:
            self.logger.log_only(self.texts.progress)

    def on_fit_model_start(self):
        self.model.logger.note(f'model {self.texts.model_str} fit start' , vb_level = 'max')
    
    def on_fit_epoch_end(self):
        in_log = getattr(self.trainer, 'status_log', None) is self
        if in_log:
            self.write(f'{self.texts.model_key} · {self.status.epoch_key} · {self.texts.progress}')
        for event in self.status.current.events:
            if in_log:
                self.write(f'{self.status.epoch_key} · {event.info}')
            if Proj.verbose(event.vb_level) and (Proj.vb.is_max_level or self.status.total_models <= self.config.model_num):
                self.logger.stdout(f'Epoch Event : {event.info}' , color = 'cyan' , add_prefix = False)
            else:
                self.logger.log_only(f'Epoch Event : {event.info}')
    
    def on_fit_model_end(self):
        self.model.logger.note(f'model {self.texts.model_str} fit done' , vb_level = 'max')
        model_summary = self.texts.model_summary
        if model_summary:
            if getattr(self.trainer, 'status_log', None) is self:
                self.write(model_summary)
            self.logger.remark(model_summary , add_prefix = False , vb_level = 2)

    def on_fit_end_after(self):  
        if self.texts.fit_summary:
            if getattr(self.trainer, 'status_log', None) is self:
                self.write(self.texts.fit_summary)
            self.logger.note(f'In Stage [{self.status.stage}], Finish All Process! {self.texts.fit_summary}')

    def on_test_model_start(self):
        if getattr(self.trainer, 'status_log', None) is self:
            self.test_dates = set()
            dates = self.trainer.data.model_test_dates
            span = f'{dates[0]}–{dates[-1]} · {len(dates)} dates' if len(dates) else 'no new dates'
            self.write(f'{self.texts.model_str} test start · {span}')
        self.model.logger.note(f'model {self.texts.model_str} test start' , vb_level = 'max')

    def on_test_model_end(self):
        if getattr(self.trainer, 'status_log', None) is self:
            dates = self.test_dates
            span = f'{min(dates)}–{max(dates)} · {len(dates)} dates' if dates else 'no new dates'
            self.write(f'{self.texts.model_str} test done · {span}')
        self.model.logger.note(f'model {self.texts.model_str} test done' , vb_level = 'max')
    
    def on_test_end_before(self): 
        time_cost = datetime.now() - self.status.times['test_start']
        self.logger.stdout(f'In Stage [{self.status.stage}], Finish Iterating Test Batches! Cost {Base.Elapsed(time_cost)}')
        self.test_end_start = datetime.now()

    def on_test_end_after(self): 
        time_cost = datetime.now() - self.test_end_start
        self.logger.note(f'In Stage [{self.status.stage}], Finish Testing Callbacks! Cost {Base.Elapsed(time_cost)}')

    def on_train_batch_end(self):  
        if self.dataloader_info: 
            self.dataloader.display(f'Train {self.status.epoch_key} loss : {self.metrics.batch_loss:.5f}')
            # self.device.status()
    def on_train_batch_start(self):
        if self.dataloader_info: 
            self.dataloader.display(f'Train {self.status.epoch_key} loss : {self.metrics.batch_loss:.5f}')

    def on_validation_batch_end(self):   
        if self.dataloader_info: 
            self.dataloader.display(f'Valid {self.status.epoch_key} accuracy : {self.metrics.batch_accuracy:.5f}')

    def on_before_dump_model(self):
        if not self.display_progress:
            return
        dump_info = f'Dump model {self.texts.model_str} with best attempt {self.metrics.model_metrics.best_attempt()}'
        if Proj.vb.is_max_level or self.status.total_models <= self.config.model_num:
            self.logger.stdout(dump_info , color = 'cyan')
        else:
            self.logger.log_only(dump_info)

    def on_test_batch_end(self):
        if getattr(self.trainer, 'status_log', None) is self and not self.trainer.batch_output.empty:
            self.test_dates.add(int(self.trainer.batch_dates[self.batch_idx]))
        if self.dataloader_info: 
            self.dataloader.display(f'Test {self.status.model_submodel} {self.trainer.batch_dates[self.batch_idx]}')