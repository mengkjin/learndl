import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any

from src.proj import Logger , Duration , Proj
from src.res.model.data_module import BatchInputLoader
from src.res.model.util import BaseCallBack

class CallbackTimer(BaseCallBack):
    '''record time cost of callback hooks'''
    WITH_CB = True
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.recording = Proj.vb.is_max_level
        self.record_hook_durations : dict[str,list[float]]  = {hook:[] for hook in self.base_hooks()}
        self.record_start_time : dict[str,datetime] = {}
    def __bool__(self):
        """disable callback timer"""
        return self.recording
    def at_enter(self , hook_name , vb_level : Any = 'max'):
        super().at_enter(hook_name , vb_level)
        if self.recording: 
            self.record_start_time[hook_name] = datetime.now()
    def at_exit(self, hook_name , vb_level : Any = 'max'):
        if self.recording: 
            self.record_hook_durations[hook_name].append((datetime.now() - self.record_start_time[hook_name]).total_seconds())
        super().at_exit(hook_name , vb_level)
    def on_summarize_model(self):
        if self.recording: 
            columns = ['hook_name' , 'num_calls', 'total_time' , 'avg_time']
            values  = [[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self.record_hook_durations.items() if v]
            df = pd.DataFrame(values , columns = columns).sort_values(by=['total_time'],ascending=False).head(5)
            Logger.display(df , caption = 'Table: Callback Time Costs:')  
            
class StatusDisplay(BaseCallBack):
    '''display epoch and event information'''
    CB_ORDER : int = 100

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.show_info_step = [0,10,5,3,2,1][min(Proj.vb.vb // 2 , 5)]
        self.dataloader_info = Proj.vb.is_max_level
        
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
        if not self.display_progress:
            return
        if self.status.total_models <= self.config.model_num:
            return Logger.stdout(self.trainer.texts.progress)
        else:
            return Logger.log_only(self.trainer.texts.progress)
    
    def on_fit_epoch_end(self):
        for event in self.status.current_epoch.events: 
            if self.status.total_models <= self.config.model_num:
                return Logger.stdout(event.info , color = 'cyan' , vb_level = event.vb_level)
            else:
                return Logger.log_only(event.info , vb_level = event.vb_level)
    
    def on_fit_model_end(self):
        model_summary = self.trainer.texts.model_summary
        if model_summary:
            Logger.remark(model_summary)
        if self.trainer.writer:
            self.trainer.writer.add_text('Model Info' , model_summary)

    def on_after_fit_end(self):  
        fit_summary = self.trainer.texts.fit_summary
        if fit_summary:
            Logger.highlight(fit_summary)
    
    def on_before_test_end(self): 
        time_cost = datetime.now() - self.status.times['test_start']
        Logger.note(f'In Stage [{self.status.stage}], Finish iterating test batches! Cost {Duration(time_cost)}' , vb_level = 3)

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

    def on_test_batch_end(self):         
        if self.dataloader_info: 
            self.dataloader.display('Test {} {}'.format(self.status.model_submodel , self.trainer.batch_dates[self.batch_idx]))