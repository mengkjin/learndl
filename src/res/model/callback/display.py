import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any , ClassVar

from src.proj import Logger , LogFile , Duration , Proj
from src.res.model.data_module import BatchInputLoader
from src.res.model.util import BaseCallBack

class CallbackTimer(BaseCallBack):
    '''record time cost of callback hooks'''
    WITH_CB = True
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.recording = Proj.vb.is_max_level
        self.record_hook_durations : dict[str,list[float]]  = {hook:[] for hook in self.possible_hooks}
        self.record_start_time : dict[str,datetime] = {}
    def __bool__(self):
        """disable callback timer"""
        return False # self.recording and not self.turn_off
    def at_enter(self , hook_name , vb_level : int = Proj.vb.max):
        super().at_enter(hook_name , vb_level)
        if self.recording: 
            self.record_start_time[hook_name] = datetime.now()
    def at_exit(self, hook_name , vb_level : int = Proj.vb.max):
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
    SUMMARY_NDIGITS : ClassVar[dict[str,int]] = {'Avg':4,'Sum':2,'Std':4,'T':2,'IR':4}

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.show_info_step = [0,10,5,3,2,1][min(Proj.vb.vb // 2 , 5)]
        self.dataloader_info = Proj.vb.is_max_level
        self.record_init_time = datetime.now()
        self.record_times : dict[str,datetime] = {}
        self.record_texts : dict[str,str]   = {}
        self.record_epoch_model : int = 0
        self.record_epoch_stage : int = 0
        self.record_model_stage : int = 0

    @property
    def log_file(self):
        log_name = 'st_results' if self.config.base_path.is_short_test else 'results'
        return LogFile.initiate('model' , 'summary' , log_name)
    @property
    def dataloader(self) -> BatchInputLoader | Any : return self.trainer.dataloader
    @property
    def initial_models(self) -> bool: return self.record_model_stage <= self.config.model_num
    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    @property
    def speedup2x(self) -> bool:
        try:
            return self.config.callback_kwargs['ResetOptimizer']['speedup2x']
        except KeyError:
            return False
    @property
    def optimizer(self): 
        return getattr(self.trainer.model , 'optimizer')
    def display(self , *args , **kwargs):
        if (self.show_info_step or self.initial_models):
            return Logger.stdout(*args , **kwargs)
        else:
            return Logger.log_only(*args , **kwargs)
    def event_sdout(self , event) -> str:
        if event == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.status.epoch} , effective at epoch {self.status.epoch + 1}'
            if self.speedup2x: 
                sdout += ', and will speedup2x'
        elif event == 'new_attempt':
            sdout = '{attempt} {epoch} : {status}, Next attempt goes!'.format(**self.record_texts)
        elif event == 'nanloss':
            sdout = 'Model {model_date}.{model_num} Attempt{attempt}, epoch{epoch} got nanloss!'.format(**self.status.as_dict())
        else:
            raise KeyError(event)
        return sdout

    def tic(self , key : str): 
        self.record_times[key] = datetime.now()
    def tc(self , key : str):
        return (datetime.now() - self.record_times[key]).total_seconds()
    def toc(self , key : str): 
        tc = self.tc(key)
        if self.status.stage == 'fit' and self.record_model_stage * self.record_epoch_stage:
            per_model = tc / 60 / self.record_model_stage
            per_epoch = tc / self.record_epoch_stage
            self.record_texts[key] = f'Cost {tc / 3600:.1f} Hours, {per_model:.1f} Min/model, {per_epoch:.1f} Sec/Epoch'
            Logger.highlight(f'Finish Process [Fit], {self.record_texts[key]}')
        else:
            self.record_texts[key] = f'Cost {tc:.1f} Secs'
    
    @classmethod
    def format_messages(cls , messages : dict[str, str | dict] , indent : int = 0) -> list[str]:
        key_len = max(len(key) for key in messages.keys())
        msgs : list[str] = []
        for key , msg in messages.items():
            if isinstance(msg , dict):
                msgs.append('  ' * indent + f'{key:{key_len}s} : ')
                msgs.extend(cls.format_messages(msg , indent + 1))
            else:
                msgs.append('  ' * indent + f'{key:{key_len}s} : {msg}')
        return msgs

    # callbacks
    def on_summarize_model(self):
        """export test summary to json"""
        if self.status.test_summary.empty: 
            return
        
        # metrics = {
        #     '{}.{}'.format(*col):'|'.join([f'{k}({self.status.test_summary[col].round(v).loc[k]})' 
        #     for k,v in self.SUMMARY_NDIGITS.items() if k in self.status.test_summary[col].index]) for col in self.status.test_summary.columns}
        test_name = f'{self.config.base_path.full_name}'

        duration : dict[str,str] = {
            stage : self.record_texts.get(stage , "N/A") for stage in self.config.stage_queue
        }
        metrics : dict[str,str] = {}
        for col in self.status.test_summary.columns:
            series = self.status.test_summary[col]
            key = '{}.{}'.format(*col)
            value = '|'.join([f'{k}({series.round(v).loc[k]})' for k,v in self.SUMMARY_NDIGITS.items() if k in series.index])
            metrics[key] = value

        messages = {
            f'model' : f'{self.config.model_name} x {len(self.config.model_num_list)})',
            f'submodel' : f'{self.config.submodels}',
            f'start' : f'{self.record_init_time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'stages' : f'{self.config.stage_queue}',
            f'inputs' : self.data.input_keys_subkeys,
            f'labels' : f'{self.config.labels}',
            f'range' : f'{self.config.beg_date} - {self.config.end_date}',
            f'duration' : duration,
            f'metrics' : metrics,
        }
        msgs = self.format_messages(messages , indent = 0)
        self.log_file.write(test_name , *msgs)

    def on_before_data_start(self):    
        self.tic('data')
    def on_after_data_end(self):      
        self.toc('data')
    def on_before_fit_start(self):     
        self.tic('fit')
    def on_after_fit_end(self):       
        self.toc('fit')
    def on_before_test_start(self):
        self.tic('test')
    def on_after_test_end(self):
        self.toc('test')

    def on_fit_model_start(self):
        self.tic('model')
        self.record_model_stage += 1
        self.record_epoch_model = 0
        self.record_texts = {k:'' for k in ['model','attempt','epoch','epoch_model','status','time','exit']}
        self.record_texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.status.model_num , self.status.model_date)
    
    def on_fit_epoch_start(self):
        self.record_epoch_model += 1
        self.record_epoch_stage += 1
        self.record_texts['epoch'] = 'Ep#{:3d}'.format(self.status.epoch)
        self.record_texts['epoch_model'] = 'Ep#{:3d}'.format(self.record_epoch_model)
        self.record_texts['attempt'] = f'FirstBite' if self.status.attempt == 0 else f'Retrain#{self.status.attempt}'
    
    def on_validation_epoch_end(self):
        if self.trainer.status.epoch < 0: 
            return
        self.record_texts['status'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, best{: .4f}, lr{:.1e}'.format(
            self.metrics.attempt_metrics.latest('train' , 'loss') , 
            self.metrics.attempt_metrics.latest('train' , 'score') ,
            self.metrics.attempt_metrics.latest('valid' , 'score') , 
            self.metrics.best_epoch_metric , 
            self.optimizer.last_lr)
        if self.status.epoch % self.show_info_step == 0: 
            self.display('{attempt} {epoch} : {status}'.format(**self.record_texts))
    
    def on_fit_epoch_end(self):
        if self.status.fit_loop_breaker: 
            self.record_texts['exit'] = self.status.fit_loop_breaker.trigger_reason
        while self.status.epoch_event: 
            self.display(self.event_sdout(self.status.epoch_event.pop()))
    
    def on_fit_model_end(self):
        train_score = self.metrics.attempt_metrics.latest('train' , 'score')
        valid_score = self.metrics.attempt_metrics.latest('valid' , 'score')
        best_score = self.status.best_attempt_metric if self.status.best_attempt_metric else valid_score
        self.record_texts['status'] = f'Train{train_score: .4f} Valid{valid_score: .4f} BestVal{best_score: .4f}'
        self.record_texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.tc('model') / 60 , self.tc('model') / (self.record_epoch_model + 1))
        Logger.remark('{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.record_texts))
    
    def on_before_test_end(self): 
        Logger.note(f'In Stage [{self.status.stage}], Finish iterating test batches! Cost {Duration(self.tc('test'))}' , vb_level = 3)

    def on_train_batch_end(self):  
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.batch_loss:.5f}')
            # self.device.status()
    def on_train_batch_start(self):
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.batch_loss:.5f}')

    def on_validation_batch_end(self):   
        if self.dataloader_info: 
            self.dataloader.display(f'Valid Ep#{self.status.epoch:3d} score : {self.metrics.batch_score:.5f}')

    def on_test_batch_end(self):         
        if self.dataloader_info: 
            self.dataloader.display('Test {} {} score : {:.5f}'.format(
                self.status.model_submodel , self.trainer.batch_dates[self.batch_idx] , self.metrics.batch_score))

