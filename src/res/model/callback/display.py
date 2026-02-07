import json
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any , ClassVar

from src.proj import PATH , MACHINE , Logger , Duration , Proj
from src.res.model.data_module import BatchDataLoader
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
    RESULT_PATH = PATH.rslt_train.joinpath('model_results.json')
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
    def dataloader(self) -> BatchDataLoader | Any : return self.trainer.dataloader
    @property
    def initial_models(self) -> bool: return self.record_model_stage <= self.config.model_num
    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    @property
    def speedup2x(self) -> bool:
        try:
            return self.config.callbacks['ResetOptimizer']['speedup2x']
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
    def toc(self , key : str , avg = False): 
        tc = self.tc(key)
        if avg and self.record_model_stage * self.record_epoch_stage:
            self.record_texts[key] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.title() , tc / 3600 , tc / 60 / self.record_model_stage , tc / self.record_epoch_stage)
            Logger.highlight(self.record_texts[key])
        else:
            self.record_texts[key] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.title() , tc)

    # callbacks
    def on_summarize_model(self):
        """export test summary to json"""
        if self.status.test_summary.empty or not MACHINE.cuda_server: 
            return
        test_scores = {
            '{}.{}'.format(*col):'|'.join([f'{k}({self.status.test_summary[col].round(v).loc[k]})' for k,v in self.SUMMARY_NDIGITS.items() 
                                           if k in self.status.test_summary[col].index]) for col in self.status.test_summary.columns}
    
        test_name = f'{self.config.model_name}(x{len(self.config.model_num_list)})_at_{self.record_init_time.strftime("%Y%m%d%H%M%S")}'
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : self.record_init_time.strftime("%Y-%m-%d %H:%M:%S") ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : str(self.config.model_data_types) ,
            '4_label' : ','.join(self.config.model_labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_fit'   : self.record_texts.get('fit'),
            '7_test'  : self.record_texts.get('test'),
            '8_result': test_scores,
        }
        assert self.RESULT_PATH.suffix == '.json', 'RESULT_PATH must be a json file'
        with open(self.RESULT_PATH, 'a') as f:
            json.dump({test_name:result}, f, indent=4)

    def on_before_data_start(self):    
        self.tic('data')
    def on_after_data_end(self):      
        self.toc('data')
    def on_before_fit_start(self):     
        self.tic('fit')
    def on_after_fit_end(self):       
        self.toc('fit' , avg=True)
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
            self.metrics.latest['train.loss'], self.metrics.latest['train.score'] ,
            self.metrics.latest['valid.score'] , self.metrics.best_metric , 
            self.optimizer.last_lr)
        if self.status.epoch % self.show_info_step == 0: 
            self.display('{attempt} {epoch} : {status}'.format(**self.record_texts))
    
    def on_fit_epoch_end(self):
        if self.status.fit_loop_breaker: 
            self.record_texts['exit'] = self.status.fit_loop_breaker.trigger_reason
        while self.status.epoch_event: 
            self.display(self.event_sdout(self.status.epoch_event.pop()))
    
    def on_fit_model_end(self):
        train_score = self.metrics.latest.get('train.score' , 0)
        valid_score = self.metrics.latest.get('valid.score' , 0)
        best_score = self.status.best_attempt_metric if self.status.best_attempt_metric else valid_score
        self.record_texts['status'] = f'Train{train_score: .4f} Valid{valid_score: .4f} BestVal{best_score: .4f}'
        self.record_texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.tc('model') / 60 , self.tc('model') / (self.record_epoch_model + 1))
        Logger.remark('{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.record_texts) , prefix = True)
    
    def on_before_test_end(self): 
        Logger.note(f'In Stage [{self.status.stage}], Finish iterating test batches! Cost {Duration(self.tc('test'))}' , vb_level = 3)

    def on_train_batch_end(self):  
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.output.loss_item:.5f}')
            # self.device.status()
    def on_train_batch_start(self):
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.output.loss_item:.5f}')

    def on_validation_batch_end(self):   
        if self.dataloader_info: 
            self.dataloader.display(f'Valid Ep#{self.status.epoch:3d} score : {self.metrics.output.score:.5f}')

    def on_test_batch_end(self):         
        if self.dataloader_info: 
            self.dataloader.display('Test {} {} score : {:.5f}'.format(
                self.status.model_submodel , self.trainer.batch_dates[self.batch_idx] , self.metrics.output.score))

