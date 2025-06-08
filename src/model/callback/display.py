import time
import numpy as np
import pandas as pd

from typing import Any , ClassVar

from src import func as FUNC
from src.basic import PATH
from src.model.data_module import BatchDataLoader
from src.model.util import BaseCallBack

class CallbackTimer(BaseCallBack):
    '''record time cost of callback hooks'''
    WITH_CB = True
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info(verbosity = self.verbosity)
        self.recording = self.verbosity >= 10
        self.record_hook_times : dict[str,list]  = {hook:[] for hook in self.possible_hooks()}
        self.record_start_time : dict[str,float] = {}
    def at_enter(self , hook_name , verbosity : int = 0):
        super().at_enter(hook_name , verbosity)
        if self.recording: 
            self.record_start_time[hook_name] = time.time()
    def at_exit(self, hook_name , verbosity : int = 0):
        if self.recording: 
            self.record_hook_times[hook_name].append(time.time() - self.record_start_time[hook_name])
        super().at_exit(hook_name , verbosity)
    def on_summarize_model(self):
        if self.recording: 
            columns = ['hook_name' , 'num_calls', 'total_time' , 'avg_time']
            values  = [[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self.record_hook_times.items() if v]
            df = pd.DataFrame(values , columns = columns).sort_values(by=['total_time'],ascending=False).head(5)
            FUNC.display.data_frame(df , text_ahead='Callback Time costs')

class BatchDisplay(BaseCallBack):
    '''display batch progress bar'''
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info(verbosity = self.verbosity)
        self.show_info = self.verbosity >= 10

    @property
    def dataloader(self) -> BatchDataLoader | Any : return self.trainer.dataloader

    def on_train_batch_end(self):  
        if self.show_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.output.loss_item:.5f}')
    
    def on_validation_batch_end(self):   
        if self.show_info: 
            self.dataloader.display(f'Valid Ep#{self.status.epoch:3d} score : {self.metrics.output.score:.5f}')

    def on_test_batch_end(self):         
        if self.show_info: 
            self.dataloader.display('Test {} {} score : {:.5f}'.format(
                self.status.model_submodel , self.trainer.batch_dates[self.batch_idx] , self.metrics.output.score))
            
class StatusDisplay(BaseCallBack):
    '''display epoch and event information'''
    RESULT_PATH = PATH.upload.joinpath('model_results.yaml')
    SUMMARY_NDIGITS : ClassVar[dict[str,int]] = {'Avg':4,'Sum':2,'Std':4,'T':2,'IR':4}

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info(verbosity = self.verbosity)
        self.show_info_step = [0,10,5,3,2,1][min(self.verbosity // 2 , 5)]
        self.record_init_time = time.time()
        self.record_times : dict[str,float] = {}
        self.record_texts : dict[str,str]   = {}
        self.record_epoch_model : int = 0
        self.record_epoch_stage : int = 0
        self.record_model_stage : int = 0
    @property
    def path_test(self): return str(self.config.model_base_path.rslt('test.xlsx'))
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
    def optimizer(self): return getattr(self.trainer.model , 'optimizer')
    def display(self , *args , **kwargs):
        return self.logger.info(*args , **kwargs) if (self.show_info_step or self.initial_models) else self.logger.debug(*args , **kwargs)
    def event_sdout(self , event) -> str:
        if event == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.status.epoch} , effective at epoch {self.status.epoch + 1}'
            if self.speedup2x: sdout += ', and will speedup2x'
        elif event == 'new_attempt':
            sdout = '{attempt} {epoch} : {status}, Next attempt goes!'.format(**self.record_texts)
        elif event == 'nanloss':
            sdout = 'Model {model_date}.{model_num} Attempt{attempt}, epoch{epoch} got nanloss!'.format(self.status.as_dict())
        else:
            raise KeyError(event)
        return sdout

    def tic(self , key : str): self.record_times[key] = time.time()
    def toc(self , key : str): return time.time() - self.record_times[key]
    def tic_str(self , key : str):
        self.tic(key)
        return 'Start Process [{}] at {:s}!'.format(key.capitalize() , time.ctime(self.record_times[key]))
    def toc_str(self , key : str , avg = False): 
        toc = self.toc(key)
        if avg and self.record_model_stage * self.record_epoch_stage:
            self.record_texts[key] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.capitalize() , toc / 3600 , toc / 60 / self.record_model_stage , toc / self.record_epoch_stage)
        else:
            self.record_texts[key] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.capitalize() , toc)
        return self.record_texts[f'{key}']

    # callbacks
    def on_configure_model(self):
        self.logger.warning('Model Specifics:')
        self.config.print_out()
    def on_summarize_model(self):
        if not self.test_summarized: self.summarize_test_result()
        if self.summary_df.empty: return
        test_scores = {
            '{}.{}'.format(*col):'|'.join([f'{k}({round(self.summary_df[col][k],v)})' for k,v in self.SUMMARY_NDIGITS.items() 
                                           if k in self.summary_df[col].index]) for col in self.summary_df.columns}
    
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : time.ctime(self.record_init_time) ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : str(self.config.model_data_types) ,
            '4_label' : ','.join(self.config.model_labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_fit'   : self.record_texts.get('fit'),
            '7_test'  : self.record_texts.get('test'),
            '8_result': test_scores,
        }
        PATH.dump_yaml(result , self.RESULT_PATH)

    def on_data_start(self):    self.logger.critical(self.tic_str('data'))
    def on_data_end(self):      self.logger.critical(self.toc_str('data'))
    def on_fit_start(self):     self.logger.critical(self.tic_str('fit'))
    def on_fit_end(self):       self.logger.critical(self.toc_str('fit' , avg=True))

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
        if self.trainer.status.epoch < 0: return
        self.record_texts['status'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, best{: .4f}, lr{:.1e}'.format(
            self.metrics.latest['train.loss'], self.metrics.latest['train.score'] ,
            self.metrics.latest['valid.score'] , self.metrics.best_metric , 
            self.optimizer.last_lr)
        if self.status.epoch % self.show_info_step == 0: 
            self.display('{attempt} {epoch} : {status}'.format(**self.record_texts))
    
    def on_fit_epoch_end(self):
        if self.status.fit_loop_breaker: self.record_texts['exit'] = self.status.fit_loop_breaker.trigger_reason
        while self.status.epoch_event: self.display(self.event_sdout(self.status.epoch_event.pop()))
    
    def on_fit_model_end(self):
        train_score = self.metrics.latest.get('train.score' , 0)
        valid_score = self.metrics.latest.get('valid.score' , 0)
        best_score = self.status.best_attempt_metric if self.status.best_attempt_metric else valid_score
        self.record_texts['status'] = f'Train{train_score: .4f} Valid{valid_score: .4f} BestVal{best_score: .4f}'
        self.record_texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.toc('model') / 60 , self.toc('model') / (self.record_epoch_model + 1))
        self.logger.warning('{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.record_texts))
    
    def on_test_start(self): 
        self.logger.critical(self.tic_str('test'))
                
        self.test_df_date = pd.DataFrame()
        self.test_df_model = pd.DataFrame()
        self.test_summarized = False

    def on_test_submodel_end(self):
        self.update_test_score()

    def on_test_end(self): 
        self.logger.warning('Testing Mean Score({}):'.format(self.config.train_criterion_score))
        self.summarize_test_result()
        self.logger.critical(self.toc_str('test'))

    def update_test_score(self):
        df_date = pd.DataFrame({
            'model_num' : self.status.model_num , 
            'model_date' : self.status.model_date ,
            'submodel' : self.status.model_submodel ,
            'date' : self.model_test_dates ,
            'value' : self.metrics.scores[-len(self.model_test_dates):]
        })
        df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index()
        df_model['model_date'] = df_model['model_date'].astype(str)
        self.test_df_date = pd.concat([self.test_df_date , df_date])
        self.test_df_model = pd.concat([self.test_df_model , df_model])

    def summarize_test_result(self):
        self.summary_df = pd.DataFrame()
        if self.test_df_model.empty: return
        cat_stat = [md for md in self.test_df_model['model_date'].unique()] + ['Avg' , 'Sum' , 'Std' , 'T' , 'IR']
        cat_subm = ['best' , 'swalast' , 'swabest']

        dfs : dict[str,pd.DataFrame|pd.Series] = {}
        dfs['Avg'] = self.test_df_date.groupby(['model_num','submodel'])['value'].mean()
        dfs['Sum'] = self.test_df_date.groupby(['model_num','submodel'])['value'].sum()
        dfs['Std'] = self.test_df_date.groupby(['model_num','submodel'])['value'].std()

        dfs['T']   = ((dfs['Avg'] / dfs['Std']) * (len(self.test_df_date['date'].unique())**0.5))
        dfs['IR']  = ((dfs['Avg'] / dfs['Std']) * ((240 / 10)**0.5))

        stat_df = pd.concat([df.reset_index().assign(stat=k) for k,df in dfs.items()])

        # display summary
        df = pd.concat([self.test_df_model.rename(columns={'model_date':'stat'}) , stat_df])

        base_name = self.config.model_module
        if self.config.module_type == 'boost' and self.config.model_booster_optuna: base_name += '.optuna'
        df['model_num'] = df['model_num'].map(lambda x: f'{base_name}.{x}')
        df['submodel']  = pd.Categorical(df['submodel'] , categories = cat_subm, ordered=True) 
        df['stat']      = pd.Categorical(df['stat']     , categories = cat_stat, ordered=True) 

        self.summary_df = df.rename(columns={'model_num':'model'}).pivot_table('value' , 'stat' , ['model' , 'submodel'] , observed=False).round(4)

        # more than 100 rows of test_df_model means the cycle is month / day
        df_display = self.summary_df
        if len(df_display) > 100: df_display = df_display.loc[['Avg' , 'Sum' , 'Std' , 'T' , 'IR']]
        FUNC.display.data_frame(df_display , text_after = f'Test results are saved to {self.path_test}')

        # export excel
        rslt = {'summary' : self.summary_df , 'by_model' : self.test_df_model}
        for model_num in self.config.model_num_list:
            df : pd.DataFrame = self.test_df_date[self.test_df_date['model_num'] == model_num].pivot_table(
                'value' , 'date' , 'submodel' , observed=False)
            df_cum = df.cumsum().rename(columns = {submodel:f'{submodel}_cum' for submodel in df.columns})
            df = df.merge(df_cum , on = 'date').rename_axis(None , axis = 'columns')
            rslt[f'{model_num}'] = df
        FUNC.dfs_to_excel(rslt , self.path_test)
        self.test_summarized = True
