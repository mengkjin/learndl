#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --stage=0 --resume=0 --checkname=1 
import itertools , os
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from inspect import currentframe
from torch import nn , no_grad , Tensor
from typing import ClassVar , Literal , Optional

from .environ import DIR
from . import util as U
from .util import callback as CB
from .data.DataFetcher import DataFetcher
from .DataModule import BatchData , DataModule
from .model import model as MODEL
from .func.date import today , date_offset

logger = U.Logger()
config = U.TrainConfig()
ptimer = U.time.PTimer(True)
hooker = CB.ModelHook(ptimer)

@dataclass
class Predictor:
    '''for a model to predict recent/history data'''
    model_name : str
    model_type : Literal['best' , 'swalast' , 'swabest'] = 'swalast'
    model_num  : int = 0
    alias : str | None = None
    df    : pd.DataFrame | None = None

    destination : ClassVar[str] = '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha'
    secid_col : ClassVar[str] = 'secid'
    date_col  : ClassVar[str] = 'date'

    def __post_init__(self):
        if self.alias is None: self.alias = self.model_name

    def deploy(self , df : pd.DataFrame | None = None , overwrite = False , secid_col = secid_col , date_col = date_col):
        '''deploy df by day to class.destination'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        os.makedirs(f'{self.destination}/{self.alias}' , exist_ok=True)
        for date , subdf in df.groupby(date_col):
            des_path = f'{self.destination}/{self.alias}/{self.alias}_{date}.txt'
            if overwrite or not os.path.exists(des_path):
                subdf.drop(columns='date').set_index(secid_col).to_csv(des_path, sep='\t', index=True, header=False)

    def get_df(self , start_dt = -10 , end_dt = 20991231):
        '''save recent prediction to self.df'''
        self.df = self.predict(start_dt= start_dt , end_dt = end_dt)
        return self

    def df_corr(self , df = None , window = 30 , secid_col = secid_col , date_col = date_col):
        '''prediction correlation of ecent days'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        df = df[df[date_col] >= today(-window)]
        assert isinstance(df , pd.DataFrame)
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    def write_df(self , path):
        '''write down prediction df'''
        assert isinstance(self.df , pd.DataFrame)
        self.df.to_feather(path)

    def predict(self , start_dt = -10 , end_dt = 20991231) -> pd.DataFrame:
        '''predict recent days'''
        if start_dt <= 0: start_dt = today(start_dt)

        model_path = f'{DIR.model}/{self.model_name}'
        device       = U.Device()
        model_config = U.TrainConfig.load(model_path)

        model_param = model_config.model_param[self.model_num]
        model_files = sorted([p for p in os.listdir(f'{model_path}/{self.model_num}') if p.endswith(f'{self.model_type}.pt')])
        model_dates = np.array([int(mf.split('.')[0]) for mf in model_files])

        start_dt = max(start_dt , int(date_offset(min(model_dates) ,1)))
        calendar = DataFetcher.load_target_file('information' , 'calendar')
        assert calendar is not None

        require_model_data_old = (start_dt <= today(-100))

        data_mod_old = DataModule(model_config , False).load_data() if require_model_data_old else None
        data_mod_new = DataModule(model_config , True).load_data() 

        end_dt = min(end_dt , max(data_mod_new.test_full_dates))
        pred_dates = calendar[(calendar['calendar'] >= start_dt) & (calendar['calendar'] <= end_dt) & (calendar['trade'])]['calendar'].values

        df_task = pd.DataFrame({
            'pred_dates' : pred_dates ,
            'model_date' : [max(model_dates[model_dates < d_pred]) for d_pred in pred_dates] ,
            'calculated' : 0 ,
        })

        with no_grad():
            df_list = []
            for data_mod in [data_mod_old , data_mod_new]:
                if data_mod is None: continue
                for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
                    print(model_date , 'old' if (data_mod is data_mod_old) else 'new') 
                    assert isinstance(model_date , int) , model_date
                    data_mod.setup('predict' ,  model_param , model_date)
                    sd_path = f'{model_path}/{self.model_num}/{model_date}.{self.model_type}.pt'

                    net = MODEL.new(model_config.model_module , model_param , torch.load(sd_path) , device)
                    net.eval()

                    loader = data_mod.predict_dataloader()
                    secid  = data_mod.datas.secid
                    tdates = data_mod.model_test_dates
                    iter_tdates = np.intersect1d(df_sub['pred_dates'][df_sub['calculated'] == 0] , tdates)

                    for tdate in iter_tdates:
                        batch_data = loader[np.where(tdates == tdate)[0][0]]

                        pred = U.trainer.Output(net(batch_data.x)).pred
                        if len(pred) == 0: continue
                        df = pd.DataFrame({'secid' : secid[batch_data.i[:,0].cpu().numpy()] , 'date' : tdate , 
                                           self.model_name : pred.cpu().flatten().numpy()})
                        df_list.append(df)
                        df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

            del data_mod_new , data_mod_old
        return pd.concat(df_list , axis = 0)

@dataclass
class ModelTestor:
    '''Check if a newly defined model can be forward correctly'''
    config      : U.TrainConfig
    net         : nn.Module
    data_mod    : DataModule
    batch_data  : BatchData
    metrics     : U.Metrics

    @classmethod
    def new(cls , module = 'tra_lstm' , model_data_type = 'day'):
        config = U.TrainConfig.load(override = {'model_module' : module , 'model_data_type' : model_data_type} , makedir = False)
        data_mod = DataModule(config , True).load_data()
        data_mod.setup('predict' , config.model_param[0] , data_mod.model_date_list[0])   
        
        batch_data = data_mod.predict_dataloader()[0]

        net = MODEL.new(module , config.model_param[0])
        metrics = U.Metrics(config.train_param['criterion']).new_model(config.model_param[0] , config)
        return cls(config , net , data_mod , batch_data , metrics)

    def try_forward(self) -> None:
        '''as name says, try to forward'''
        if isinstance(self.batch_data.x , Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')
        getattr(self.net , 'dynamic_data_assign' , lambda *x:None)(self)
        self.net_output = U.trainer.Output(self.net(self.batch_data.x))
        print(f'y shape is {self.net_output.pred.shape}')
        print(f'Test Forward Success')

    def try_metrics(self) -> None:
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): self.try_forward()
        label , weight = self.batch_data.y , self.batch_data.w
        penalty_kwargs = {}
        penalty_kwargs.update({'net' : self.net , 'hidden' : self.net_output.hidden , 'label' : label})
        metrics = self.metrics.calculate('train' , label , self.net_output.pred , weight , self.net , penalty_kwargs)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')
    
class ModelTrainer():
    '''run through the whole process of training'''
    default_model_type = 'best'

    def __init__(self ,  **kwargs):
        self.config     = config
        self.ptimer     = ptimer
        self.logger     = logger
        config.reload(do_parser = True , par_args = config.parser_args(**kwargs))
        
        self.device     = U.Device()
        self.data_mod   = DataModule(config)

        self.pipe       = U.Pipeline(config , logger)
        self.checkpoint = U.Checkpoint('mem' if config.mem_storage else 'disk')
        self.deposition = U.Deposition()
        self.metrics    = U.Metrics(config.train_param['criterion'])

        self.callbacks  = [self.pipe , self.checkpoint , self.deposition , self.metrics ,
                           CB.DynamicDataLink()]

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.configure_model()
        for self.stage in config.stage_queue: 
            getattr(self , f'stage_{self.stage}')()
        ptimer.summarize()

    @hooker.hook
    def configure_model(self):
        pass
    
    def stage_data(self):
        '''stage of loading model data'''
        self.on_data_start()
        self.data_mod.load_data()
        self.on_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.on_fit_start()
        for self.model_date , self.model_num in self.model_iter():
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.on_test_start()
        for self.model_date , self.model_num in self.model_iter():
            self.test_model()
        self.on_test_end()

    def model_iter(self):
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(self.data_mod.model_date_list , config.model_num_list))
        if config.resume_training and self.stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(self.model_path(model_date,model_num = model_num)):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = U.Filtered(new_iter , ~models_trained)
        return new_iter
    
    @hooker.hook
    def on_data_start(self): pass
    @hooker.hook
    def on_data_end(self): pass
    @hooker.hook
    def on_fit_start(self): pass
    @hooker.hook
    def on_fit_end(self): pass
    @hooker.hook
    def on_test_start(self): pass
    @hooker.hook
    def on_test_end(self): pass
    @hooker.hook
    def on_fit_model_start(self):
        '''to do when starting fit of one model'''
        self.param : dict = config.model_param[self.model_num]
        self.data_mod.setup('fit' , self.config.model_param[self.model_num] , self.model_date)
        pass

    @hooker.hook
    def fit_model(self):
        self.on_fit_model_start()
        while self.fit_loop:   
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            if self.pipe.nanloss: return

            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()
            
            if self.pipe.loop_new_attempt:  self.print_progress('new_attempt')
            if self.pipe.loop_terminate: self.save_model()
        self.on_fit_model_end()

    @hooker.hook
    def test_model(self):
        self.on_test_model_start()
        for self.model_type in config.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    @hooker.hook
    def on_fit_model_end(self): pass

    @hooker.hook
    def on_train_batch(self):
        self.batch_forward()
        self.batch_metrics()
        self.batch_backward()
    @hooker.hook
    def on_validation_batch(self):
        self.batch_forward()
        self.batch_metrics()
    @hooker.hook
    def on_test_batch(self):
        self.assert_equity(self.test_dates[self.batch_idx] , self.data_mod.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        if self.batch_idx < self.test_warm_up: return  # before this is warmup stage , only forward
        self.batch_metrics()
   
    def print_progress(self , key : Optional[str] = None , sdout = None):
        printers = [logger.info] if (config.verbosity > 2 or self.pipe.initial_models) else [logger.debug]
        if key == 'epoch_step':
            if self.pipe.epoch_print: sdout = '{attempt} {epoch} : {status}'.format(**self.pipe.texts)
        elif key == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.pipe.epoch} ,' + \
                f' effective at epoch {self.pipe.epoch+1}' + \
                ', and will speedup2x' * config.train_param['trainer']['learn_rate']['reset']['speedup2x']
        if sdout is not None: [prt(sdout) for prt in printers]

    @hooker.hook 
    def on_train_epoch_start(self):
        self.dataset = 'train'
        if self.pipe.loop_new_attempt: 
            self.pipe.new_attempt()
            self.checkpoint.new_model(self.param , self.model_date)
            self.models = U.FittestModel.get_models(config.model_types , self.checkpoint)
            self.load_model(True)
            self.optimizer = U.Optimizer(self.net , config , self.transferred , self.pipe.attempt)
        self.pipe.new_epoch()
        self.pipe.new_metric(self.dataset , 'best')
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data_mod.train_dataloader().init_tqdm('Train Ep#{ep:3d} loss : {ls:.5f}')

    @hooker.hook 
    def on_train_epoch_end(self):
        self.pipe.collect_metric()
        self.pipe.collect_lr(self.optimizer.last_lr)
        self.print_progress(self.optimizer.scheduler_step(self.pipe.epoch))

    @hooker.hook 
    def on_validation_epoch_start(self):
        self.dataset = 'valid'
        self.pipe.new_metric(self.dataset , 'best')
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data_mod.val_dataloader().init_tqdm('Valid Ep#{ep:3d} score : {sc:.5f}')
        
    @hooker.hook 
    def on_validation_epoch_end(self):
        self.pipe.collect_metric()
        self.pipe.assess_terminate()
        
        for fittest_model in self.models.values():
            fittest_model.assess(self.net , self.pipe.epoch , self.pipe.valid_score , self.pipe.valid_loss)
        self.print_progress('epoch_step')
        torch.set_grad_enabled(True)

    @hooker.hook 
    def on_test_model_start(self) -> None:
        if not self.deposition.exists(self.model_path(self.model_date)): 
            self.fit_model()
        
        self.param = config.model_param[self.model_num]
        self.data_mod.setup('test' , self.param , self.model_date)
        self.pipe.new_test_model(self.model_num , self.model_date , self.data_mod.model_test_dates)
        self.metrics.new_model(self.param , config)

        self.dataset = 'test'
        self.test_dates = np.concatenate([self.data_mod.early_test_dates , self.data_mod.model_test_dates])
        self.test_warm_up = len(self.data_mod.early_test_dates)
        self.net.eval()
        torch.set_grad_enabled(False)

    @hooker.hook 
    def on_test_model_end(self) -> None:
        torch.set_grad_enabled(True)
        self.pipe.end_test_model()

    @hooker.hook 
    def on_test_model_type_start(self) -> None:
        self.load_model(False , self.model_type)
        self.pipe.new_metric(self.dataset , self.model_type)
        self.dataloader = self.data_mod.test_dataloader().init_tqdm('Test {mt} {dt} score : {sc:.5f}')
        self.assert_equity(len(self.dataloader) , len(self.test_dates))

    @hooker.hook 
    def on_test_model_type_end(self):
        self.pipe.collect_metric()
        self.pipe.update_test_score(self.model_type)

    @hooker.hook 
    def on_train_batch_start(self): pass
    
    @hooker.hook 
    def on_train_batch_end(self):
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(ep=self.pipe.epoch, ls=self.pipe.aggloss)
    
    @hooker.hook
    def on_validation_batch_start(self): pass
    @hooker.hook
    def on_validation_batch_end(self):
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(ep=self.pipe.epoch, sc=self.pipe.aggscore)
    
    @hooker.hook
    def on_test_batch_start(self): pass
    @hooker.hook
    def on_test_batch_end(self) -> None:
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(dt=self.test_dates[self.batch_idx] , mt=self.model_type , sc=self.pipe.aggscore)

    def batch_forward(self) -> None:
        if self.batch_data.is_empty:
            self.net_output = U.trainer.Output.empty()
        else:
            self.net_output = U.trainer.Output(self.net(self.batch_data.x))
    
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.dataset , self.batch_data.y , self.net_output.pred , 
                               self.batch_data.w , self.net , self.penalty_kwargs , assert_nan = True)
        self.pipe.record_metric(self.metrics)
    
    @hooker.hook 
    def on_before_backward(self): pass
    @hooker.hook 
    def on_after_backward(self): pass

    def batch_backward(self) -> None:
        assert self.dataset == 'train' , self.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.loss)
        self.on_after_backward()

    @hooker.hook
    def load_model(self , training : bool , model_type = default_model_type) -> None:
        '''load model state dict, return net and a sign of whether it is transferred'''
        if training and config.train_param['transfer']:         
            model_path = self.model_path(self.data_mod.prev_model_date(self.model_date))
        elif training:
            model_path = self.model_path()
        else:
            model_path = self.model_path(self.model_date , model_type)
        self.transferred = training and config.train_param['transfer'] and self.deposition.exists(model_path)
        self.net = MODEL.new(config.model_module , self.param , self.deposition.load(model_path) , self.device)

    @hooker.hook 
    def on_before_save_model(self): pass
     
    def save_model(self):
        '''save model state dict to deposition'''
        self.on_before_save_model()
        self.net = self.net.cpu()
        for model_type , fittest_model in self.models.items():
            sd = fittest_model.state_dict(self.net , self.data_mod.train_dataloader())
            self.deposition.save_state_dict(sd , self.model_path(self.model_date , model_type)) 
    
    def model_path(self , model_date = -1, model_type = default_model_type , base_path = None , model_num = None):
        '''get model path of deposition giving model date/type/base_path/num'''
        if model_num is None:
            model_dir = self.param.get('path')
        elif base_path is None:
            model_dir = f'{config.model_base_path}/{model_num}'
        else:
            model_dir = f'{base_path}/{model_num}'
        return '{}/{}.{}.pt'.format(model_dir , model_date , model_type)

    @property
    def fit_loop(self): return self.pipe.loop_continue

    @property
    def penalty_kwargs(self): return {'net':self.net,'hidden':self.net_output.hidden,'label':self.batch_data.y}
    
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)
    
    @staticmethod
    def hook_name(): return getattr(currentframe(),'f_back').f_code.co_name

    @classmethod
    def fit(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        app = cls(stage = stage , resume = resume , checkname = checkname , **kwargs)
        app.main_process()

if __name__ == '__main__': 
    ModelTrainer.fit(**config.parser_args().__dict__)
 