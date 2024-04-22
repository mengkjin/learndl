#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --process=0 --resume=0 --checkname=1 

import torch
import numpy as np
import pandas as pd
import itertools , os

from copy import deepcopy
from dataclasses import dataclass , field
from torch import nn , Tensor
from typing import Any , ClassVar , Literal , Optional

import util as U
from .environ import DIR

from .data.DataFetcher import DataFetcher
from .DataModule import BatchData , DataModule
from .util.trainer import pipeline as TRAIN # TrainerUtil
from .util import optim as OPTIM # TrainerUtil
from .model import model as MODEL

from .func.date import today , date_offset
from .func import list_converge


# from audtorch.metrics.functional import *

logger = U.Logger()
config = U.TrainConfig()

@dataclass
class Predictor:
    '''for any model to predict recent/history data'''
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

        data_module_old = DataModule(model_config , predict = False) if require_model_data_old else None
        data_module_new = DataModule(model_config , predict = True) 

        end_dt = min(end_dt , max(data_module_new.test_full_dates))
        pred_dates = calendar[(calendar['calendar'] >= start_dt) & (calendar['calendar'] <= end_dt) & (calendar['trade'])]['calendar'].values

        df_task = pd.DataFrame({
            'pred_dates' : pred_dates ,
            'model_date' : [max(model_dates[model_dates < d_pred]) for d_pred in pred_dates] ,
            'calculated' : 0 ,
        })

        with torch.no_grad():
            df_list = []
            for data_module in [data_module_old , data_module_new]:
                if data_module is None: continue
                assert isinstance(data_module , DataModule)
                for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
                    print(model_date , 'old' if (data_module is data_module_old) else 'new') 
                    assert isinstance(model_date , int) , model_date
                    data_module.setup('predict' ,  model_param , model_date)
                    sd_path = f'{model_path}/{self.model_num}/{model_date}.{self.model_type}.pt'

                    net = MODEL.new(model_config.model_module , model_param , torch.load(sd_path) , device)
                    net.eval()

                    loader = data_module.predict_dataloader()
                    secid  = data_module.datas.secid
                    tdates = data_module.model_test_dates
                    iter_tdates = np.intersect1d(df_sub['pred_dates'][df_sub['calculated'] == 0] , tdates)

                    for tdate in iter_tdates:
                        batch_data = loader[np.where(tdates == tdate)[0][0]]

                        pred = U.trainer.Output(net(batch_data.x)).pred()
                        if len(pred) == 0: continue
                        df = pd.DataFrame({'secid' : secid[batch_data.i[:,0].cpu().numpy()] , 'date' : tdate , 
                                           self.model_name : pred.cpu().flatten().numpy()})
                        df_list.append(df)
                        df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

            del data_module_new , data_module_old
        return pd.concat(df_list , axis = 0)

@dataclass
class ModelTestor:
    '''Check if a newly defined model can be forward correctly'''
    config      : U.TrainConfig
    net         : nn.Module
    data_module : DataModule
    batch_data  : BatchData
    metrics     : U.Metrics

    @classmethod
    def new(cls , module = 'tra_lstm' , model_data_type = 'day'):
        config = U.TrainConfig.load(override = {'model_module' : module , 'model_data_type' : model_data_type} , makedir = False)
        data_module = DataModule(config , predict=True)
        data_module.setup('predict' , config.model_param[0] , data_module.model_date_list[0])   
        
        batch_data = data_module.predict_dataloader()[0]

        net = MODEL.new(module , config.model_param[0])
        metrics = U.Metrics(config.train_param['criterion']).new_model(config.model_param[0] , config)
        return cls(config , net , data_module , batch_data , metrics)

    def try_forward(self) -> None:
        '''as name says, try to forward'''
        if isinstance(self.batch_data.x , torch.Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')
        getattr(self.net , 'dynamic_data_assign' , lambda *x:None)(self)
        self.net_output = U.trainer.Output(self.net(self.batch_data.x))
        print(f'y shape is {self.net_output.pred().shape}')
        print(f'Test Forward Success')

    def try_metrics(self) -> None:
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): self.try_forward()
        label , weight = self.batch_data.y , self.batch_data.w
        penalty_kwargs = {}
        penalty_kwargs.update({'net' : self.net , 'hidden' : self.net_output.hidden() , 'label' : label})
        metrics = self.metrics.calculate('train' , label , self.net_output.pred() , weight , self.net , penalty_kwargs)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')

class ModelTrainer():
    '''Control the whole process of training'''
    default_model_type = 'best'

    def __init__(self , timer = True ,  **kwargs):
        self.pipe           = U.Pipeline(config , logger)
        self.ptimer         = U.time.PTimer(timer)
        self.checkpoint     = U.Checkpoints(config.mem_storage) # state_dict checkpoint
        self.deposition     = U.Checkpoints(False)              # state_dict deposition
        self.device         = U.Device()
        self.metrics        = U.Metrics(config.train_param['criterion'])

    def main_process(self):
        '''Main stage of load_data + fit + test'''
        for self.stage in config.stage_queue:
            getattr(self , f'process_{self.stage}')()
        self.pipe.dump_info()
        self.ptimer.print()
    
    def process_data(self):
        '''Main process of loading model data'''
        logger.critical(self.pipe.tic_str('data'))
        self.data_module = DataModule(config)
        logger.critical(self.pipe.toc_str('data'))
        
    def process_fit(self):
        '''Main process of fitting'''
        self.process_fit_start()
        for self.model_date , self.model_num in self.model_iter():
            self.model_fit_start()
            self.model_fit_body()
            self.model_fit_end()
        self.process_fit_end()

    def process_test(self):
        '''Main process of testing'''
        self.process_test_start()
        for self.model_date , self.model_num in self.model_iter():
            self.model_test_start()
            self.model_test_body()
            self.model_test_end()
        self.process_test_end()

    def model_iter(self):
        '''Iteration of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(self.data_module.model_date_list , config.model_num_list))
        if config.resume_training and self.stage == 'fit':
            models_trained = np.full(len(new_iter) , True)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(self.model_path(model_date,model_num = model_num)):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = U.Filtered(new_iter , ~models_trained)
        return new_iter
    
    def process_fit_start(self):
        '''to do at process fit start'''
        logger.critical(self.pipe.tic_str('fit'))
        self.data_module.reset_dataloaders()
        config.Model.save(config.model_base_path)
        self.pipe.fit_stage()
        self.model_type = 'best'

    def process_fit_end(self):
        '''to do at process fit end'''
        logger.critical(self.pipe.toc_str('fit'))

    def model_fit_start(self):
        '''to do when starting fit of one model'''
        with self.ptimer(f'{self.stage}/start'):
            self.pipe.tic('model')
            self.pipe.new_fit_model(self.model_num , self.model_date)

            self.param = config.model_param[self.model_num]
            self.metrics.new_model(self.param , config)
            self.checkpoint.new_model(self.param , self.model_date)

            self.data_module.setup('fit' , self.param , self.model_date)
            self.model_dict = U.FittedModel.get_dict(config.output_types , self.checkpoint)

    def model_fit_body(self):
        '''main method of fitting one model'''
        while self.pipe.loop_status != 'model':
            self.model_fit_loop_start()
            self.model_fit_loop_body()
            self.model_fit_loop_assess()

    def model_fit_loop_start(self):
        '''on new attempt, initialize net , optimizer(scheduler)'''
        with self.ptimer(f'{self.stage}/loop_start'):
            self.pipe.new_loop()
            if self.pipe.loop_status == 'attempt':
                self.load_model(True)
                self.optimizer = OPTIM.Optimizer(self.net , config , self.transferred , self.pipe.attempt)

    def model_fit_loop_body(self):
        '''go through train/val dataset, calculate loss/score , update values'''
        with self.ptimer(f'{self.stage}/train_loader'):
            self.loop_dataloader('train')
        
        with self.ptimer(f'{self.stage}/valid_loader') , torch.no_grad():
            self.loop_dataloader('valid')

        self.print_progress(self.optimizer.step(self.pipe.epoch))

    def model_fit_loop_assess(self):
        '''Update condition of continuing training epochs , restart attempt if early exit or nanloss'''
        if self.pipe.nanloss: return
        with self.ptimer(f'{self.stage}/loop_assess'):
            self.pipe.assess_terminate()
            for model in self.model_dict.values():
                model.assess(self.net , self.pipe.epoch , self.pipe.valid_score , self.pipe.valid_loss)
            self.print_progress('epoch_step')

    def model_fit_loop_end(self):
        '''on model fitted save model , print if new_attempt'''
        with self.ptimer(f'{self.stage}/loop_end'):
            if self.pipe.loop_status == 'model':
                self.save_model()
            elif self.pipe.loop_status == 'attempt': 
                self.print_progress('new_attempt')

    def model_fit_end(self):
        '''to do when ending fit of one model'''
        with self.ptimer(f'{self.stage}/end'):
            self.checkpoint.del_all()
            self.pipe.end_fit_model()
            self.print_progress('model_end')

    def process_test_start(self):
        '''to do at process test start'''
        logger.critical(self.pipe.tic_str('test'))
        self.data_module.reset_dataloaders()
        logger.warning('Each Model Date Testing Mean Score({}):'.format(config.train_param['criterion']['score']))
        self.pipe.test_stage()

    def process_test_end(self):
        '''to do at process test end'''
        self.pipe.end_test_stage()
        logger.critical(self.pipe.toc_str('test'))

    def model_test_start(self):
        '''to do at test start on every model'''
        with self.ptimer(f'{self.stage}/start'):
            self.pipe.new_test_model(self.model_num , self.model_date , self.data_module.model_test_dates)

            self.param = config.model_param[self.model_num]
            self.metrics.new_model(self.param , config)
            self.checkpoint.new_model(self.param , self.model_date)

            self.data_module.setup('test' , self.param , self.model_date)   
                            
    def model_test_body(self):
        '''to do at on every model'''
        if not self.deposition.exists(self.model_path(self.model_date)): 
            self.model_fit_start()
            self.model_fit_body()
            self.model_fit_end()
        
        with self.ptimer(f'{self.stage}/forecast') , torch.no_grad():
            test_dates = np.concatenate([self.data_module.early_test_dates , self.data_module.model_test_dates])
            for self.model_type in config.output_types:
                self.load_model(False , self.model_type)
                self.loop_dataloader('test' , warm_up = len(self.data_module.early_test_dates) , dates = test_dates)
                self.pipe.update_test_score(self.model_type)

    def model_test_end(self):
        '''to do at test end on every model'''
        with self.ptimer(f'{self.stage}/end'):
            self.pipe.end_test_model()

    def loop_dataloader(self , dataset , warm_up : int = 0 , dates : Optional[np.ndarray] = None):
        '''loop of batch data'''
        self.dataset : Literal['train' , 'valid' , 'test'] = dataset
        self.pipe.set_dataset(dataset)
        self.pipe.new_metric(self.model_type)

        self.net.train() if dataset == 'train' else self.net.eval()
        if dataset == 'train':   loader = self.data_module.train_dataloader().add_text('Train Ep#{ep:3d} loss : {ls:.5f}')
        elif dataset == 'valid': loader = self.data_module.val_dataloader().add_text('Valid Ep#{ep:3d} score : {sc:.5f}')
        elif dataset == 'test' : loader = self.data_module.test_dataloader().add_text('Test {mtype} {i} score : {sc:.5f}')
        if dates is None: dates = np.zeros(len(loader))
        self.assert_equity(len(loader) , len(dates))
        for i , self.batch_data in enumerate(loader):
            if dates[i]: self.assert_equity(dates[i] , self.data_module.y_date[self.batch_data.i[0,1]]) 
            self.forward()
            if i < warm_up: continue  # before this is warmup stage , only forward
            self.cal_metric()
            self.backward()
            loader.display(i=dates[i],mtype=self.model_type,ep=self.pipe.epoch,ls=self.pipe.aggloss,sc=self.pipe.aggscore)
            if (i + 1) % 20 == 0 : torch.cuda.empty_cache()

        self.pipe.collect_metric()
        self.pipe.collect_lr(self.optimizer.last_lr)

    def forward(self):
        with self.ptimer(f'{self.stage}/{self.dataset}/forward'):
            if self.batch_data.is_empty: 
                self.net_output = U.trainer.Output(torch.Tensor().requires_grad_())
            else:
                getattr(self.net , 'dynamic_data_assign' , lambda *x:None)(self)
                self.net_output = U.trainer.Output(self.net(self.batch_data.x))

    def cal_metric(self):
        with self.ptimer(f'{self.stage}/{self.dataset}/loss'):
            self.metrics.calculate(self.dataset , self.batch_data.y , self.net_output.pred() , 
                                   self.batch_data.w , self.net , self.penalty_kwargs , assert_nan = True)
            self.pipe.record_metric(self.metrics)

    def backward(self):
        if self.dataset != 'train': return
        with self.ptimer(f'{self.stage}/{self.dataset}/backward'): 
            self.optimizer.backward(self.metrics.loss)

    @property
    def penalty_kwargs(self) -> dict:
        return {'net' : self.net , 'hidden' : self.net_output.hidden() , 'label' : self.batch_data.y}
    
    def print_progress(self , key , sdout = None) -> None:
        '''Print out status giving display conditions and looping conditions'''
        if key is None: return
        printers = [logger.info] if (config.verbosity > 2 or self.pipe.initial_models) else [logger.debug]
        if key == 'model_end':
            sdout = '{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.pipe.texts)
            printers = [logger.warning]
        elif key == 'epoch_step':
            if self.pipe.epoch_print: sdout = '{attempt} {epoch} : {trainer}'.format(**self.pipe.texts)
        elif key == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.pipe.epoch} , effective at epoch {self.pipe.epoch+1}' + \
                ', and will speedup2x' * config.train_param['trainer']['learn_rate']['reset']['speedup2x']
        elif key == 'new_attempt':
            sdout = '{attempt} {epoch} : {status}, Next attempt goes!'.format(**self.pipe.texts)
        else:
            raise KeyError(key)
        if sdout is not None: [prt(sdout) for prt in printers]
    
    def load_model(self , training : bool , model_type = default_model_type):
        '''load model state dict, return net and a sign of whether it is transferred'''
        with self.ptimer(f'load_model'):
            self.transferred = False
            if training:         
                if config.train_param['transfer']:
                    model_path = self.model_path(self.data_module.prev_model_date(self.model_date))
                    self.transferred = self.deposition.exists(model_path)
                else:
                    model_path = self.model_path()
            else:
                model_path = self.model_path(self.model_date , model_type)
            self.net = MODEL.new(config.model_module , self.param , self.deposition.load(model_path) , self.device)
     
    def save_model(self):
        '''save model state dict to deposition'''
        with self.ptimer(f'save_model'):
            getattr(self.net , 'dynamic_data_unlink' , lambda *x:None)()
            self.net = self.net.cpu()
            for model_type , model in self.model_dict.items():
                sd = model.state_dict(self.net , self.data_module.train_dataloader())
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

    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)

    @classmethod
    def fit(cls , process = -1 , resume = -1 , checkname = -1 , timer = False , parser_args = None):
        if parser_args is None: parser_args = config.parser_args({'process':process,'resume':resume,'checkname':checkname})

        config.reload(do_process=True,par_args=parser_args)
        config.set_config_environment()

        logger.warning('Model Specifics:')
        config.print_out()

        app = cls(timer = timer)
        app.main_process()

if __name__ == '__main__': 
    ModelTrainer.fit(parser_args = config.parser_args())
 