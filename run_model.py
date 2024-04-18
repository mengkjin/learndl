#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --process=0 --resume=0 --checkname=1 

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools , os , gc , time

from copy import deepcopy
from dataclasses import dataclass , field
from torch.optim.swa_utils import AveragedModel , update_bn
from torch.nn.utils.clip_grad import clip_grad_value_
from typing import Any , ClassVar , Literal

import src as src

from src.environ import DIR

from src.data.DataFetcher import DataFetcher
from src.data.ModelData import ModelData
from src.util import (
    BatchData , Device , FilteredIterator , Logger , ModelOutputs , 
    ProcessTimer , Storage , TrainConfig , Metrics
)

from src.func.date import today , date_offset
from src.func.basic import list_converge
from src.model import model

# from audtorch.metrics.functional import *

logger = Logger()
config = TrainConfig()

@dataclass
class ModelPred:
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
        if df is None: df = self.df
        if df is None: return NotImplemented
        os.makedirs(f'{self.destination}/{self.alias}' , exist_ok=True)
        for date , subdf in df.groupby(date_col):
            des_path = f'{self.destination}/{self.alias}/{self.alias}_{date}.txt'
            if overwrite or not os.path.exists(des_path):
                subdf.drop(columns='date').set_index(secid_col).to_csv(des_path, sep='\t', index=True, header=False)

    def get_df(self , start_dt = -10 , end_dt = 20991231 , old_model_data = False):
        self.df = self.predict(start_dt= start_dt , end_dt = end_dt, old_model_data = old_model_data)
        return self

    def df_corr(self , df = None , window = 30 , secid_col = secid_col , date_col = date_col):
        if df is None: df = self.df
        if df is None: return NotImplemented
        return df[df[date_col] >= today(-window)].pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    def write_df(self , path):
        assert isinstance(self.df , pd.DataFrame)
        self.df.to_feather(path)

    def predict(self , start_dt = -10 , end_dt = 20991231 , old_model_data = False):
        if start_dt <= 0: start_dt = today(start_dt)

        model_path = f'{DIR.model}/{self.model_name}'
        device       = Device()
        model_config = TrainConfig.load(model_path)

        model_param = model_config.model_param[self.model_num]
        model_files = sorted([p for p in os.listdir(f'{model_path}/{self.model_num}') if p.endswith(f'{self.model_type}.pt')])
        model_dates = np.array([int(mf.split('.')[0]) for mf in model_files])

        start_dt = max(start_dt , int(date_offset(min(model_dates) ,1)))
        calendar = DataFetcher.load_target_file('information' , 'calendar')
        assert calendar is not None

        require_model_data_old = (start_dt <= today(-100))

        model_data_old = ModelData(model_config , if_train = True) if require_model_data_old else None
        model_data_new = ModelData(model_config , if_train = True if old_model_data else False) 

        end_dt = min(end_dt , max(model_data_new.test_full_dates))
        pred_dates = calendar[(calendar['calendar'] >= start_dt) & (calendar['calendar'] <= end_dt) & (calendar['trade'])]['calendar'].values

        df_task = pd.DataFrame({
            'pred_dates' : pred_dates ,
            'model_date' : [max(model_dates[model_dates < d_pred]) for d_pred in pred_dates] ,
            'calculated' : 0 ,
        })

        with torch.no_grad():
            df_list = []

            for model_data in [model_data_old , model_data_new]:
                if model_data is None: continue

                for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
                    print(model_date , 'old' if (model_data is model_data_old) else 'new')
                    dataloader_param = model_data.get_dataloader_param('test' , model_date = model_date , param=model_param)   
                    model_data.create_dataloader(*dataloader_param)

                    sd_path = f'{model_path}/{self.model_num}/{model_date}.{self.model_type}.pt'
                    model_sd = torch.load(sd_path , map_location = model_data.device.device)
                    model = device(RunModel.new_model(model_config.model_module , model_param , state_dict=model_sd))
                    model.eval()

                    loader = model_data.dataloaders['test']
                    secid  = model_data.datas.secid
                    tdates = model_data.model_test_dates
                    #print(df_sub.loc[df_sub['calculated'] == 0 , 'pred_dates'])
                    #print(tdates)
                    iter_tdates = np.intersect1d(df_sub.loc[df_sub['calculated'] == 0 , 'pred_dates'] , tdates)

                    for tdate in iter_tdates:
                        batch_data = loader[np.where(tdates == tdate)[0][0]]

                        pred = ModelOutputs(model(batch_data.x)).pred()
                        if len(pred):
                            df_list.append(
                                pd.DataFrame({
                                'secid' : secid[batch_data.i[:,0].cpu().numpy()] , 'date' : tdate , 
                                self.model_name : pred.cpu().numpy().flatten() ,
                            }))
                            df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1
            df = pd.concat(df_list , axis = 0)
        del model_data_new , model_data_old
        return df

@dataclass
class ModelTest:
    config : TrainConfig
    net    : nn.Module
    batch_data : BatchData
    model_data : ModelData
    Metrics : Metrics

    @classmethod
    def new(cls , module = 'tra_lstm' , model_data_type = 'day'):
        config = TrainConfig.load(override = {'model_module' : module , 'model_data_type' : model_data_type} , makedir = False)
        model_data = ModelData(config , if_train=False)

        model_date = model_data.model_date_list[0]
        dataloader_param = model_data.get_dataloader_param('test' , model_date=model_date , param=config.model_param[0])   
        model_data.create_dataloader(*dataloader_param)

        batch_data = model_data.dataloaders['test'][0]
        net = getattr(model , f'{module}')(**config.model_param[0])
        metrics = Metrics(config.train_param['criterion']).model_update(config.model_param[0] , config)
        return cls(config , net , batch_data , model_data , metrics)

    def try_forward(self) -> None:
        if isinstance(self.batch_data.x , torch.Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')

        if hasattr(self.net , 'dynamic_data_assign'): getattr(self.net , 'dynamic_data_assign')(self.batch_data , self.model_data)
        outputs = ModelOutputs(self.net(self.batch_data.x))
        print(f'y shape is {outputs.pred().shape}')
        self.outputs = outputs
        print(f'Test Forward Success')

    def try_metrics(self) -> None:
        if not hasattr(self , 'outputs'): self.try_forward()
        label , weight = self.batch_data.y , self.batch_data.w
        penalty_kwargs = {}
        penalty_kwargs.update({'net' : self.net , 'hidden' : self.outputs.hidden() , 'label' : label})
        metrics = self.Metrics.calculate('train' , label , self.outputs.pred() , weight , self.net , penalty_kwargs)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')

class RunModel():
    '''
    Control the whole process of training:
    '''
    def __init__(self , mem_storage = True , timer = True ,  device = None , **kwargs):
        self.model_info = self.ModelInfo(config)
        self.ptimer     = ProcessTimer(timer)
        self.storage    = Storage(mem_storage)
        self.device     = Device(device)

    @dataclass
    class ModelInfo:
        config    : TrainConfig
        init_time : float = 0
        contents  : dict[str,Any] = field(default_factory=dict)

        result_path : ClassVar[str] = f'{DIR.result}/model_results.yaml'

        def __post_init__(self):
            os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
            self.init_time = time.time()

        def __setitem__(self , key , value):
            self.contents[key] = value

        def __getitem__(self , key , default = None):
            return self.contents.get(key , default)

        def dump_result(self):
            result = {
                '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
                '1_start' : time.ctime(self.init_time) ,
                '2_basic' : 'short' if self.config.short_test else 'full' , 
                '3_datas' : self.config.model_data_type ,
                '4_label' : ','.join(self.config.labels),
                '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
                '6_train' : self['train_process'],
                '7_test'  : self['test_process'],
                '8_result': self['test_score_sum'],
            }
            DIR.dump_yaml(result , self.result_path)

    def main_process(self):
        '''
        Main process of load_data + train + test
        '''
        for process_name in config.process_queue:
            self.process_setname(process_name)
            self.__getattribute__(f'process_{process_name.lower()}')()

        self.model_info.dump_result()
        self.ptimer.print()
    
    def process_setname(self , key = 'data'):
        self.process_name = key.lower()
        self.model_count = 0
        self.epoch_count = 0
        if self.process_name == 'data': 
            pass
        elif self.process_name in ['train' , 'test']: 
            self.data.reset_dataloaders()
            self.Metrics = Metrics(config.train_param['criterion'])
        else:
            raise Exception(f'KeyError : {key}')
        
    def process_data(self):
        '''
        Main process of loading basic data
        '''
        self.model_info['data_time'] = time.time()
        logger.critical(f'Start Process [Load Data]!')
        self.data = ModelData(config)

        logger.critical('Finish Process [Load Data]! Cost {:.1f}Secs'.format(time.time() - self.model_info['data_time']))
        
    def process_train(self):
        '''
        Main process of training
        1. loop over model(model_date , model_num)
        2. loop over attempt(if converge too soon) , epoch
        '''
        self.process_train_start()
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation('train')
            self.model_train()
        self.process_train_end()

    def process_test(self):
        self.process_test_start()
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation('test')
            self.model_test()
        self.process_test_end()

    def model_iter(self):
        new_iter = list(itertools.product(self.data.model_date_list , config.model_num_list))
        if config.resume_training and self.process_name == 'train':
            models_trained = np.full(len(new_iter) , True)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(f'{config.model_base_path}/{model_num}/{model_date}.best.pt'):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = FilteredIterator(new_iter , ~models_trained)
        return new_iter
    
    def process_train_start(self):
        self.model_info[f'{self.process_name}_time'] = time.time()
        logger.critical(f'Start Process [{self.process_name.capitalize()} Model]!')        
        config.Model.save(config.model_base_path)    

    def process_train_end(self):
        total_time = time.time() - self.model_info['train_time']
        self.model_info['train_process'] = \
            'Finish Process [Train Model]! Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                total_time / 3600 , total_time / 60 / max(self.model_count , 1) , total_time / max(self.epoch_count , 1))
        logger.critical(self.model_info['train_process'])
    
    def model_preparation(self , process , last_n = 30 , best_n = 5):
        assert process in ['train' , 'test']
        with self.ptimer('model_preparation' , process):
            param = config.model_param[self.model_num]

            # In a new model , alters the penalty function's lamb
            self.Metrics.model_update(param , config)

            path_prefix = '{}/{}'.format(param.get('path') , self.model_date)
            path = {'target'      : {op_type:f'{path_prefix}.{op_type}.pt' for op_type in config.output_types} , 
                    'source'      : {op_type:[] for op_type in config.output_types} , # del at each model train
                    'candidate'   : {op_type:None for op_type in (config.output_types + ['transfer'])} , # not del at each model train
                    'performance' : {op_type:None for op_type in config.output_types}}
            if 'best'    in config.output_types:
                path['candidate']['best'] = f'{path_prefix}.best.pt'
            if 'swalast' in config.output_types: 
                path['source']['swalast'] = [f'{path_prefix}.lastn.{i}.pt' for i in range(last_n)]
            if 'swabest' in config.output_types: 
                path['source']['swabest']      = [f'{path_prefix}.bestn.{i}.pt' for i in range(best_n)] 
                path['candidate']['swabest']   = path['source']['swabest']
                path['performance']['swabest'] = [-10000. for i in range(best_n)]
            if config.train_param['transfer'] and self.model_date > self.data.model_date_list[0]:
                last_model_date = max([d for d in self.data.model_date_list if d < self.model_date])
                path['candidate']['transfer'] = '{}/{}.best.pt'.format(param.get('path') , last_model_date)
                
            self.param , self.path = param , path
    
    def model_train(self):
        self.model_train_start()
        while self.cond.get('loop_status') != 'model':
            self.model_train_init_loop()
            self.model_train_init_trainer()
            self.model_train_epoch()
            self.model_train_assess_status()
        self.model_train_end()
    
    def model_test(self):
        self.model_test_start()
        self.model_forecast()
        self.model_test_end()
        
    def _init_variables(self , key = 'model'):
        '''
        Reset variables of 'model' , 'attempt' start
        '''
        if key == 'epoch' : return
        assert key in ['model' , 'attempt'] , f'KeyError : {key}'

        self.epoch_i = -1
        self.epoch_attempt_best = -1
        self.score_attempt_best = -10000.
        self.loss_list  = {'train' : [] , 'valid' : []}
        self.score_list = {'train' : [] , 'valid' : []}
        self.lr_list    = []
        
        if key in ['model']:
            self.attempt_i = -1
            self.epoch_all = -1
            self.tick = np.full((10,) , fill_value=time.time())
            self.text = {k : '' for k in ['model','attempt','epoch','exit','stat','time','trainer']}
            self.cond = {'terminate' : {} , 'nan_loss' : False , 'loop_status' : 'attempt'}

    def model_train_start(self):
        '''
        Reset model specific variables
        '''
        with self.ptimer('model_train/start'):
            self._init_variables('model')
            self.nanloss_life = config.train_param['trainer']['nanloss']['retry']
            self.text['model'] = '{:s} #{:d} @{:4d}'.format(config.model_name , self.model_num , self.model_date)
            dataloader_param = self.data.get_dataloader_param('train' , model_date = self.model_date , param=self.param)   
            if (self.data.dataloader_param != dataloader_param):
                self.data.create_dataloader(*dataloader_param) 
                self.tick[1] = time.time()
                self.print_progress('train_dataloader')
            
    def model_train_end(self):
        '''
        Do necessary things of ending a model(model_data , model_num)
        '''
        with self.ptimer('model_train/end'):
            self.storage.del_path([p for pl in self.path['source'].values() for p in pl])
            if self.process_name == 'train' : self.model_count += 1
            self.tick[2] = time.time()
            self.print_progress('model_end')
            gc.collect() 
            torch.cuda.empty_cache()

    def model_train_init_loop(self):
        '''
        Reset and loop variables giving loop_status
        '''
        with self.ptimer('model_train/init_loop'):
            self._init_variables(self.cond.get('loop_status' , ''))
            self.epoch_i += 1
            self.epoch_all += 1
            self.epoch_count += 1
            if self.cond.get('loop_status') in ['attempt']:
                self.attempt_i += 1
                self.text['attempt'] = f'FirstBite' if self.attempt_i == 0 else f'Retrain#{self.attempt_i}'
        
    def model_train_init_trainer(self):
        '''
        Initialize net , optimizer , scheduler if loop_status in ['attempt']
        net : 1. Create an model of f'{config.model_module}' or inherit from 'transfer'
              2. In transfer mode , p_late and p_early with be trained with different lr's. If not net.parameters are trained by same lr
        optimizer : Adam or SGD
        scheduler : Cosine or StepLR
        '''
        with self.ptimer('model_train/init_trainer'):
            if self.cond.get('loop_status') == 'epoch': return
            self.load_model('train')
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler() 

    def model_train_epoch(self):
        '''
        Iterate train and valid dataset, calculate loss/score , update values
        If nan loss occurs, turn to nanloss_reviver
        '''
        with self.ptimer('model_train/epoch/train'):
            self.net.train() 
            iterator = self.data.dataloaders['train']
            list_loss  = torch.full([len(iterator)] , fill_value=torch.nan)
            list_score = torch.full([len(iterator)] , fill_value=torch.nan)
            for i , batch_data in enumerate(iterator):
                
                outputs = self.model_forward('train' , batch_data)
                metrics = self.model_metric('train' , outputs , batch_data)
                if metrics.loss.isnan():
                    print(i , batch_data.y)
                    print(metrics)
                    raise Exception('here')
                self.model_backward(metrics.loss)

                list_loss[i] , list_score[i] = metrics.loss_item , metrics.score
                iterator.display(f'Ep#{self.epoch_i:3d} train loss:{list_loss[:i+1].mean():.5f}')
            if list_loss.isnan().any(): return self.nanloss_reviver()
            self.loss_list['train'].append(list_loss.mean()) 
            self.score_list['train'].append(list_score.mean())
        
        with self.ptimer('model_train/epoch/valid') , torch.no_grad():
            self.net.eval()  
            iterator = self.data.dataloaders['valid']
            list_loss  = torch.full([len(iterator)] , fill_value=torch.nan)
            list_score = torch.full([len(iterator)] , fill_value=torch.nan)
            for i , batch_data in enumerate(iterator):
                # self.device.print_cuda_memory()
                outputs = self.model_forward('valid' , batch_data)
                metrics = self.model_metric('valid' , outputs , batch_data)

                list_loss[i] , list_score[i] = metrics.loss_item , metrics.score
                iterator.display(f'Ep#{self.epoch_i:3d} valid ic:{list_score[:i+1].mean():.5f}')
            self.loss_list['valid'].append(list_loss.mean()) 
            self.score_list['valid'].append(list_score.mean())

        self.lr_list.append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()
        self.reset_scheduler()

    def model_train_assess_status(self):
        '''
        Update condition of continuing training epochs , restart attempt if early exit or nan loss
        '''
        with self.ptimer('model_train/assess'):
            if self.cond['nan_loss']:
                logger.error(f'Initialize a new model to retrain! Lives remaining {self.nanloss_life}')
                self._init_variables('model')
                self.cond['loop_status'] = 'attempt'
                return
                
            valid_score = self.score_list['valid'][-1]
            
            save_targets = [] 
            if valid_score > self.score_attempt_best: 
                self.epoch_attempt_best = self.epoch_i 
                self.score_attempt_best = valid_score
                save_targets.append(self.path['target']['best'])

            if 'swalast' in config.output_types:
                self.path['source']['swalast'] = self.path['source']['swalast'][1:] + self.path['source']['swalast'][:1]
                save_targets.append(self.path['source']['swalast'][-1])
                
                p_valid = self.path['source']['swalast'][-len(self.score_list['valid']):]
                arg_max = np.argmax(self.score_list['valid'][-len(p_valid):])
                arg_swa = (lambda x:x[(x>=0) & (x<len(p_valid))])(min(3,len(p_valid)//3)*np.arange(-5,3)+arg_max)[-5:]
                self.path['candidate']['swalast'] = [p_valid[i] for i in arg_swa]
                
            if 'swabest' in config.output_types:
                arg_min = np.argmin(self.path['performance']['swabest'])
                if valid_score > self.path['performance']['swabest'][arg_min]:
                    self.path['performance']['swabest'][arg_min] = valid_score
                    save_targets.append(self.path['candidate']['swabest'][arg_min])
                
            self.save_model(paths = save_targets)
            self.print_progress('epoch_step')
        
        with self.ptimer('model_train/status'):
            self.text['exit'] , self.cond['terminate'] = self.check_train_end() 
            if self.text['exit']:
                if (self.epoch_i < config.train_param['trainer']['retrain']['min_epoch'] - 1 and 
                    self.attempt_i < config.train_param['trainer']['retrain']['attempts'] - 1):
                    self.cond['loop_status'] = 'attempt'
                    self.print_progress('new_attempt')
                else:
                    self.cond['loop_status'] = 'model'
                    # print(self.net.get_probs())
                    self.save_model(disk_key = config.output_types)
            else:
                self.cond['loop_status'] = 'epoch'

    def model_test_start(self):
        '''
        Reset model specific variables
        '''
        with self.ptimer('model_test/start'):
            self._init_variables('model')
            dataloader_param = self.data.get_dataloader_param('test' , model_date = self.model_date , param=self.param)   
            self.data.create_dataloader(*dataloader_param)
                
            if self.model_num == 0:
                score_date  = np.zeros((len(self.data.model_test_dates) , 
                                        len(self.test_model_num)))
                score_model = np.zeros((1 , len(self.test_model_num)))
                self.score_by_date  = np.concatenate([getattr(self,'score_by_date' ,np.empty((0,len(self.test_model_num)))) , score_date])
                self.score_by_model = np.concatenate([getattr(self,'score_by_model',np.empty((0,len(self.test_model_num)))) , score_model])
                
    def model_forecast(self):
        if not os.path.exists(self.path['target']['best']): self.model_train()
        
        with self.ptimer('model_test/forecast') , torch.no_grad():
            iter_dates = np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
            l0 , l1 = len(self.data.early_test_dates) , len(self.data.model_test_dates)
            assert self.data.dataloaders['test'].__len__() == len(iter_dates)
            for oi , op_type in enumerate(config.output_types):
                self.load_model('test' , op_type)
                self.net.eval() 
                iterator = self.data.dataloaders['test']
                test_score = np.full(len(iter_dates),np.nan)
                for i , batch_data in enumerate(iterator):
                    if len(torch.where(batch_data.nonnan)[0]) == 0: continue
                    outputs = self.model_forward('test' , batch_data)

                    assert not outputs.pred().isnan().any()
                    assert iter_dates[i] == self.data.y_date[batch_data.i[0,1]] , (iter_dates[i] , self.data.y_date[batch_data.i[0,1]])
                    
                    if i < l0: continue # before this date is warmup stage
                    metrics = self.model_metric('test' , outputs, batch_data)
                    test_score[i] = metrics.score

                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                    iterator.display(f'Date {iter_dates[i]}:{test_score[l0:i+1].mean():.5f}')
                self.score_by_date[-l1:,self.model_num*len(config.output_types) + oi] = np.nan_to_num(test_score[-l1:])
        
    def model_test_end(self):
        '''
        Do necessary things of ending a model(model_data , model_num)
        '''
        with self.ptimer('model_test/end'):
            if self.model_num == config.model_num_list[-1]:
                self.score_by_model[-1,:] = np.nanmean(self.score_by_date[-len(self.data.model_test_dates):,],axis = 0)
                self.print_test_table(self.model_date , self.score_by_model[-1,:] , 4)
            gc.collect() 
            torch.cuda.empty_cache()
  
    def process_test_start(self):
        self.model_info[f'{self.process_name}_time'] = time.time()
        logger.critical(f'Start Process [{self.process_name.capitalize()} Model]!')        
        logger.warning('Each Model Date Testing Mean Score({}):'.format(config.train_param['criterion']['score']))

        self.test_model_num = np.repeat(config.model_num_list,len(config.output_types))
        self.test_output_type = np.tile(config.output_types,len(config.model_num_list))
        self.score_by_date  = np.empty((0,len(self.test_model_num)))
        self.score_by_model = np.empty((0,len(self.test_model_num)))

        self.print_test_table('Models' , self.test_model_num , 0)
        self.print_test_table('Output' , self.test_output_type)

    def process_test_end(self):
        # date ic writed down
        date_step = self.data.config.test_step_day
        date_list = self.data.test_full_dates[::date_step]
        for model_num in config.model_num_list:
            date_str = np.array(list(map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , date_list.astype(str))))
            df = pd.DataFrame({'dates' : date_list} , index = date_str)
            # print(self.score_by_date.shape)
            for oi , op_type in enumerate(config.output_types):
                df[f'score.{op_type}'] = self.score_by_date[:,model_num*len(config.output_types) + oi]
                df[f'cum_score.{op_type}'] = np.nancumsum(self.score_by_date[:,model_num*len(config.output_types) + oi])
            df.to_csv(config.model_param[model_num]['path'] + f'/{config.model_name}_score_by_date_{model_num}.csv')

        # model ic presentation
        add_row_key   = ['AllTimeAvg' , 'AllTimeSum' , 'Std'      , 'TValue'   , 'AnnIR']
        score_mean   = np.nanmean(self.score_by_date , axis = 0)
        score_sum    = np.nansum(self.score_by_date  , axis = 0) 
        score_std    = np.nanstd(self.score_by_date  , axis = 0)
        score_tvalue = score_mean / score_std * (len(self.score_by_date)**0.5) # 10 days return predicted
        score_annir  = score_mean / score_std * ((240 / 10)**0.5) # 10 days return predicted
        add_row_value = (score_mean , score_sum , score_std , score_tvalue , score_annir)
        df = pd.DataFrame(np.concatenate([self.score_by_model , np.stack(add_row_value)]) , 
                          index   = [str(d) for d in self.data.model_date_list] + add_row_key , 
                          columns = [f'{mn}_{o}' for mn,o in zip(self.test_model_num , self.test_output_type)])
        df.to_csv(f'{config.model_base_path}/{config.model_name}_score_by_model.csv')
        for i , digits in enumerate([4,2,4,2,4]):
            self.print_test_table(add_row_key[i] , add_row_value[i] , digits)
        self.model_info['test_score_sum'] = {k:round(v,4) for k,v in zip(df.columns , score_sum.tolist())}  

        total_time = time.time() - self.model_info['train_time']
        self.model_info['test_process'] = 'Finish Process [Test Model]! Cost {:.1f} Secs'.format(total_time)
        logger.critical(self.model_info['test_process'])

    def model_forward(self , key , batch_data : BatchData):
        with self.ptimer(f'{key}/forward'):
            if hasattr(self.net , 'dynamic_data_assign'): getattr(self.net , 'dynamic_data_assign')(batch_data , self.data)
            return ModelOutputs(self.net(batch_data.x))

    def model_metric(self, key , outputs_of_net : ModelOutputs, batch_data : BatchData , **kwargs) -> Metrics.MetricOutput:
        # outputs_of_net = self.net(batch_data.x)
        # valid_sample = torch.where(batch_data.nonnan)[0]
        with self.ptimer(f'{key}/loss'):
            label , weight = batch_data.y , batch_data.w
            penalty_kwargs = {}
            if key == 'train': penalty_kwargs.update({'net' : self.net , 'hidden' : outputs_of_net.hidden() , 'label' : label})
            return self.Metrics.calculate(key , label , outputs_of_net.pred() , weight , self.net , penalty_kwargs , **kwargs)
        
    def model_backward(self, loss : torch.Tensor) -> None:
        clip_value = config.train_param['trainer']['gradient'].get('clip_value')
        with self.ptimer(f'train/backward'):
            self.optimizer.zero_grad()
            loss.backward()
            if clip_value is not None : clip_grad_value_(self.net.parameters(), clip_value = clip_value) 
            self.optimizer.step()
    
    def print_progress(self , key):
        '''
        Print out status giving display conditions and looping conditions
        '''
        printers = [logger.info] if (config.verbosity > 2 or self.model_count < config.model_num) else [logger.debug]
        sdout   = None
        if key == 'model_end':
            self.text['epoch'] = 'Ep#{:3d}'.format(self.epoch_all)
            self.text['stat']  = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
                self.score_list['train'][-1],self.score_list['valid'][-1],self.score_attempt_best)
            self.text['time']  = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
                (self.tick[2]-self.tick[0])/60 , (self.tick[2]-self.tick[1])/(self.epoch_all+1))
            sdout = self.text['model'] + '|' + self.text['attempt'] + ' ' + \
                    self.text['epoch'] + ' ' + self.text['exit'] + '|' + self.text['stat'] + '|' + self.text['time']
            printers = [logger.warning]
        elif key == 'epoch_step':
            self.text['trainer'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, max{: .4f}, best{: .4f}, lr{:.1e}'.format(
                self.loss_list['train'][-1] , self.score_list['train'][-1] , self.score_list['valid'][-1] , 
                self.score_attempt_best , self.score_attempt_best , self.lr_list[-1])
            if self.epoch_i % [10,5,5,3,3,1][min(config.verbosity // 2 , 5)] == 0:
                sdout = ' '.join([self.text['attempt'],'Ep#{:3d}'.format(self.epoch_i),':', self.text['trainer']])
        elif key == 'reset_learn_rate':
            sdout = 'Reset learn rate and scheduler at the end of epoch {} , effective at epoch {}'.format(
                self.epoch_i , self.epoch_i+1 , ', and will speedup2x' * config.train_param['trainer']['learn_rate']['reset']['speedup2x'])
        elif key == 'new_attempt':
            sdout = ' '.join([self.text['attempt'],'Epoch #{:3d}'.format(self.epoch_i),':',self.text['trainer'],', Next attempt goes!'])
        elif key == 'train_dataloader':
            sdout = ' '.join([self.text['model'],'LoadData Cost {:>6.1f}Secs'.format(self.tick[1]-self.tick[0])])  
        else:
            raise Exception(f'KeyError : {key}')
        
        if sdout is not None: [prt(sdout) for prt in printers]

    def print_test_table(self , rowname , values , digits = 2):
        fmt = 's' if isinstance(values[0] , str) else (f'd' if digits == 0 else f'.{digits}f')
        logger.info(('{: <11s}'+('{: >8'+fmt+'}')*len(values)).format(str(rowname) , *values))

    def nanloss_reviver(self):
        '''
        Deal with nan loss, life -1 and change nan_loss condition to True
        '''
        logger.error(f'{self.text["model"]} Attempt{self.attempt_i}, epoch{self.epoch_i} got nan loss!')
        if self.nanloss_life > 0:
            self.nanloss_life -= 1
            self.cond['nan_loss'] = True
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')

    def check_train_end(self):
        '''
        Whether terminate condition meets
        '''
        term_dict = config.train_param['terminate']
        term_cond = {}
        exit_text = ''
        for key , arg in term_dict.items():
            if key == 'max_epoch':
                term_cond[key] = self.epoch_i >= min(arg , config.max_epoch) - 1
                if term_cond[key] and exit_text == '': exit_text = 'Max Epoch'
            elif key == 'early_stop':
                term_cond[key] = self.epoch_i - self.epoch_attempt_best >= arg
                if term_cond[key] and exit_text == '': exit_text = 'EarlyStop'
            elif key == 'tv_converge':
                term_cond[key] = (list_converge(self.loss_list['train']  , arg.get('min_epoch') , arg.get('eps')) and
                                  list_converge(self.score_list['valid'] , arg.get('min_epoch') , arg.get('eps')))
                if term_cond[key] and exit_text == '': exit_text = 'T & V Cvg'
            elif key == 'train_converge':
                term_cond[key] = list_converge(self.loss_list['train']  , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text == '': exit_text = 'Train Cvg'
            elif key == 'valid_converge':
                term_cond[key] = list_converge(self.score_list['valid'] , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text == '': exit_text = 'Valid Cvg'
            else:
                raise Exception(f'KeyError : {key}')

        return exit_text , term_cond
    
    def save_model(self , paths = None , disk_key = None , savable_net = None):
        if paths is None and disk_key is None: return NotImplemented # nothing to save

        with self.ptimer('save_model'):
            if paths is not None:
                savable_net = self.net
                if hasattr(savable_net,'dynamic_data_unlink'):
                    # remove unsavable part
                    savable_net = getattr(savable_net , 'dynamic_data_unlink')()
                self.storage.save_model_state(savable_net , paths)

            if disk_key is not None:
                if isinstance(disk_key , str): disk_key = [disk_key]
                for key in disk_key:
                    assert key in ['best' , 'swalast' , 'swabest']
                    p_exists = self.storage.valid_paths(self.path['candidate'][key])
                    if len(p_exists) == 0: print(key , self.path['candidate'][key] , self.path['performance'][key])
                    if key == 'best':
                        savable_net = self.storage.load(p_exists[0])
                    else:
                        savable_net = self.load_swa_model(p_exists)
                    self.storage.save_model_state(savable_net , self.path['target'][key] , to_disk = True) 
    
    def load_model(self , process , key = 'best'):
        assert process in ['train' , 'test']
        with self.ptimer('load_model'):
            if process == 'train':           
                if self.path['candidate'].get('transfer'):
                    if not config.train_param['transfer']: raise Exception('get transfer')
                    model_path = self.path['candidate']['transfer']
                else:
                    model_path = -1
            else:
                model_path = self.path['target'][key]

            net = self.new_model(config.model_module , self.param , self.storage.load(model_path , from_disk = True))
            self.net = self.device(net)
            
    def load_swa_model(self , model_path_list = []):
        if len(model_path_list) == 0: raise Exception('empty swa input')
        net = self.new_model(config.model_module , self.param)
        swa_net = self.device(AveragedModel(net))
        for p in model_path_list:
            swa_net.update_parameters(self.storage.load_model_state(net , p)) 
        update_bn(self._swa_update_bn_loader(self.data.dataloaders['train']) , swa_net) 
        return swa_net.module 
    
    def _swa_update_bn_loader(self , loader):
        for batch_data in loader: 
            yield (batch_data.x , batch_data.y , batch_data.w)
    
    def load_optimizer(self , new_opt_kwargs = None , new_lr_kwargs = None):
        if new_opt_kwargs is None:
            opt_kwargs = config.train_param['trainer']['optimizer']
        else:
            opt_kwargs = deepcopy(config.train_param['trainer']['optimizer'])
            opt_kwargs.update(new_opt_kwargs)
        
        if new_lr_kwargs is None:
            lr_kwargs = config.train_param['trainer']['learn_rate']
        else:
            lr_kwargs = deepcopy(config.train_param['trainer']['learn_rate'])
            lr_kwargs.update(new_lr_kwargs)
        base_lr = lr_kwargs['base'] * lr_kwargs['ratio']['attempt'][:self.attempt_i+1][-1]
        
        return self.new_optimizer(self.net , opt_kwargs['name'] , base_lr , transfer = self.path['candidate'].get('transfer') , 
                                  encoder_lr_ratio = lr_kwargs['ratio']['transfer'], **opt_kwargs['param'])
    
    def load_scheduler(self , new_shd_kwargs = None):
        if new_shd_kwargs is None:
            shd_kwargs = config.train_param['trainer']['scheduler']
        else:
            shd_kwargs = deepcopy(config.train_param['trainer']['scheduler'])
            shd_kwargs.update(new_shd_kwargs)
        return self.new_scheduler(self.optimizer, shd_kwargs['name'], **shd_kwargs['param'])
    
    def reset_scheduler(self):
        rst_kwargs = config.train_param['trainer']['learn_rate']['reset']
        if rst_kwargs['num_reset'] <= 0 or (self.epoch_i + 1) < rst_kwargs['trigger']: return

        trigger_intvl = rst_kwargs['trigger'] // 2 if rst_kwargs['speedup2x'] else rst_kwargs['trigger']
        if (self.epoch_i + 1 - rst_kwargs['trigger']) % trigger_intvl != 0: return
        
        trigger_times = ((self.epoch_i + 1 - rst_kwargs['trigger']) // trigger_intvl) + 1
        if trigger_times > rst_kwargs['num_reset']: return
        
        # confirm reset : change back optimizor learn rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr_param']  * rst_kwargs['recover_level']
        
        # confirm reset : reassign scheduler
        if rst_kwargs['speedup2x']:
            shd_kwargs = deepcopy(config.train_param['trainer']['scheduler'])
            for k in np.intersect1d(list(shd_kwargs['param'].keys()),['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']):
                shd_kwargs['param'][k] //= 2
        else:
            shd_kwargs = None
        self.scheduler = self.load_scheduler(shd_kwargs)
        self.print_progress('reset_learn_rate')
    
    @staticmethod
    def new_model(module , param = {} , state_dict = None , **kwargs):
        net = getattr(model , f'{module}')(**param)
        if state_dict: net.load_state_dict(state_dict)
        return net

    @staticmethod
    def new_scheduler(optimizer, key = 'cycle', **kwargs):
        if key == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif key == 'cycle':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=[pg['lr_param'] for pg in optimizer.param_groups],cycle_momentum=False,mode='triangular2',**kwargs)
        return scheduler

    @staticmethod
    def new_optimizer(net , key ='Adam', base_lr = 0.005, transfer = False , encoder_lr_ratio = 1., decoder_lr_ratio = 1., **kwargs):
        if transfer:
            # define param list to train with different learn rate
            p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in net.named_parameters() if 'encoder' in x.split('.')[:3]]
            p_dec = [p for x,p in net.named_parameters() if 'encoder' not in x.split('.')[:3]]
            net_param_groups = [{'params': p_dec , 'lr': base_lr * decoder_lr_ratio , 'lr_param': base_lr * decoder_lr_ratio},
                                {'params': p_enc , 'lr': base_lr * encoder_lr_ratio , 'lr_param': base_lr * encoder_lr_ratio}]
        else:
            net_param_groups = [{'params': [p for p in net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

        optimizer = {
            'Adam': torch.optim.Adam ,
            'SGD' : torch.optim.SGD ,
        }[key](net_param_groups , **kwargs)
        return optimizer


def main(process = -1 , resume = -1 , checkname = -1 , timer = False , parser_args = None):
    if parser_args is None: parser_args = config.parser_args({'process':process,'resume':resume,'checkname':checkname})

    config.reload(do_process=True,par_args=parser_args)
    config.set_config_environment()

    logger.warning('Model Specifics:')
    config.print_out()

    app = RunModel(mem_storage = config.mem_storage , timer = timer)
    app.main_process()

if __name__ == '__main__':
    main(parser_args = config.parser_args())
 