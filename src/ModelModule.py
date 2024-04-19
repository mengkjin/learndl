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
import itertools , os , gc

from copy import deepcopy
from dataclasses import dataclass , field
from torch import nn , Tensor
from typing import Any , ClassVar , Literal , Optional

from .environ import DIR

from .data.DataFetcher import DataFetcher
from .DataModule import BatchData , DataModule
from .util import Device , Metrics , Logger , Storage , TrainConfig
from .util import trainer as TRAIN # TrainerUtil
from .util import optim as OPTIM # TrainerUtil
from .model import model as MODEL

from .func.date import today , date_offset
from .func.basic import list_converge


# from audtorch.metrics.functional import *

logger = Logger()
config = TrainConfig()

@dataclass
class Predictor:
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

        model_data_old = DataModule(model_config , predict = False) if require_model_data_old else None
        model_data_new = DataModule(model_config , predict = True if old_model_data else False) 

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

                    net = MODEL.new(model_config.model_module , model_param , model_sd)
                    net = device(net)

                    net.eval()

                    loader = model_data.dataloaders['test']
                    secid  = model_data.datas.secid
                    tdates = model_data.model_test_dates
                    #print(df_sub.loc[df_sub['calculated'] == 0 , 'pred_dates'])
                    #print(tdates)
                    iter_tdates = np.intersect1d(df_sub.loc[df_sub['calculated'] == 0 , 'pred_dates'] , tdates)

                    for tdate in iter_tdates:
                        batch_data = loader[np.where(tdates == tdate)[0][0]]

                        pred = TRAIN.Output(net(batch_data.x)).pred()
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
class ModelTestor:
    config : TrainConfig
    net    : nn.Module
    batch_data : BatchData
    model_data : DataModule
    Metrics : Metrics

    @classmethod
    def new(cls , module = 'tra_lstm' , model_data_type = 'day'):
        config = TrainConfig.load(override = {'model_module' : module , 'model_data_type' : model_data_type} , makedir = False)
        model_data = DataModule(config , predict=True)

        model_date = model_data.model_date_list[0]
        model_data.setup('test' , model_date , config.model_param[0])   
        
        batch_data = model_data.test_dataloader()[0]

        net = MODEL.new(module , config.model_param[0])
        metrics = Metrics(config.train_param['criterion']).model_update(config.model_param[0] , config)
        return cls(config , net , batch_data , model_data , metrics)

    def try_forward(self) -> None:
        if isinstance(self.batch_data.x , torch.Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')

        if hasattr(self.net , 'dynamic_data_assign'): getattr(self.net , 'dynamic_data_assign')(self.batch_data , self.model_data)
        outputs = TRAIN.Output(self.net(self.batch_data.x))
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

class ModelTrainer():
    '''Control the whole process of training'''
    default_model_type = 'best'

    def __init__(self , mem_storage = True , timer = True ,  device = None , **kwargs):
        self.info           = TRAIN.Info(config)
        self.pipe           = TRAIN.Pipeline(config , logger)
        self.ptimer         = TRAIN.PTimer(timer)
        self.checkpoints    = Storage(mem_storage) # state_dict checkpoints
        self.storage        = Storage(False)       # state_dict hard save
        self.device         = Device(device)

    def main_process(self):
        '''Main stage of load_data + fit + test'''
        for stage in config.stage_queue:
            self.set_stage(stage)
            getattr(self , f'process_{stage}')()
        self.info.dump_result()
        self.ptimer.print()
    
    def set_stage(self , stage : Literal['data' , 'fit' , 'test']):
        self.stage = stage
        self.info.new_count()
        if stage in ['fit' , 'test']: 
            self.data.reset_dataloaders()
            self.Metrics = Metrics(config.train_param['criterion'])
        self.pipe.new_stage(stage)
        
    def process_data(self):
        '''Main process of loading model data'''
        logger.critical(self.info.tic_str('data'))
        self.data = DataModule(config)
        logger.critical(self.info.toc_str('data'))
        
    def process_fit(self):
        '''Main process of fitting'''
        self.process_fit_start()
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation()
            self.model_fit()
        self.process_fit_end()

    def process_test(self):
        '''Main process of testing'''
        self.process_test_start()
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation()
            self.model_test()
        self.process_test_end()

    def model_iter(self):
        new_iter = list(itertools.product(self.data.model_date_list , config.model_num_list))
        if config.resume_training and self.stage == 'fit':
            models_trained = np.full(len(new_iter) , True)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(f'{config.model_base_path}/{model_num}/{model_date}.{self.default_model_type}.pt'):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = TRAIN.Filtered(new_iter , ~models_trained)
        return new_iter
    
    def process_fit_start(self):
        logger.critical(self.info.tic_str('fit'))
        config.Model.save(config.model_base_path)    

    def process_fit_end(self):
        logger.critical(self.info.toc_str('fit' , True))
    
    def model_preparation(self , last_n = 30 , best_n = 5):
        with self.ptimer(f'{self.stage}/prepare' , self.stage):
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
    
    def model_fit(self):
        self.model_fit_start()
        while self.pipe.loop_status != 'model':
            self.model_fit_init_loop()
            self.model_fit_init_trainer()
            self.model_fit_epoch()
            self.model_fit_assess_status()
        self.model_fit_end()
    
    def model_test(self):
        self.model_test_start()
        self.model_forecast()
        self.model_test_end()

    def model_fit_start(self):
        '''Reset model specific variables'''
        with self.ptimer(f'{self.stage}/start'):
            self.info.tic('model')
            self.pipe.texts['model'] = '{:s} #{:d} @{:4d}'.format(config.model_name , self.model_num , self.model_date)
            self.data.setup('fit' , self.model_date , self.param)
            
    def model_fit_end(self):
        with self.ptimer(f'{self.stage}/end'):
            self.checkpoints.del_all()
            # self.storage.del_path([p for pl in self.path['source'].values() for p in pl])
            if self.stage == 'fit' : self.info.add_model()
            self.print_progress('model_end')
            gc.collect() 
            torch.cuda.empty_cache()

    def model_fit_init_loop(self):
        '''Reset and loop variables giving loop_status'''
        with self.ptimer(f'{self.stage}/init_loop'):
            self.pipe.new_loop()
            self.info.add_epoch()
        
    def model_fit_init_trainer(self):
        '''Initialize net , optimizer(scheduler)'''
        with self.ptimer(f'{self.stage}/init_trainer'):
            if self.pipe.loop_status == 'epoch': return
            self.net = self.load_model(True)
            self.optimizer = OPTIM.Optimizer(self.net , config , bool(self.path['candidate'].get('transfer')) , self.pipe.attempt_i)

    def model_fit_epoch(self):
        '''Iterate train and valid dataset, calculate loss/score , update values'''
        with self.ptimer(f'{self.stage}/train_loader'):
            self.loop_dataloader('train')
        
        with self.ptimer(f'{self.stage}/valid_loader') , torch.no_grad():
            self.loop_dataloader('valid')

        if self.optimizer.step(self.pipe.epoch_i): self.print_progress('reset_learn_rate')

    def model_fit_assess_status(self):
        '''Update condition of continuing training epochs , restart attempt if early exit or nan_loss'''
        if self.pipe.nanloss: return
        with self.ptimer(f'{self.stage}/assess'):                
            valid_score = self.pipe.score_list['valid'][-1]
            
            save_targets = [] 
            if valid_score > self.pipe.score_attempt_best: 
                self.pipe.epoch_attempt_best = self.pipe.epoch_i 
                self.pipe.score_attempt_best = valid_score
                save_targets.append(self.path['target']['best'])

            if 'swalast' in config.output_types:
                self.path['source']['swalast'] = self.path['source']['swalast'][1:] + self.path['source']['swalast'][:1]
                save_targets.append(self.path['source']['swalast'][-1])
                
                p_valid = self.path['source']['swalast'][-len(self.pipe.score_list['valid']):]
                arg_max = np.argmax(self.pipe.score_list['valid'][-len(p_valid):])
                arg_swa = (lambda x:x[(x>=0) & (x<len(p_valid))])(min(3,len(p_valid)//3)*np.arange(-5,3)+arg_max)[-5:]
                self.path['candidate']['swalast'] = [p_valid[i] for i in arg_swa]
                
            if 'swabest' in config.output_types:
                arg_min = np.argmin(self.path['performance']['swabest'])
                if valid_score > self.path['performance']['swabest'][arg_min]:
                    self.path['performance']['swabest'][arg_min] = valid_score
                    save_targets.append(self.path['candidate']['swabest'][arg_min])
                
            self.save_model(paths = save_targets)
            self.print_progress('epoch_step')
        
        with self.ptimer(f'{self.stage}/status'):
            self.pipe.texts['exit'] , self.pipe.conds = self.check_train_end() 
            if self.pipe.texts['exit']:
                if (self.pipe.epoch_i < config.train_param['trainer']['retrain']['min_epoch'] - 1 and 
                    self.pipe.attempt_i < config.train_param['trainer']['retrain']['attempts'] - 1):
                    self.pipe.loop_status = 'attempt'
                    self.print_progress('new_attempt')
                else:
                    self.pipe.loop_status = 'model'
                    # print(self.net.get_probs())
                    self.save_model(disk_key = config.output_types)
            else:
                self.pipe.loop_status = 'epoch'

    def model_test_start(self):
        '''Reset model specific variables'''
        with self.ptimer(f'{self.stage}/start'):
            self.pipe.new_model()
            self.data.setup('test' , self.model_date , self.param)   
                
            if self.model_num == 0:
                score_date  = np.zeros((len(self.data.model_test_dates) , 
                                        len(self.test_model_num)))
                score_model = np.zeros((1 , len(self.test_model_num)))
                self.score_by_date  = np.concatenate([getattr(self,'score_by_date' ,np.empty((0,len(self.test_model_num)))) , score_date])
                self.score_by_model = np.concatenate([getattr(self,'score_by_model',np.empty((0,len(self.test_model_num)))) , score_model])
                
    def model_forecast(self):
        if not os.path.exists(self.path['target']['best']): self.model_fit()
        
        with self.ptimer(f'{self.stage}/forecast') , torch.no_grad():
            batch_date = np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
            l0 , l1 = len(self.data.early_test_dates) , len(self.data.model_test_dates)
            assert len(self.data.test_dataloader()) == len(batch_date) , (len(self.data.test_dataloader()) , len(batch_date))
            for i , self.model_type in enumerate(config.output_types):
                self.net = self.load_model(False , self.model_type)
                self.loop_dataloader('test' , l0 , batch_date)
                '''
                self.net.eval() 
                self.data_loader = self.data.test_dataloader()
                for i , self.batch_data in enumerate(self.data_loader):
                    if self.batch_data.is_empty: continue
                    assert batch_date[i] == self.data.y_date[self.batch_data.i[0,1]] , (batch_date[i] , self.data.y_date[self.batch_data.i[0,1]])
                    self.model_forward('test')
                    if i < l0: continue # before this date is warmup stage
                    self.model_metric('test')
                    self.data_loader.record(self.Metrics , f'{batch_date[i]}test/score')
                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                '''
 
                self.score_by_date[-l1:,self.model_num*len(config.output_types) + i] = np.nan_to_num(self.pipe.agg_metric.score)

    def loop_dataloader(self , dataset : Literal['train' , 'valid' , 'test'] , 
                        warm_up : int = 0 , batch_date : Optional[np.ndarray] = None):

        self.net.train() if dataset == 'train' else self.net.eval()

        if dataset == 'train':
            self.data_loader = self.data.train_dataloader()
            display = f'Train Ep#{self.pipe.epoch_i:3d} loss'
        elif dataset == 'valid':
            self.data_loader = self.data.val_dataloader()
            display = f'Valid Ep#{self.pipe.epoch_i:3d} score'
        else:
            self.data_loader = self.data.test_dataloader()
            display = 'Test Date {} score'

        self.pipe.agg_metric.new(dataset,self.model_date,self.model_num,getattr(self.pipe,'epoch_i',0),getattr(self,'model_type','best'))
        for i , self.batch_data in enumerate(self.data_loader):
            if self.batch_data.is_empty: continue
            self.model_forward(dataset)

            # before this is warmup stage , only forward
            if i < warm_up: continue 
            self.model_metric(dataset)

            if batch_date is not None:
                assert batch_date[i] == self.data.y_date[self.batch_data.i[0,1]] , \
                    (batch_date[i] , self.data.y_date[self.batch_data.i[0,1]])
                
            self.pipe.agg_metric.record(self.Metrics)
            
            text = display.format(batch_date[i]) if batch_date is not None else display
            value = self.pipe.agg_metric.loss if dataset == 'train' else self.pipe.agg_metric.score
            self.data_loader.display(f'{text} : {value:.5f}')

            if dataset == 'train': self.model_backward()
            if (i + 1) % 20 == 0 : torch.cuda.empty_cache()

        self.pipe.check_nanloss(self.pipe.agg_metric.nanloss)
        self.pipe.agg_metric.collect()
        self.pipe.loss_list[dataset].append(self.pipe.agg_metric.loss) 
        self.pipe.score_list[dataset].append(self.pipe.agg_metric.score)
        if dataset == 'train': self.pipe.lr_list.append(self.optimizer.last_lr)

    def model_test_end(self):
        '''Do necessary things of ending a model(model_data , model_num)'''
        with self.ptimer(f'{self.stage}/end'):
            if self.model_num == config.model_num_list[-1]:
                self.score_by_model[-1,:] = np.nanmean(self.score_by_date[-len(self.data.model_test_dates):,],axis = 0)
                self.print_test_table(self.model_date , self.score_by_model[-1,:] , 4)
            gc.collect() 
            torch.cuda.empty_cache()
  
    def process_test_start(self):
        logger.critical(self.info.tic_str('test'))

        self.test_model_num = np.repeat(config.model_num_list,len(config.output_types))
        self.test_output_type = np.tile(config.output_types,len(config.model_num_list))
        self.score_by_date  = np.empty((0,len(self.test_model_num)))
        self.score_by_model = np.empty((0,len(self.test_model_num)))

        logger.warning('Each Model Date Testing Mean Score({}):'.format(config.train_param['criterion']['score']))
        self.print_test_table('Models' , self.test_model_num , 0)
        self.print_test_table('Output' , self.test_output_type)

    def process_test_end(self):
        # date ic writed down
        for model_num in config.model_num_list:
            date_str = np.array(list(map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , self.data.test_full_dates.astype(str))))
            df = pd.DataFrame({'dates' : self.data.test_full_dates} , index = date_str)

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
        self.info.add_data('test_score_sum' , {k:round(v,4) for k,v in zip(df.columns , score_sum.tolist())})
        logger.critical(self.info.toc_str('test'))

    def model_forward(self , dataset : Literal['train' , 'valid' , 'test']) -> None:
        with self.ptimer(f'{self.stage}/{dataset}/forward'):
            if hasattr(self.net , 'dynamic_data_assign'): getattr(self.net , 'dynamic_data_assign')(self.batch_data , self.data)
            self.net_output = TRAIN.Output(self.net(self.batch_data.x))

    def model_metric(self, dataset : Literal['train' , 'valid' , 'test']) -> None:
        with self.ptimer(f'{self.stage}/{dataset}/loss'):
            self.Metrics.calculate(dataset , self.batch_data.y , self.net_output.pred() , self.batch_data.w , self.net , self.penalty_kwargs , assert_nan = True)
        
    @property
    def penalty_kwargs(self) -> dict:
        return {'net' : self.net , 'hidden' : self.net_output.hidden() , 'label' : self.batch_data.y}

    def model_backward(self) -> None:
        with self.ptimer(f'{self.stage}/train/backward'): 
            self.optimizer.backward(self.Metrics.loss)
    
    def print_progress(self , key) -> None:
        '''Print out status giving display conditions and looping conditions'''
        printers = [logger.info] if (config.verbosity > 2 or self.info.initial_models) else [logger.debug]
        sdout   = None
        if key == 'model_end':
            self.info.add_text('epoch' , 'Ep#{:3d}'.format(self.pipe.epoch_all))
            self.info.add_text('stat'  , 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
                self.pipe.score_list['train'][-1],self.pipe.score_list['valid'][-1],self.pipe.score_attempt_best))

            self.pipe.texts['stat']  = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
                self.pipe.score_list['train'][-1],self.pipe.score_list['valid'][-1],self.pipe.score_attempt_best)
            self.pipe.texts['time']  = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
                (self.info.toc('model'))/60 , (self.info.toc('model'))/(self.pipe.epoch_all+1))
            sdout = self.pipe.texts['model'] + '|' + self.pipe.texts['attempt'] + ' ' + \
                    self.pipe.texts['epoch'] + ' ' + self.pipe.texts['exit'] + '|' + \
                        self.pipe.texts['stat'] + '|' + self.pipe.texts['time']
            printers = [logger.warning]
        elif key == 'epoch_step':
            self.pipe.texts['trainer'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, max{: .4f}, best{: .4f}, lr{:.1e}'.format(
                self.pipe.loss_list['train'][-1] , self.pipe.score_list['train'][-1] , self.pipe.score_list['valid'][-1] , 
                self.pipe.score_attempt_best , self.pipe.score_attempt_best , self.pipe.lr_list[-1])
            if self.pipe.epoch_i % [10,5,5,3,3,1][min(config.verbosity // 2 , 5)] == 0:
                sdout = ' '.join([self.pipe.texts['attempt'],'Ep#{:3d}'.format(self.pipe.epoch_i),':', self.pipe.texts['trainer']])
        elif key == 'reset_learn_rate':
            sdout = 'Reset learn rate and scheduler at the end of epoch {} , effective at epoch {}'.format(
                self.pipe.epoch_i , self.pipe.epoch_i+1 , ', and will speedup2x' * config.train_param['trainer']['learn_rate']['reset']['speedup2x'])
        elif key == 'new_attempt':
            sdout = ' '.join([self.pipe.texts['attempt'],'Epoch #{:3d}'.format(self.pipe.epoch_i),':',
                              self.pipe.texts['trainer'],', Next attempt goes!'])
        else:
            raise Exception(f'KeyError : {key}')
        
        if sdout is not None: [prt(sdout) for prt in printers]

    def print_test_table(self , rowname , values , digits = 2):
        fmt = 's' if isinstance(values[0] , str) else (f'd' if digits == 0 else f'.{digits}f')
        logger.info(('{: <11s}'+('{: >8'+fmt+'}')*len(values)).format(str(rowname) , *values))

    def check_train_end(self):
        '''Whether terminate condition meets'''
        term_dict = config.train_param['terminate']
        term_cond = {}
        exit_text = ''
        for key , arg in term_dict.items():
            if key == 'max_epoch':
                term_cond[key] = self.pipe.epoch_i >= min(arg , config.max_epoch) - 1
                if term_cond[key] and exit_text == '': exit_text = 'Max Epoch'
            elif key == 'early_stop':
                term_cond[key] = self.pipe.epoch_i - self.pipe.epoch_attempt_best >= arg
                if term_cond[key] and exit_text == '': exit_text = 'EarlyStop'
            elif key == 'tv_converge':
                term_cond[key] = (list_converge(self.pipe.loss_list['train']  , arg.get('min_epoch') , arg.get('eps')) and
                                  list_converge(self.pipe.score_list['valid'] , arg.get('min_epoch') , arg.get('eps')))
                if term_cond[key] and exit_text == '': exit_text = 'T & V Cvg'
            elif key == 'train_converge':
                term_cond[key] = list_converge(self.pipe.loss_list['train']  , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text == '': exit_text = 'Train Cvg'
            elif key == 'valid_converge':
                term_cond[key] = list_converge(self.pipe.score_list['valid'] , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text == '': exit_text = 'Valid Cvg'
            else:
                raise Exception(f'KeyError : {key}')

        return exit_text , term_cond
    
    def save_model(self , paths = None , disk_key = None , savable_net = None):
        if paths is None and disk_key is None: return NotImplemented # nothing to save

        with self.ptimer(f'save_model'):
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
                        if len(p_exists) == 0: raise Exception('empty swa input')
                        swa_net = TRAIN.SWAModel(config.model_module , self.param , self.device.device)
                        for p in p_exists: swa_net.update_sd(self.storage.load(p))
                        swa_net.update_bn(self.data.train_dataloader())
                        savable_net = swa_net.module 

                    self.storage.save_model_state(savable_net , self.path['target'][key] , to_disk = True) 
    
    def load_model(self , training : bool , model_type = default_model_type) -> nn.Module:
        with self.ptimer(f'load_model'):
            if training:           
                if self.path['candidate'].get('transfer'):
                    if not config.train_param['transfer']: raise Exception('get transfer')
                    model_path = self.path['candidate']['transfer']
                else:
                    model_path = -1
            else:
                model_path = self.path['target'][model_type]

            net = MODEL.new(config.model_module , self.param , self.storage.load(model_path , from_disk = True))
            net = self.device(net)
        return net

    @classmethod
    def fit(cls , process = -1 , resume = -1 , checkname = -1 , timer = False , parser_args = None):
        if parser_args is None: parser_args = config.parser_args({'process':process,'resume':resume,'checkname':checkname})

        config.reload(do_process=True,par_args=parser_args)
        config.set_config_environment()

        logger.warning('Model Specifics:')
        config.print_out()

        app = cls(mem_storage = config.mem_storage , timer = timer)
        app.main_process()

if __name__ == '__main__':
    ModelTrainer.fit(parser_args = config.parser_args())
 