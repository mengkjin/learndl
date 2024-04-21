import logging , os , time
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass , field
from typing import Any , Callable , ClassVar , Literal , Optional 

from ..config import TrainConfig
from ..metric import Metrics , AggMetrics

from ...func import list_converge
from ...environ import DIR

@dataclass
class Pipeline:
    config : TrainConfig
    logger : logging.Logger
    nanloss     : bool = False
    loop_status : Literal['epoch' , 'attempt' , 'model'] = 'epoch'
    times       : dict[str,float] = field(default_factory=dict)

    test_score  : Any = None

    result_path : ClassVar[str] = f'{DIR.result}/model_results.yaml'

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
        self.tic('init')
        self.nanloss_life = self.config.train_param['trainer']['nanloss']['retry']
        self.metric_batchs = AggMetrics()
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}

    @property
    def initial_models(self): return self.model_stage < self.config.model_num

    def get(self , key , default = None): return self.__dict__.get(key , default)

    def set_stage(self , stage : Literal['data' , 'fit' , 'test']):
        self.stage = stage
        self.epoch_stage = 0
        self.model_stage = 0

    def set_dataset(self , dataset : Literal['train' , 'valid' , 'test']):
        self.dataset = dataset

    def new_loop(self):
        if self.loop_status == 'epoch':
            self.new_epoch()
        elif self.loop_status == 'attempt':
            self.new_attempt()
        elif self.loop_status == 'model':
            pass
        else:
            raise KeyError(self.loop_status)

    def new_epoch(self):
        self.epoch += 1
        self.epoch_model += 1
        self.epoch_stage += 1

        self.texts['epoch'] = 'Ep#{:3d}'.format(self.epoch)
        self.texts['epoch_model'] = 'Ep#{:3d}'.format(self.epoch_model)

    def new_attempt(self):
        '''Reset variables at attempt start'''
        self.nanloss = False

        self.epoch = -1
        self.epoch_attempt_best = -1
        self.score_attempt_best = -10000.
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}
        self.lr_list    = []

        self.attempt += 1
        self.texts['attempt'] = f'FirstBite' if self.attempt == 0 else f'Retrain#{self.attempt}'

        self.new_epoch()

    def new_model(self , model_num , model_date):
        '''Reset variables of model start'''
        self.model_num , self.model_date = model_num , model_date
        self.model_stage += 1

        self.attempt = -1
        self.epoch_model = -1

        self.texts = {k : '' for k in ['model','attempt','epoch','epoch_model','exit','status','time']}
        self.texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.model_num , self.model_date)
        self.loop_status = 'attempt'

    def end_model(self):
        self.texts['status'] = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
            self.train_score , self.valid_score , self.score_attempt_best)
        self.texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.toc('model') / 60 , self.toc('model') / (self.epoch_model + 1))
        
    def check_nanloss(self , is_nanloss):
        '''Deal with nanloss, life -1 and change nanloss condition to True'''
        if self.stage == 'fit' and is_nanloss: 
            self.logger.error(self.texts['model'] + f' Attempt{self.attempt}, epoch{self.epoch} got nanloss!')
            if self.nanloss_life > 0:
                self.logger.error(f'Initialize a new model to retrain! Lives remaining {self.nanloss_life}')
                self.nanloss_life -= 1
                self.nanloss   = True
                self.attempt -= 1
                self.loop_status = 'attempt'
            else:
                raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')
            
    def assess_terminate(self):
        '''Whether terminate condition meets'''
        if self.valid_score > self.score_attempt_best: 
            self.epoch_attempt_best = self.epoch 
            self.score_attempt_best = self.valid_score

        for key , arg in self.terminate_cond.items():
            if cond := self.is_terminate(key , arg):
                self.texts['exit'] = cond
                break
        
        if self.texts['exit']:
            too_early = (self.epoch < self.min_epoch - 1 and self.attempt < self.max_attempt - 1)
            self.loop_status = 'attempt' if too_early else 'model'
        else:
            self.loop_status = 'epoch'
        self.texts['status'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, best{: .4f}, lr{:.1e}'.format(
            self.train_loss , self.train_score , self.valid_score , self.score_attempt_best , self.lr_list[-1])
    
    def is_terminate(self , key , arg):
        if key =='max_epoch':
            return 'Max Epoch' * (self.epoch >= min(arg , self.config.max_epoch) - 1)
        elif key == 'early_stop':
            return 'EarlyStop' * (self.epoch - self.epoch_attempt_best >= arg)
        elif key == 'tv_converge':
            return 'T & V Cvg' * (self.metric_converge('train',arg) and self.metric_converge('valid',arg))
        elif key == 'train_converge':
            return 'Train Cvg' * self.metric_converge('train',arg)
        elif key == 'valid_converge':
            return 'Valid Cvg' * self.metric_converge('valid',arg)
        else:
            raise KeyError(key)
    
    def metric_converge(self , dataset , arg):
        if dataset == 'train':
            return list_converge(self.metric_epochs['train.loss'] , arg.get('min_epoch') , arg.get('eps'))
        else:
            return list_converge(self.metric_epochs['valid.score'], arg.get('min_epoch') , arg.get('eps'))

    def new_metric(self , model_type='best'):
        self.model_type = model_type
        self.metric_batchs.new(self.dataset , self.model_num , self.model_date , self.epoch , model_type)
    
    def record_metric(self , metrics : Metrics):
        self.metric_batchs.record(metrics)

    def collect_metric(self):
        self.check_nanloss(self.metric_batchs.nanloss)
        self.metric_batchs.collect()
        self.metric_epochs[f'{self.dataset}.loss'].append(self.metric_batchs.loss) 
        self.metric_epochs[f'{self.dataset}.score'].append(self.metric_batchs.score)

    def collect_lr(self , last_lr): 
        if self.dataset == 'train': self.lr_list.append(last_lr)

    def tic(self , key : str): self.times[key] = time.time()

    def tic_str(self , key : str):
        self.tic(key)
        return 'Start Process [{}] at {:s}!'.format(key.capitalize() , time.ctime(self.times[key]))

    def toc(self , key : str): 
        return time.time() - self.times[key]
    
    def toc_str(self , key : str): 
        toc = self.toc(key)
        if self.model_stage * self.epoch_stage:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.capitalize() , toc / 3600 , toc / 60 / self.model_stage , toc / self.epoch_stage)
        else:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.capitalize() , toc)
        return self.texts[f'{key}']

    def dump_info(self):
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : time.ctime(self.times['init']) ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : self.config.model_data_type ,
            '4_label' : ','.join(self.config.labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_fit'   : self.texts.get('fit'),
            '7_test'  : self.texts.get('test'),
            '8_result': self.test_score,
        }
        DIR.dump_yaml(result , self.result_path)

    @property
    def losses(self): return self.metric_batchs.losses
    @property
    def scores(self): return self.metric_batchs.scores
    @property
    def aggloss(self): return self.metric_batchs.loss
    @property
    def aggscore(self): return self.metric_batchs.score
    @property
    def max_epoch(self): return self.config.max_epoch
    @property
    def min_epoch(self): return self.config.train_param['trainer']['retrain']['min_epoch']
    @property
    def terminate_cond(self): return self.config.train_param['terminate']
    @property
    def max_attempt(self): return self.config.train_param['trainer']['retrain']['attempts']
    @property
    def train_score(self): return self.metric_epochs['train.score'][-1]
    @property
    def train_loss(self):  return self.metric_epochs['train.loss'][-1]
    @property
    def valid_score(self): return self.metric_epochs['valid.score'][-1]
    @property
    def valid_loss(self):  return self.metric_epochs['valid.loss'][-1]
    @property
    def epoch_print(self):
        return self.epoch % [10,5,5,3,3,1][min(self.config.verbosity // 2 , 5)] == 0
    
@dataclass
class TestResult:
    config : TrainConfig
    logger : logging.Logger

    def __post_init__(self):
        self.n_nums      = len(self.config.model_num_list)
        self.n_types     = len(self.config.output_types)
        self.model_nums  = np.repeat(self.config.model_num_list, self.n_types)
        self.model_types = np.tile(self.config.output_types, self.n_nums)
        self.ncols = self.n_nums * self.n_types

        self.score_by_date  = np.empty((0,self.ncols))
        self.score_by_model = np.empty((0,self.ncols))
        
        self.print_table_row('Models' , self.model_nums , 0)
        self.print_table_row('Output' , self.model_types)

    def new_model(self , model_date , model_num , test_dates):
        self.model_date , self.model_num , self.test_dates = model_date , model_num , test_dates
        if model_num == 0:
            score_date  = np.zeros((len(test_dates) , self.ncols))
            score_model = np.zeros((1 , self.ncols))
            self.score_by_date  = np.concatenate([self.score_by_date  , score_date])
            self.score_by_model = np.concatenate([self.score_by_model , score_model])

    def record_metric(self , model_type , scores):
        col = self.model_num * self.n_types + self.config.output_types.index(model_type)
        self.score_by_date[-len(self.test_dates):, col] = np.nan_to_num(scores[-len(self.test_dates):])

    def end_model(self):
        if self.model_num == self.config.model_num_list[-1]:
            self.score_by_model[-1,:] = np.nanmean(self.score_by_date[-len(self.test_dates):,],axis = 0)
            self.print_table_row(self.model_date , self.score_by_model[-1,:] , 4)

    def write_result(self , test_dates):
        # date ic writed down
        for model_num in self.config.model_num_list:
            date_str = np.array(list(map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , test_dates)))
            df = pd.DataFrame({'dates' : test_dates} , index = date_str)

            for i , model_type in enumerate(self.config.output_types):
                df[f'score.{model_type}'] = self.score_by_date[:,model_num * self.n_types  + i]
                df[f'cum_score.{model_type}'] = np.nancumsum(self.score_by_date[:,model_num * self.n_types  + i])
            df.to_csv(self.config.model_param[model_num]['path'] + f'/{self.config.model_name}_score_by_date_{model_num}.csv')

    def end(self , model_dates):
        # model ic presentation
        add_row_key   = ['AllTimeAvg' , 'AllTimeSum' , 'Std'      , 'TValue'   , 'AnnIR']
        score_mean   = np.nanmean(self.score_by_date , axis = 0)
        score_sum    = np.nansum(self.score_by_date  , axis = 0) 
        score_std    = np.nanstd(self.score_by_date  , axis = 0)
        score_tvalue = score_mean / score_std * (len(self.score_by_date)**0.5) # 10 days return predicted
        score_annir  = score_mean / score_std * ((240 / 10)**0.5) # 10 days return predicted
        add_row_value = (score_mean , score_sum , score_std , score_tvalue , score_annir)
        df = pd.DataFrame(np.concatenate([self.score_by_model , np.stack(add_row_value)]) , 
                          index   = [str(d) for d in model_dates] + add_row_key , 
                          columns = [f'{mn}.{mt}' for mn,mt in zip(self.model_nums , self.model_types)])
        df.to_csv(f'{self.config.model_base_path}/{self.config.model_name}_score_by_model.csv')
        for i , digits in enumerate([4,2,4,2,4]):
            self.print_table_row(add_row_key[i] , add_row_value[i] , digits)

    def print_table_row(self , rowname , values , digits = 2):
        fmt = 's' if isinstance(values[0] , str) else (f'd' if digits == 0 else f'.{digits}f')
        self.logger.info(('{: <11s}'+('{: >8'+fmt+'}')*len(values)).format(str(rowname) , *values))


