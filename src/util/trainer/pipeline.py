import gc , logging , os , time
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from typing import Any , Callable , ClassVar , Literal , Optional

from ..config import TrainConfig
from ..metric import Metrics , AggMetrics
from ...func import list_converge
from ...environ import DIR

@dataclass
class Pipeline:
    '''pipeline of a training / testing process'''
    config      : TrainConfig
    logger      : logging.Logger
    nanloss     : bool = False
    times       : dict[str,float] = field(default_factory=dict)
    texts       : dict[str,str]   = field(default_factory=dict)
    
    result_path : ClassVar[str] = f'{DIR.result}/model_results.yaml'

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
        self.tic('init')
        self._epoch_stage : int = 0
        self._model_stage : int = 0
        self._loop_status : Literal['epoch' , 'attempt' , 'model'] = 'epoch'
        self._nanloss_life = self.config.train_param['trainer']['nanloss']['retry']
        self.metric_batchs = AggMetrics()
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}

    @property
    def initial_models(self): return self._model_stage < self.config.model_num

    def get(self , key , default = None): return self.__dict__.get(key , default)

    def start_process(self):
        self.logger.warning('Model Specifics:')
        self.config.print_out()

    def end_process(self):
        result = {
            '0_model' : f'{self.config.model_name}(x{len(self.config.model_num_list)})',
            '1_start' : time.ctime(self.times['init']) ,
            '2_basic' : 'short' if self.config.short_test else 'full' , 
            '3_datas' : self.config.model_data_type ,
            '4_label' : ','.join(self.config.labels),
            '5_dates' : '-'.join([str(self.config.beg_date),str(self.config.end_date)]),
            '6_fit'   : self.texts.get('fit'),
            '7_test'  : self.texts.get('test'),
            '8_result': self.test_record.test_scores,
        }
        DIR.dump_yaml(result , self.result_path)

    def start_stage(self , stage):
        self.stage = stage
        self.logger.critical(self.tic_str(self.stage))
        if self.stage == 'test':
            self.logger.warning('Each Model Date Testing Mean Score({}):'.format(self.config.train_param['criterion']['score']))
            self.test_record = self.TestRecord(self.config.model_num_list , self.config.model_types , self.logger.info)
            self.test_record.print_colnames()
        else:
            self.test_record = self.TestRecord()

    def end_stage(self):
        if self.stage == 'test':
            for model_num in self.config.model_num_list:
                path = self.config.model_param[model_num]['path'] + f'/{self.config.model_name}_score_by_date_{model_num}.csv'
                self.test_record.score_table(model_num , path)
            self.test_record.summarize().to_csv(f'{self.config.model_base_path}/{self.config.model_name}_score_by_model.csv')
            self.test_record.print_summary()
        self.logger.critical(self.toc_str(self.stage , avg=self.stage == 'fit'))

    def new_epoch(self):
        self.epoch += 1
        self.epoch_model += 1
        self._epoch_stage += 1
        self.texts['epoch'] = 'Ep#{:3d}'.format(self.epoch)
        self.texts['epoch_model'] = 'Ep#{:3d}'.format(self.epoch_model)

    def new_attempt(self):
        '''Reset variables at attempt start'''
        self.nanloss = False
        self.epoch = -1
        self.epoch_attempt_best = -1
        self.score_attempt_best = -10000.
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}
        self.lr_list = []
        self.attempt += 1
        self.texts['attempt'] = f'FirstBite' if self.attempt == 0 else f'Retrain#{self.attempt}'
        self.new_epoch()

    def new_fit_model(self , model_num , model_date):
        '''Reset variables of model start'''
        self.tic('model')
        self.model_num , self.model_date = model_num , model_date
        self._model_stage += 1
        self.attempt = -1
        self.epoch_model = -1
        self.texts = {k : '' for k in ['model','attempt','epoch','epoch_model','exit','status','time']}
        self.texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.model_num , self.model_date)
        self._loop_status = 'attempt'

    def end_fit_model(self):
        self.texts['status'] = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
            self.train_score , self.valid_score , self.score_attempt_best)
        self.texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.toc('model') / 60 , self.toc('model') / (self.epoch_model + 1))
        
    def loop_continue(self): 
        cond = True
        if self._loop_status == 'epoch':
            self.new_epoch()
        elif self._loop_status == 'attempt':
            self.new_attempt()
        elif self._loop_status == 'model':
            cond = False
        else:
            raise KeyError(self._loop_status)
        return cond
    
    def loop_new_attempt(self): return self._loop_status == 'attempt'
    def loop_terminate(self): return self._loop_status == 'model'
        
    def check_nanloss(self , is_nanloss):
        '''Deal with nanloss, life -1 and change nanloss condition to True'''
        if self.stage == 'fit' and is_nanloss: 
            self.logger.error(self.texts['model'] + f' Attempt{self.attempt}, epoch{self.epoch} got nanloss!')
            if self._nanloss_life > 0:
                self.logger.error(f'Initialize a new model to retrain! Lives remaining {self._nanloss_life}')
                self._nanloss_life -= 1
                self.nanloss   = True
                self.attempt -= 1
                self._loop_status = 'attempt'
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
            self._loop_status = 'attempt' if too_early else 'model'
        else:
            self._loop_status = 'epoch'
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

    def new_metric(self , dataset , model_type='best'):
        self.dataset : Literal['train' , 'valid' , 'test'] = dataset
        self.model_type = model_type
        self.metric_batchs.new(self.dataset , self.model_num , self.model_date , self.epoch , model_type)
    
    def record_metric(self , metrics : Metrics):
        self.metric_batchs.record(metrics)

    def collect_metric_and_lr(self , last_lr):
        self.check_nanloss(self.metric_batchs.nanloss)
        self.metric_batchs.collect()
        self.metric_epochs[f'{self.dataset}.loss'].append(self.metric_batchs.loss) 
        self.metric_epochs[f'{self.dataset}.score'].append(self.metric_batchs.score)
        if self.dataset == 'train': self.lr_list.append(last_lr)

    def new_test_model(self , model_num , model_date , test_dates):
        self.model_date , self.model_num , self.test_dates = model_date , model_num , test_dates
        if model_num == 0: self.test_record.add_rows(model_date , test_dates)

    def update_test_score(self , model_type):
        self.test_record.update_score(self.model_num , model_type , self.scores[-len(self.test_dates):])

    def end_test_model(self):
        if self.model_num == self.test_record.model_nums[-1]:
            self.test_record.print_score()
            gc.collect()

    def tic(self , key : str): self.times[key] = time.time()

    def tic_str(self , key : str):
        self.tic(key)
        return 'Start Process [{}] at {:s}!'.format(key.capitalize() , time.ctime(self.times[key]))

    def toc(self , key : str): 
        return time.time() - self.times[key]
    
    def toc_str(self , key : str , avg = False): 
        toc = self.toc(key)
        if avg and self._model_stage * self._epoch_stage:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
                key.capitalize() , toc / 3600 , toc / 60 / self._model_stage , toc / self._epoch_stage)
        else:
            self.texts[key] = 'Finish Process [{}], Cost {:.1f} Secs'.format(key.capitalize() , toc)
        return self.texts[f'{key}']

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
    class TestRecord:
        model_nums : list  = field(default_factory=list)
        model_types : list = field(default_factory=list)
        printer : Callable = print

        summary_rname : ClassVar[dict[str,str]] = {'Avg':'AllTimeAvg','Sum':'AllTimeSum','Std':'Std','T':'TValue','IR':'AnnIR'}
        summary_digit : ClassVar[dict[str,int]] = {'Avg':4,'Sum':2,'Std':4,'T':2,'IR':4}

        def __post_init__(self):
            self.n_num = len(self.model_nums)
            self.n_type = len(self.model_types)
            self.n_col = self.n_num * self.n_type
            self.col_num  = np.repeat(self.model_nums , self.n_type).astype(str)
            self.col_type = np.tile(self.model_types , self.n_num).astype(str)
            self.col_summary = [f'{mn}.{mt}' for mn,mt in zip(self.col_num , self.col_type)]

            self.n_date  = 0
            self.n_model = 0
            self.row_date  = np.array([]).astype(int)
            self.row_model = np.array([]).astype(int)
            self.score_by_date  = np.empty((self.n_date , self.n_col))
            self.score_by_model = np.empty((self.n_model, self.n_col))

        def add_rows(self , model , dates):
            if isinstance(model , int): model = [model]
            self.n_date += len(dates)
            self.n_model += 1
            self.row_date  = np.concatenate([self.row_date  , dates])
            self.row_model = np.concatenate([self.row_model  , [model]])
            self.score_by_date  = np.concatenate([self.score_by_date  , np.zeros((len(dates) , self.n_col))])
            self.score_by_model = np.concatenate([self.score_by_model , np.zeros((1 , self.n_col))])

        def update_score(self , model_num , model_type , scores):
            col = model_num * self.n_type + self.model_types.index(model_type)
            self.score_by_date[-len(scores):, col] = np.nan_to_num(scores)
            self.score_by_model[-1,col] = np.nanmean(scores)

        def summarize(self):
            self.summary : dict[str,np.ndarray] = {
                'Avg' : (score_mean := np.nanmean(self.score_by_date , axis = 0)) , 
                'Sum' : np.nansum(self.score_by_date  , axis = 0) , 
                'Std' : (score_std  := np.nanstd(self.score_by_date  , axis = 0)), 
                'T'   : score_mean / score_std * (len(self.score_by_date)**0.5), 
                'IR'  : score_mean / score_std * ((240 / 10)**0.5) ,
            }
            values = np.concatenate([self.score_by_model , np.stack(list(self.summary.values()))])
            index  = [str(d) for d in self.row_model] + [self.summary_rname[k] for k in self.summary.keys()]
            return pd.DataFrame(values , index = index , columns = self.col_summary)

        def print_colnames(self):
            self.print_row('Models' , self.col_num)
            self.print_row('Output' , self.col_type)

        def print_score(self):
            self.print_row(self.row_model[-1] , self.score_by_model[-1,:] , 4)

        def print_summary(self):
            [self.print_row(self.summary_rname[k],v,self.summary_digit[k]) for k,v in self.summary.items()]

        def print_row(self , row , values , digit = 2):
            fmt = 's' if isinstance(values[0] , str) else (f'd' if digit == 0 else f'.{digit}f')
            self.printer(('{: <11s}'+('{: >8'+fmt+'}')*len(values)).format(str(row) , *values))

        def score_table(self , model_num , save_to : Optional[str] = None):
            df = pd.DataFrame({'dates' : self.row_date} , 
                              index = list(map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , self.row_date.astype(str))))
            for i , model_type in enumerate(self.model_types):
                df[f'score.{model_type}'] = self.score_by_date[:,model_num * self.n_type  + i]
                df[f'cum_score.{model_type}'] = np.nancumsum(self.score_by_date[:,model_num * self.n_type  + i])

            if save_to: df.to_csv(save_to)
            return df
        
        @property
        def test_scores(self):
            if not hasattr(self , 'summary'): return
            return {col:'|'.join([f'{k}({round(v[i],self.summary_digit[k])})' for k,v in self.summary.items()]) 
                    for i , col in enumerate(self.col_summary)}
