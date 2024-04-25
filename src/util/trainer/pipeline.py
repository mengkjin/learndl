import gc , logging , os , time
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from typing import Any , Callable , ClassVar , Literal , Optional

from ..config import TrainConfig
from ..trainer.optim import Optimizer
from ..metric import Metrics , AggMetrics
from ...func import list_converge
from ...environ import DIR

class Pipeline:
    '''pipeline of a training / testing process'''    
    result_path = f'{DIR.result}/model_results.yaml'

    def __init__(self , Mmod : Any):
        self.Mmod = Mmod
        os.makedirs(os.path.dirname(self.result_path) , exist_ok=True)
        self.tic('init')
        self.nanloss      : bool = False
        self.times        : dict[str,float] = field(default_factory=dict)
        self.texts        : dict[str,str]   = field(default_factory=dict)
        self._epoch_stage : int = 0
        self._model_stage : int = 0
        self._loop_status : Literal['epoch' , 'attempt' , 'model'] = 'epoch'
        self._nanloss_life = self.config.train_param['trainer']['nanloss']['retry']
        self.metric_batchs = AggMetrics()
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}
        self.test_record = self.TestRecord()
    
    @property
    def config(self) -> TrainConfig:    return self.Mmod.config
    @property
    def logger(self) -> logging.Logger: return self.Mmod.logger
    @property
    def initial_models(self): return self._model_stage < self.config.model_num

    def get(self , key , default = None): return self.__dict__.get(key , default)

    def summerize_model(self):
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
        self._loop_status = 'epoch'

    def new_model(self , model_num , model_date):
        self.tic('model')
        self.model_num , self.model_date = model_num , model_date
        self._model_stage += 1
        self.attempt = -1
        self.epoch_model = -1
        self.texts = {k : '' for k in ['model','attempt','epoch','epoch_model','exit','status','time']}
        self.texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.model_num , self.model_date)
        self._loop_status = 'attempt'
    
    @property
    def loop_continue(self): 
        if self._loop_status in ['attempt' , 'epoch']:
            return True
        elif self._loop_status == 'model':
            return False
        else:
            raise KeyError(self._loop_status)
    @property
    def loop_new_attempt(self): return self._loop_status == 'attempt'
    @property
    def loop_terminate(self): return self._loop_status == 'model'
        
    def check_nanloss(self , is_nanloss):
        '''Deal with nanloss, life -1 and change nanloss condition to True'''
        if self.stage == 'fit' and is_nanloss: 
            self.logger.error(self.texts['model'] + f' Attempt{self.attempt}, epoch{self.epoch} got nanloss!')
            if self._nanloss_life > 0:
                self.logger.error(f'Initialize a new model to retrain! Lives remaining {self._nanloss_life}')
                self._nanloss_life -= 1
                self.nanloss  = True
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

    def collect_metric(self):
        self.check_nanloss(self.metric_batchs.nanloss)
        self.metric_batchs.collect()
        self.metric_epochs[f'{self.dataset}.loss'].append(self.metric_batchs.loss) 
        self.metric_epochs[f'{self.dataset}.score'].append(self.metric_batchs.score)

    def collect_lr(self , last_lr):
        assert self.dataset == 'train' , self.dataset
        self.lr_list.append(last_lr)

    def update_test_score(self , model_type):
        self.test_record.update_score(self.model_num , model_type , self.scores[-len(self.test_dates):])

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
    def progress_log(self):
        return self.logger.info if (self.config.verbosity > 2 or self.initial_models) else self.logger.debug
    
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
        
    # callbacks
    def configure_model(self , Mmod):
        self.config.set_config_environment()
        self.logger.warning('Model Specifics:')
        self.config.print_out()
    def on_data_start(self , Mmod): 
        self.stage = 'data'
        self.logger.critical(self.tic_str('data'))
    def on_data_end(self , Mmod): 
        self.logger.critical(self.toc_str('data'))
    def on_fit_start(self , Mmod): 
        self.stage = 'fit'
        self.logger.critical(self.tic_str('fit'))
    def on_fit_end(self , Mmod): 
        self.logger.critical(self.toc_str('fit' , avg=True))
    def on_fit_model_start(self , Mmod):
        self.new_model(Mmod.model_num , Mmod.model_date)
    def on_fit_model_end(self , Mmod):
        self.texts['status'] = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(
            self.train_score , self.valid_score , self.score_attempt_best)
        self.texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            self.toc('model') / 60 , self.toc('model') / (self.epoch_model + 1))
        self.logger.warning('{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.texts))
    def on_train_epoch_start(self , Mmod):
        self.new_epoch()
        self.new_metric(Mmod.dataset , 'best')
    def on_train_epoch_end(self , Mmod):
        opt : Optimizer = Mmod.optimizer
        self.collect_metric()
        self.collect_lr(opt.last_lr)
        if opt.scheduler_step(self.epoch) == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.epoch} , effective at epoch {self.epoch + 1}' + \
                ', and will speedup2x' * self.config.train_param['trainer']['learn_rate']['reset']['speedup2x']
            self.progress_log(sdout)
    def on_validation_epoch_start(self , Mmod):
        self.new_metric(Mmod.dataset , 'best')
    def on_validation_epoch_end(self , Mmod):
        self.collect_metric()
        self.assess_terminate()
        if self.epoch % [10,5,5,3,3,1][min(self.config.verbosity // 2 , 5)] == 0: 
            self.progress_log('{attempt} {epoch} : {status}'.format(**self.texts))
        if self.loop_new_attempt: self.progress_log('{attempt} {epoch} : {status}, Next attempt goes!'.format(**self.texts))
    def on_test_start(self , Mmod): 
        self.stage = 'test'
        self.logger.critical(self.tic_str('test'))
        self.logger.warning('Each Model Date Testing Mean Score({}):'.format(self.config.train_param['criterion']['score']))
        self.test_record = self.TestRecord(self.config.model_num_list , self.config.model_types , self.logger.info)
        self.test_record.print_colnames()
    def on_test_end(self , Mmod): 
        for model_num in self.config.model_num_list:
            path = self.config.model_param[model_num]['path'] + f'/{self.config.model_name}_score_by_date_{model_num}.csv'
            self.test_record.score_table(model_num , path)
        self.test_record.summarize().to_csv(f'{self.config.model_base_path}/{self.config.model_name}_score_by_model.csv')
        self.test_record.print_summary()
        self.logger.critical(self.toc_str('test'))
    def on_test_model_start(self , Mmod):
        self.model_date , self.model_num , self.test_dates = Mmod.model_num , Mmod.model_date , Mmod.data_mod.model_test_dates
        if self.model_num == 0: self.test_record.add_rows(self.model_date , self.test_dates)
    def on_test_model_end(self , Mmod):
        if self.model_num == self.test_record.model_nums[-1]:
            self.test_record.print_score()
            gc.collect()
    def on_test_model_type_start(self , Mmod):
        self.new_metric(Mmod.dataset , Mmod.model_type)
    def on_test_model_type_end(self , Mmod):
        self.collect_metric()
        self.update_test_score(Mmod.model_type)

