import itertools
import numpy as np
import pandas as pd
import torch

from abc import ABC , abstractmethod
from dataclasses import dataclass
from inspect import currentframe
from pathlib import Path
from typing import Any , final , Iterator , Literal , Callable

from src.proj import Proj , Logger , DB , PATH
from src.proj.util import FilteredIterable
from src.res.algo import AlgoModule
from src.data import ModuleData

from .batch import BatchData , BatchOutput
from .buffer import BaseBuffer
from .config import TrainConfig
from .model_path import ModelDict
from .storage import MemFileStorage

class ModelStreamLine(ABC):
    """Base class for all model stream lines , e.g. trainer, predictor, data module, etc."""
    def stage_data(self): ...
    def stage_fit(self):  ...
    def stage_test(self): ...
    def on_configure_model(self): ...
    def on_summarize_model(self): ...
    def on_data_end(self): ... 
    def on_data_start(self): ... 
    def on_after_backward(self): ... 
    def on_after_fit_epoch(self): ... 
    def on_before_backward(self): ... 
    def on_before_save_model(self): ... 
    def on_before_fit_epoch_end(self): ... 
    def on_fit_end(self): ... 
    def on_fit_epoch_end(self): ... 
    def on_fit_epoch_start(self): ... 
    def on_fit_model_end(self): ... 
    def on_fit_model_start(self): ... 
    def on_fit_model_date_end(self): ... 
    def on_fit_model_date_start(self): ... 
    def on_fit_start(self): ...
    def on_test_batch_end(self): ...
    def on_test_batch_start(self): ... 
    def on_test_end(self): ... 
    def on_test_model_end(self): ... 
    def on_test_model_start(self): ... 
    def on_test_model_date_end(self): ... 
    def on_test_model_date_start(self): ... 
    def on_test_submodel_end(self): ... 
    def on_test_submodel_start(self): ... 
    def on_test_start(self): ... 
    def on_train_batch_end(self): ... 
    def on_train_batch_start(self): ... 
    def on_train_epoch_end(self): ... 
    def on_train_epoch_start(self): ... 
    def on_validation_batch_end(self): ... 
    def on_validation_batch_start(self): ...
    def on_validation_epoch_end(self): ...
    def on_validation_epoch_start(self): ...
    def on_before_data_start(self): ...
    def on_after_data_end(self): ...
    def on_before_fit_start(self): ...
    def on_after_fit_end(self): ...
    def on_before_test_start(self): ...
    def on_before_test_end(self): ...
    def on_after_test_end(self): ...
    def execute_hook(self , hook : str):
        getattr(self , hook)()

    possible_hooks : list[str] = [
        'stage_data' ,
        'stage_fit' ,
        'stage_test' ,
        'on_configure_model' ,
        'on_summarize_model' ,
        'on_data_end' , 
        'on_data_start' , 
        'on_after_backward' , 
        'on_after_fit_epoch' , 
        'on_before_backward' , 
        'on_before_save_model' , 
        'on_before_fit_epoch_end' , 
        'on_fit_end' , 
        'on_fit_epoch_end' , 
        'on_fit_epoch_start' , 
        'on_fit_model_end' , 
        'on_fit_model_start' , 
        'on_fit_model_date_end' , 
        'on_fit_model_date_start' , 
        'on_fit_start' ,
        'on_test_batch_end' ,
        'on_test_batch_start' , 
        'on_test_end' , 
        'on_test_model_end' , 
        'on_test_model_start' , 
        'on_test_model_date_end' , 
        'on_test_model_date_start' , 
        'on_test_submodel_end' , 
        'on_test_submodel_start' , 
        'on_test_start' , 
        'on_train_batch_end' , 
        'on_train_batch_start' , 
        'on_train_epoch_end' , 
        'on_train_epoch_start' , 
        'on_validation_batch_end' , 
        'on_validation_batch_start' ,
        'on_validation_epoch_end' ,
        'on_validation_epoch_start' ,
        'on_before_data_start' ,
        'on_after_data_end' ,
        'on_before_fit_start' ,
        'on_after_fit_end' ,
        'on_before_test_start' ,
        'on_before_test_end' ,
        'on_after_test_end' ,
    ]

class HookWrapper:
    wrap_count : int = 0
    max_wrap_count = 1
    raw_hooks : dict[str , Callable] = {}

    @classmethod
    def wrap(cls , trainer : 'BaseTrainer'):
        assert cls.wrap_count < cls.max_wrap_count , f'Callbacks are already wrapped {cls.wrap_count} times'
        for hook in trainer.possible_hooks:
            cls.raw_hooks[hook] = getattr(trainer , hook)
            setattr(trainer , hook , cls.wrap_single_hook(trainer , hook))
        cls.wrap_count += 1

    @classmethod
    def wrap_single_hook(cls , trainer : 'BaseTrainer' , hook : str):
        def wrapper(*args , **kwargs) -> None:
            Logger.stdout(f'{hook} of stage {trainer.status.stage} start' , vb_level = Proj.vb.level_callback)
            trainer.callback.at_enter(hook , Proj.vb.level_callback)
            trainer.status.execute_hook(hook)
            cls.raw_hooks[hook](*args , **kwargs)
            trainer.model.execute_hook(hook)
            trainer.record.execute_hook(hook)
            trainer.callback.at_exit(hook , Proj.vb.level_callback)
            Logger.stdout(f'{hook} of stage {trainer.status.stage} end' , vb_level = Proj.vb.level_callback)
        return wrapper

@dataclass
class _EndEpochStamp:
    """End epoch stamp class, used to store the end epoch of the model stream line"""
    name  : str
    epoch : int # epoch of trigger

class _FitLoopBreaker:
    """Fit loop breaker class, used to break the fit loop when the epoch is too large or meet some conditions"""
    def __init__(self , max_epoch : int = 200):
        self.max_epoch = max_epoch
        self.status : list[_EndEpochStamp] = []
    def __bool__(self): 
        return len(self.status) > 0
    def __repr__(self): 
        return f'{self.__class__.__name__}(max_epoch={self.max_epoch},status={self.status})'  
    def new_loop(self): 
        self.status = []
    def loop_end(self , epoch):
        if epoch >= self.max_epoch - 1: 
            self.add_status('Max Epoch' , epoch)
    def add_status(self , status : str , epoch : int): 
        self.status.append(_EndEpochStamp(status , epoch))
    @property
    def end_epochs(self) -> list[int]:
        return [sta.epoch for sta in self.status]
    @property
    def trigger_i(self) -> int:
        return np.argmin(self.end_epochs).item()
    @property
    def trigger_ep(self) -> int:
        return self.status[self.trigger_i].epoch
    @property
    def trigger_reason(self):
        return self.status[self.trigger_i].name

class TrainerStatus(ModelStreamLine):
    """Trainer status class, used to store the status of the trainer"""
    def __init__(self , max_epoch : int = 200):
        self.max_epoch : int = max_epoch
        self.stage   : Literal['data' , 'fit' , 'test'] = 'data'
        self.dataset : Literal['train' , 'valid' , 'test' , 'predict'] = 'train'
        self.epoch   : int = -1
        self.attempt : int = 0
        self.round   : int = 0
        
        self.model_num  : int = -1
        self.model_date : int = -1
        self.model_submodel : str = 'best'
        self.epoch_event : list[str] = []
        self.best_attempt_metric : Any = None

        self.fitted_model_num : int = 0

        self.fit_loop_breaker = _FitLoopBreaker(self.max_epoch)
        self.fit_iter_num : int = 0

        self.test_summary : pd.DataFrame = pd.DataFrame()

    def as_dict(self):
        d = {k:getattr(self,k) for k in 
             ['max_epoch' , 'stage' , 'dataset' , 'epoch' , 'attempt' , 
              'round' , 'model_num' , 'model_date' , 'model_submodel' , 
              'epoch_event' , 'best_attempt_metric' , 'fitted_model_num']}
        return d

    def __repr__(self):
        return f'TrainerStatus({", ".join([f"{k}={v}" for k,v in self.status.items()])})'

    @property
    def status(self):
        return {
            'stage' : self.stage ,
            'dataset' : self.dataset ,
            'model_num' : self.model_num ,
            'model_date' : self.model_date ,
            'model_submodel' : self.model_submodel ,
            'epoch' : self.epoch ,
            'attempt' : self.attempt ,
            'round' : self.round
        }

    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def on_train_epoch_start(self): self.dataset = 'train'
    def on_validation_epoch_start(self): self.dataset = 'valid'
    def on_test_model_start(self): self.dataset = 'test'
    def on_fit_model_start(self):
        if self.fit_iter_num == 0:
            Logger.note(f'First Iterance: ({self.model_date} , {self.model_num})')
        self.fit_iter_num += 1
        self.attempt = -1
        self.best_attempt_metric = None
        self.new_attempt()
    def on_fit_model_end(self):
        self.fitted_model_num += 1
    def on_fit_epoch_start(self):
        self.epoch   += 1
        self.epoch_event = []
    def on_fit_epoch_end(self):
        self.fit_loop_breaker.loop_end(self.epoch)
    def new_attempt(self , event : Literal['new_attempt' , 'nanloss'] = 'new_attempt'):
        self.epoch   = -1
        self.round   = 0
        self.epoch_event = []
        self.fit_loop_breaker.new_loop()
        self.add_event(event)
        if event == 'new_attempt': 
            self.attempt += 1

    def add_event(self , event : str | None):
        if event: 
            self.epoch_event.append(event)

class TrainerPredRecorder(ModelStreamLine):
    """Trainer predictor recorder class, used to record the predictor results"""
    PRED_KEYS = ['model_num' , 'model_date' , 'submodel' , 'batch_idx']
    PRED_IDXS = ['secid' , 'date']
    PRED_COLS = ['pred' , 'label']
    def __init__(self , trainer : 'BaseTrainer') -> None:
        self.trainer = trainer
        self.folder_preds.mkdir(exist_ok=True , parents=True)
        self.folder_avg_preds.mkdir(exist_ok=True , parents=True)

    def __repr__(self):
        return f'{self.__class__.__name__}(trainer={self.trainer})'

    @property
    def pred_idx(self):
        """unique index for each prediction batch"""
        return f'{self.trainer.model_num}.{self.trainer.model_date}.{self.trainer.model_submodel}.{self.trainer.batch_idx}'

    @property
    def pred_dict(self) -> dict[str,pd.DataFrame]:
        """stored predictions for each batch"""
        if not hasattr(self , '_preds_dict'):
            self._preds_dict = {}
        return self._preds_dict

    @pred_dict.setter
    def pred_dict(self , value : dict[str,pd.DataFrame]):
        self._preds_dict = value

    @property
    def resumed_preds_models(self) -> pd.DataFrame:
        """models dataframe for resumed testing"""
        if not hasattr(self , '_resumed_preds_models'):
            self._resumed_preds_models = pd.DataFrame(columns = ['model_num' , 'model_date'])
        return self._resumed_preds_models

    @resumed_preds_models.setter
    def resumed_preds_models(self , value : pd.DataFrame):
        self._resumed_preds_models = value

    @property
    def resumed_last_pred_date(self) -> int:
        """last predicted date for resumed testing"""
        if not hasattr(self , '_resumed_last_pred_date'):
            self._resumed_last_pred_date = 19000101
        return self._resumed_last_pred_date

    @resumed_last_pred_date.setter
    def resumed_last_pred_date(self , value : int):
        self._resumed_last_pred_date = value

    @property
    def snap_folder(self) -> Path: 
        """folder to save model predictions"""
        return self.trainer.config.model_base_path.snapshot('pred_recorder')
    @property
    def folder_preds(self) -> Path:
        """folder to save model predictions"""
        return self.snap_folder.joinpath('preds')
    @property
    def folder_avg_preds(self) -> Path:
        """folder to save averaged model predictions"""
        return self.snap_folder.joinpath('avg_preds')
    @property
    def min_test_date(self) -> int:
        """minimum test date"""
        return self.trainer.data.test_full_dates.min() if len(self.trainer.data.test_full_dates) > 0 else 99991231

    @property
    def max_test_date(self) -> int:
        """maximum test date"""
        return self.trainer.data.test_full_dates.max() if len(self.trainer.data.test_full_dates) > 0 else 19000101

    def save_preds(self , df : pd.DataFrame , model_date : int , model_num : int , append = False):
        if df.empty:
            return
        
        old_path = [path for path in self.folder_preds.glob('*.feather') if path.name.startswith(f'{model_num}.{model_date}.')]
        assert len(old_path) <= 1 , f'Multiple old paths found for model {model_num} at date {model_date}: {old_path}'
        if old_path and append:
            old_df = DB.load_df(old_path[0])
            df = pd.concat([old_df , df]).drop_duplicates(subset=self.PRED_KEYS + self.PRED_IDXS , keep='last').sort_values(by=self.PRED_KEYS + self.PRED_IDXS)
            
        min_pred_date , max_pred_date = df['date'].min() , df['date'].max()
        path = self.folder_preds.joinpath(f'{model_num}.{model_date}.{min_pred_date}.{max_pred_date}.feather')
        DB.save_df(df , path , overwrite = True , vb_level = Proj.vb.max)

        if old_path and path != old_path[0]:
            Path(old_path[0]).unlink()

    def save_avg_preds(self , model_date : int):
        pred_paths = [path for path in self.folder_preds.glob('*.feather') if path.name.split('.')[1] == str(model_date)]
        df = DB.load_dfs(pred_paths)
        if df.empty:
            return
        df = df.groupby(['model_date' , 'submodel' , 'secid' , 'date'])[['pred' , 'label']].mean().reset_index()
        min_pred_date , max_pred_date = df['date'].min() , df['date'].max()
        path = self.folder_avg_preds.joinpath(f'{model_date}.{min_pred_date}.{max_pred_date}.feather')
        DB.save_df(df , path , overwrite = True , vb_level = Proj.vb.max)

    def archive_model_records(self):
        records : list[tuple[int,int]] = []
        for model_num , model_date , _ , _ in self.trainer.config.model_base_path.iter_model_archives():
            records.append((model_num , model_date))
        df = pd.DataFrame(records , columns = ['model_num' , 'model_date']) if records else pd.DataFrame(columns = ['model_num' , 'model_date']).astype(int)
        df = df.drop_duplicates().sort_values(by=['model_num' , 'model_date'])
        df['next_model_date'] = df.groupby('model_num')['model_date'].shift(-1)
        return df

    def pred_records(self): 
        """model_date/model_num of saved predictions"""
        return pd.DataFrame([[path , *path.name.split('.')[:4]] for path in self.folder_preds.glob('*.feather')], columns = ['path' , 'model_num' , 'model_date' , 'min_pred_date' , 'max_pred_date']).\
            astype({'model_num' : int , 'model_date' : int , 'min_pred_date' : int , 'max_pred_date' : int})

    def avg_pred_records(self):
        return pd.DataFrame([[path , *path.name.split('.')[:3]] for path in self.folder_avg_preds.glob('*.feather')], columns = ['path' , 'model_date' , 'min_pred_date' , 'max_pred_date']).\
            astype({'model_date' : int , 'min_pred_date' : int , 'max_pred_date' : int})
        
    @property
    def retrained_models(self) -> list[tuple[int,int]]:
        """retrained models for resumed testing , must be tested"""
        if not hasattr(self , '_retrained_models'):
            self._retrained_models = []
        return self._retrained_models

    @classmethod
    def empty_preds(cls) -> pd.DataFrame:
        """empty predictions dataframe"""
        return pd.DataFrame(columns = cls.PRED_KEYS + cls.PRED_IDXS + cls.PRED_COLS)

    def append_retrained_model(self):
        """
        append retrained model to retrained_models list
        """
        self.retrained_models.append((self.trainer.model_date , self.trainer.model_num))

    def purge_retrained_model_preds(self , vb_level : int = 2):
        """purge past predictions when trained new models"""
        if not self.retrained_models:
            return
        min_retrained_model_date = min([model_date for model_date , _ in self.retrained_models])
        pred_records = self.pred_records()
        purge_models = pred_records.query('model_date >= @min_retrained_model_date')
        trim_models = pred_records.query('model_date < @min_retrained_model_date and max_pred_date > @min_retrained_model_date')
        if not purge_models.empty or not trim_models.empty:
            purge_info = f'Purged saved predictions after retrained model date {min_retrained_model_date}'
            if not purge_models.empty:
                purge_info += f', {len(purge_models)} models(date/num) purged'
            if not trim_models.empty:
                purge_info += f', {len(trim_models)} models(date/num) trimed'
            
            for _ , (model_date , model_num , path) in trim_models.loc[:,['model_date' , 'model_num' , 'path']].iterrows():
                df = DB.load_df(path).query('date <= @min_retrained_model_date')
                Path(path).unlink()
                self.save_preds(df , model_date , model_num)
                
            Logger.note(purge_info , vb_level = vb_level)
        else:
            Logger.note(f'No retrained models found, no purge needed' , vb_level = vb_level)

    def purge_outdated_model_preds(self , vb_level : int = 2):
        archive_records = self.archive_model_records()
        pred_records = self.pred_records()
        new_pred_records = archive_records.merge(pred_records , on=['model_num' , 'model_date'] , how='outer')
        new_pred_records['next_model_date'] = new_pred_records['next_model_date'].fillna(99991231)
        df = new_pred_records.query('min_pred_date <= model_date or max_pred_date > next_model_date')
        if df.empty:
            Logger.note(f'No outdated predictions found, no purge needed' , vb_level = vb_level)
            return

        purge_info = f'Purged outdated predictions, {len(df)} models(date/num) partially purged :'
        Logger.note(purge_info , vb_level = vb_level)
        Logger.Display(df , vb_level = vb_level)
        for _ , (model_date , model_num , path , next_model_date) in df.loc[:,['model_date' , 'model_num' , 'path' , 'next_model_date']].iterrows():
            df = DB.load_df(path).query('date <= @next_model_date and date >= @model_date')
            Path(path).unlink()
            self.save_preds(df , model_date , model_num)

    def setup_resuming_status(self , vb_level : int = 2):
        """
        setup resuming status for previous saved predictions
        notes:
        - only resume predictions before the last model date if resume option is 'last_model_date'
        - only resume predictions with all submodels
        """ 
        if not self.trainer.config.is_resuming or not Proj.Conf.Model.TRAIN.resume_test:
            return
        
        resume_info = f'Resume testing'
        pred_records = self.pred_records()

        latest_model_date = pred_records.groupby('model_num')['model_date'].max().min()
        pred_records = pred_records.query('max_pred_date >= @self.min_test_date & min_pred_date <= @self.max_test_date')
        min_pred_date = pred_records.groupby('model_num')['min_pred_date'].min().max()

        if not pred_records.empty:
            if self.min_test_date < min_pred_date:
                resume_info += f', but new test start {self.min_test_date} is earlier than saved preds {min_pred_date}, forfeiting resume preds'
            elif Proj.Conf.Model.TRAIN.resume_test == 'last_model_date':
                pred_records = pred_records.query('model_date < @latest_model_date')
                if not pred_records.empty:
                    self.resumed_preds_models = pred_records.loc[:,['model_date' , 'model_num']].reset_index(drop=True)
                    self.resumed_last_pred_date = min(pred_records['max_pred_date'].max() , self.max_test_date)
                    resume_info += f', recognize past saved preds before model date {latest_model_date}'
            elif Proj.Conf.Model.TRAIN.resume_test == 'last_pred_date':
                self.resumed_preds_models = pred_records.loc[:,['model_date' , 'model_num']].reset_index(drop=True)
                self.resumed_last_pred_date = min(pred_records['max_pred_date'].max() , self.max_test_date)
                resume_info += f', recognize past saved preds before prediction date {self.resumed_last_pred_date}'
            else:
                raise ValueError(f'Invalid resuming testing option: {Proj.Conf.Model.TRAIN.resume_test}')
            
        if pred_records.empty:
            resume_info += f', no saved preds found'

        Logger.note(resume_info , vb_level = vb_level)
    
    def append_batch_preds(self):
        if self.pred_idx in self.pred_dict.keys(): 
            return
        if self.trainer.batch_output.is_empty: 
            return
        
        which_output = self.trainer.model_param.get('which_output' , 0)
        
        secid = self.trainer.data.batch_secid(self.trainer.batch_data)
        date  = self.trainer.data.batch_date(self.trainer.batch_data)
        pred = self.trainer.batch_output.pred_df(secid , date).dropna()
        pred = pred.query('date in @self.trainer.data.test_full_dates')
        if len(pred) == 0: 
            return

        label = pd.DataFrame({'secid' : secid , 'date' : date , 'label' : self.trainer.data.batch_label(self.trainer.batch_data)})
        
        df = pred.merge(label , on=self.PRED_IDXS)
        if which_output is None:
            df['pred'] = df.loc[:,[col for col in df.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            df['pred'] = df[f'pred.{which_output}']
        df = df.assign(model_num = self.trainer.model_num , submodel = self.trainer.model_submodel , model_date = self.trainer.model_date , batch_idx = self.trainer.batch_idx)
        df = df.loc[:,self.PRED_KEYS + self.PRED_IDXS + self.PRED_COLS]

        self.pred_dict[self.pred_idx] = df

    def collect_model_preds(self):
        if not self.pred_dict:
            return self.empty_preds()
        self.save_preds(pd.concat(self.pred_dict.values()) , self.trainer.model_date , self.trainer.model_num , append = True)
        self.pred_dict.clear()

    def collect_avg_preds(self):
        self.save_avg_preds(self.trainer.model_date)

    def get_preds(self , pred_dates : np.ndarray , model_num : int | None = None) -> pd.DataFrame:
        # maybe give start and end dates to the function? so that analysis can start from last analysis date, instead of last pred date
        if len(pred_dates) == 0:
            return self.empty_preds()
        pred_records = self.pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        if model_num is not None:
            pred_records = pred_records.query('model_num == @model_num')
        assert not pred_records.empty , f'No pred records found for test dates {pred_dates}'
        df = DB.load_dfs(pred_records['path'].tolist()).query('date in @pred_dates')
        return df

    def get_avg_preds(self , pred_dates : np.ndarray) -> pd.DataFrame:
        # maybe give start and end dates to the function? so that analysis can start from last analysis date, instead of last pred date
        if len(pred_dates) == 0:
            return self.empty_preds()
        avg_pred_records = self.avg_pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        pred_records = self.pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        [self.save_avg_preds(model_date) for model_date in np.setdiff1d(pred_records['model_date'] , avg_pred_records['model_date'])]
                
        avg_pred_records = self.avg_pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        assert not avg_pred_records.empty , f'No avg pred records found for test dates {pred_dates}'
        df = DB.load_dfs(avg_pred_records['path'].tolist()).query('date in @pred_dates')
        return df

    def on_fit_model_end(self):
        self.append_retrained_model()

    def on_fit_end(self):
        self.purge_retrained_model_preds()

    def on_test_start(self):
        self.purge_outdated_model_preds()
        self.setup_resuming_status()
                
    def on_test_batch_end(self): 
        self.append_batch_preds()

    def on_test_model_end(self):
        self.collect_model_preds()

    def on_test_model_date_end(self):
        self.collect_avg_preds()
        
class BaseDataModule(ABC):
    '''A class to store relavant training data'''
    @abstractmethod
    def __init__(self , config : TrainConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit'):
        self.config   : TrainConfig
        self.use_data : Literal['fit','predict','both'] 
        self.storage  : MemFileStorage
        self.buffer   : BaseBuffer
    @abstractmethod
    def prepare_data() -> None: '''prepare all data in advance of training'''
    @abstractmethod
    def load_data(self) -> None: 
        '''load prepared data at training begin , only load data once in a fitting'''
        self.model_date_list : np.ndarray
        self.test_full_dates : np.ndarray
        self.datas : ModuleData
    @abstractmethod
    def setup(self , *args , **kwargs) -> None: 
        '''create train / valid / test dataloaders , perform in every different model_date / model_num'''
        self.d0 : int
        self.d1 : int
        self.y_std : torch.Tensor
        self.early_test_dates : np.ndarray
        self.model_test_dates : np.ndarray
    @abstractmethod
    def train_dataloader(self)  -> Iterator[BatchData]: '''return train dataloaders'''
    @abstractmethod
    def val_dataloader(self)    -> Iterator[BatchData]: '''return valid dataloaders'''
    @abstractmethod
    def test_dataloader(self)   -> Iterator[BatchData]: '''return test dataloaders'''
    @abstractmethod
    def predict_dataloader(self)-> Iterator[BatchData]: '''return predict dataloaders'''
    def on_before_batch_transfer(self , batch , dataloader_idx = None): return batch
    def transfer_batch_to_device(self , batch , device = None , dataloader_idx = None): return batch
    def on_after_batch_transfer(self , batch , dataloader_idx = None): return batch
    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict  = {}
        self.loader_param = self.LoaderParam()
    def prev_model_date(self , model_date):
        prev_dates = self.model_date_list[self.model_date_list < model_date]
        return max(prev_dates) if len(prev_dates) > 0 else -1
    def next_model_date(self , model_date):
        late_dates = self.model_date_list[self.model_date_list > model_date]
        return min(late_dates) if len(late_dates) > 0 else max(self.test_full_dates) + 1
        
    def batch_date(self , batch_data : BatchData):
        return self.y_date[batch_data.i.cpu()[:,1]]
    
    def batch_secid(self , batch_data : BatchData):
        return self.y_secid[batch_data.i.cpu()[:,0]]
    
    def batch_date0(self , batch_data : BatchData):
        batch_date = self.batch_date(batch_data)
        assert (batch_date == batch_date[0]).all() , batch_date
        return batch_date[0]

    def batch_label(self , batch_data : BatchData):
        label = batch_data.y.cpu().squeeze().numpy()
        if label.ndim == 1:
            return label
        elif label.ndim == 2:
            return label[:,0]
        else:
            raise ValueError(f'label shape {label.shape} is not supported')

    @property
    def stage(self) -> Literal['fit' , 'test' , 'predict' , 'extract']:
        return self.loader_param.stage

    @property
    def model_date(self) -> int:
        return self.loader_param.model_date

    @property
    def seq_lens(self) -> dict[str,int]:
        return self.loader_param.seqlens

    @property
    def y_secid(self) -> np.ndarray:
        return self.datas.y.secid

    @property
    def y_date(self) -> np.ndarray:
        return self.datas.y.date[self.d0:self.d1]

    @property
    def day_len(self) -> int:
        return self.d1 - self.d0

    @property
    def data_step(self) -> int:
        return self.config.train_data_step if self.stage in ['fit'] else 1

    @dataclass
    class LoaderParam:
        stage : Literal['fit' , 'test' , 'predict' , 'extract'] | Any = None
        model_date : int | Any = None
        seqlens : dict[str,int] | Any = None
        extract_backward_days : int | Any = None
        extract_forward_days  : int | Any = None

        def __post_init__(self):
            assert self.stage is None or self.stage in ['fit' , 'test' , 'predict' , 'extract'] , self.stage
            assert self.model_date is None or self.model_date > 0 , self.model_date
            assert self.seqlens is None or self.seqlens , self.seqlens
            if self.seqlens is None:
                self.seqlens = {}
            if self.stage != 'extract':
                self.extract_backward_days = None 
                self.extract_forward_days  = None
        
    @property
    def device(self): return self.config.device

class BaseTrainer(ModelStreamLine):
    '''run through the whole process of training'''
    _instance : 'BaseTrainer | None' = None
    _raw_hooks : dict[str, Callable] = {}
    _hooks_wrapped : bool = False

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            HookWrapper.wrap(cls._instance)
            Proj.States.trainer = cls._instance
        return cls._instance
    
    @final
    def __init__(self , base_path = None , override : dict | None = None , schedule_name = None , **kwargs):
        with Logger.Paragraph('Stage [Setup]' , 2):
            self.init_config(base_path = base_path , override = override , schedule_name = schedule_name , **kwargs)
            self.init_data(**kwargs)
            self.init_model(**kwargs)
            self.init_callbacks(**kwargs)

    def __bool__(self): return True

    def __repr__(self): 
        return f'{self.__class__.__name__}(path={self.config.model_base_path.base})'
        
    @final
    def init_config(self , base_path = None , override : dict | None = None , schedule_name = None , **kwargs) -> None:
        '''initialized configuration'''
        with Logger.Paragraph('Config Setup' , 3):
            self.config = TrainConfig(base_path , override = override , schedule_name = schedule_name , **kwargs)
            self.config.print_out()
        self.status = TrainerStatus(self.config.train_max_epoch)
        self.record = TrainerPredRecorder(self)

    @abstractmethod
    def init_model(self , **kwargs): 
        '''initialized data_module'''
        self.model : BasePredictorModel

    @abstractmethod
    def init_callbacks(self , **kwargs): 
        '''initialized data_module'''
        self.callback : BaseCallBack

    @abstractmethod
    def init_data(self , **kwargs): 
        '''initialized data_module'''
        self.data : BaseDataModule

    @property
    def device(self): return self.config.device
    @property
    def metrics(self):  return self.config.metrics
    @property
    def checkpoint(self): return self.config.checkpoint
    @property
    def deposition(self): return self.config.deposition   
    @property
    def stage_queue(self): return self.config.stage_queue
    @property
    def batch_dates(self): return np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
    @property
    def batch_warm_up(self): 
        return len(self.data.early_test_dates) if self.status.stage == 'test' else 0
    @property
    def batch_aftermath(self): 
        return len(self.data.early_test_dates) + len(self.data.model_test_dates) if self.status.stage == 'test' else np.inf
    @property
    def batch_resumed(self): 
        if self.status.stage == 'test'  and self.batch_warm_up == 0 and self.config.is_resuming and Proj.Conf.Model.TRAIN.resume_test == 'last_pred_date':
            return sum(self.data.model_test_dates <= self.record.resumed_last_pred_date)
        else:
            return 0
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_submodel(self): return self.status.model_submodel
    @property
    def model_str(self): return f'{self.config.model_name}.{self.model_num}.{self.model_submodel}.{self.model_date}'
    @property
    def prev_model_date(self): return self.data.prev_model_date(self.model_date)
    @property
    def model_param(self): return self.config.Model.params[self.model_num]
    @property
    def model_submodels(self): return self.config.model_submodels
    @property
    def if_transfer(self): return self.config.train_trainer_transfer     
    @property
    def batch_output(self): return self.model.batch_output
    @property
    def html_catcher_export_path(self): 
        if 'fit' in self.stage_queue and 'test' in self.stage_queue:
            status = 'fitting_testing'
        elif 'test' in self.stage_queue:
            status = 'testing'
        elif 'fit' in self.stage_queue:
            status = 'fitting'
        else:
            status = 'unknown'
        return self.config.model_base_path.log(f'{self.config.model_name}_{status}.html')
    
    def main_process(self):
        '''Main stage of data & fit & test'''
        self.on_configure_model()

        if not self.stage_queue:
            Logger.error("stage_queue is empty , please check src.proj.Proj.States.trainer")
            raise Exception("stage_queue is empty , please check src.proj.Proj.States.trainer")

        if 'data' in self.stage_queue:
            with Logger.Paragraph('Stage [Data]' , 2):
                self.stage_data()

        if 'fit' in self.stage_queue:  
            with Logger.Paragraph('Stage [Fit]' , 2):
                self.stage_fit()

        if 'test' in self.stage_queue: 
            with Logger.Paragraph('Stage [Test]' , 2):
                self.stage_test()
        
        self.on_summarize_model()

        return self

    def go(self):
        '''alias of main_process'''
        return self.main_process()

    def stage_data(self):
        '''stage of loading model data'''
        self.on_before_data_start()
        self.on_data_start()
        self.data.load_data()
        self.on_data_end()
        self.on_after_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.on_before_fit_start()
        self.on_fit_start()
        for self.status.model_date , self.status.model_num in self.iter_model_num_date():
            if self.status.model_num == 0:
                self.on_fit_model_date_start()
            if self.status.fit_iter_num == 0:
                Logger.note(f'First Iterance: ({self.status.model_date} , {self.status.model_num})')
            self.on_fit_model_start()
            self.model.fit()
            self.on_fit_model_end()
            if self.status.model_num == self.config.model_num:
                self.on_fit_model_date_end()
        self.on_fit_end()
        self.on_after_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.on_before_test_start()
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.iter_model_num_date():
            if self.status.model_num == 0:
                self.on_test_model_date_start()
            self.on_test_model_start()
            self.model.test()
            self.on_test_model_end()
            if self.status.model_num == self.config.model_num_list[-1]:
                self.on_test_model_date_end()
        self.on_before_test_end()
        self.on_test_end()
        self.on_after_test_end()

    def iter_model_num_date(self): 
        '''iter of model_date and model_num , considering is_resuming'''
       
        model_iter = list(itertools.product(self.data.model_date_list , self.config.model_num_list))
        assert self.status.stage in ['fit' , 'test'] , self.status.stage
        num_all_models = len(model_iter)
        iter_info = f'In stage [{self.status.stage}], number of all models (model_date x model_num) is {num_all_models}, '
        if self.config.is_resuming:
            if self.status.stage == 'fit':
                models_trained = np.full(len(model_iter) , True , dtype = bool)
                for i , (model_date , model_num) in enumerate(model_iter):
                    if not self.deposition.exists(model_num , model_date):
                        models_trained[max(i,0):] = False
                        break
                condition = ~models_trained
            elif self.status.stage == 'test':
                resumed_models = self.record.resumed_preds_models.groupby(['model_date' , 'model_num']).groups
                resumed = [(model_date , model_num) not in resumed_models for model_date , model_num in model_iter]
                condition = np.array(resumed)
            else:
                Logger.error(f'Invalid stage for resuming iter_model_num_date: {self.status.stage}')
                condition = np.full(len(model_iter) , True , dtype = bool)
            model_iter = FilteredIterable(model_iter , condition)
            iter_info += f'resuming {num_all_models - sum(condition)} models, {sum(condition)} to go!'
        #elif self.status.stage == 'test' and self.status.fitted_model_num <= 0:
        #    model_iter = []
        else:
            iter_info += f'{num_all_models} to go!'
        Logger.note(iter_info , vb_level = 2)
        return model_iter

    def iter_model_submodels(self):
        for self.status.model_submodel in self.model_submodels: 
            self.on_test_submodel_start()
            yield self.status
            self.on_test_submodel_end()

    def iter_fit_epoches(self):
        while not self.status.fit_loop_breaker:
            self.on_fit_epoch_start()
            yield self.status
            self.on_before_fit_epoch_end()
            self.on_fit_epoch_end()

    def iter_train_dataloader(self , given_loader = None):
        self.dataloader = self.data.train_dataloader() if given_loader is None else given_loader
        self.on_train_epoch_start()
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_train_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_train_batch_end()
        self.on_train_epoch_end()

    def iter_val_dataloader(self , given_loader = None):
        self.dataloader = self.data.val_dataloader() if given_loader is None else given_loader
        self.on_validation_epoch_start()
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_validation_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_validation_batch_end()
        self.on_validation_epoch_end()

    def iter_test_dataloader(self , given_loader = None):
        self.dataloader = self.data.test_dataloader() if given_loader is None else given_loader
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_test_batch_end()

    def iter_predict_dataloader(self , given_loader = None):
        self.dataloader = self.data.predict_dataloader() if given_loader is None else given_loader
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_test_batch_end()

    def stack_model(self):
        '''temporaly save self to somewhere'''
        self.on_before_save_model()
        for submodel in self.model_submodels:
            model_dict = self.model.collect(submodel)
            self.deposition.stack_model(model_dict , self.model_num , self.model_date , submodel) 

    def save_model(self):
        '''save self to somewhere'''
        if self.metrics.better_attempt(self.status.best_attempt_metric): 
            self.stack_model()
        [self.deposition.dump_model(self.model_num , self.model_date , submodel) for submodel in self.model_submodels]

    def on_configure_model(self):  
        self.config.set_config_environment()
        
    def on_fit_model_start(self):
        self.data.setup('fit' , self.model_param , self.model_date)

    def on_fit_model_end(self): 
        self.save_model()

    def on_fit_epoch_start(self): ...

    def on_fit_epoch_end(self): ...

    def on_train_epoch_start(self):
        self.metrics.new_epoch(**self.status.status)

    def on_train_epoch_end(self):
        self.metrics.collect_epoch()
    
    def on_validation_epoch_start(self):
        self.metrics.new_epoch(**self.status.status)

    def on_validation_epoch_end(self):
        self.metrics.collect_epoch()
    
    def on_test_model_start(self):
        self.data.setup('test' , self.model_param , self.model_date)
    
    def on_test_submodel_start(self):
        self.metrics.new_epoch(**self.status.status)
        
    def on_test_submodel_end(self): 
        self.metrics.collect_epoch()

    def on_test_batch_start(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data.batch_date0(self.batch_data)) 

    @property
    def penalty_kwargs(self): return {}
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)
    @staticmethod
    def available_modules(module_type : Literal['nn' , 'booster' , 'all'] = 'all'):
        return AlgoModule.available_modules(module_type)
    @staticmethod
    def available_models(short_test : bool | Literal['both'] = 'both'):
        bases = [model.name for model in PATH.model.iterdir() if model.is_dir() and not model.name.startswith('.')]
        if short_test == 'both':
            return bases
        elif short_test:
            return [model for model in bases if model.endswith('ShortTest')]
        else:
            return [model for model in bases if not model.endswith('ShortTest')]

class ModelStreamLineWithTrainer(ModelStreamLine):
    def bound_with_trainer(self , trainer): 
        self.trainer : BaseTrainer | Any = trainer
        return self

    @property
    def config(self): return self.trainer.config
    @property
    def status(self):  return self.trainer.status
    @property
    def metrics(self):  return self.config.metrics
    @property
    def checkpoint(self): return self.config.checkpoint
    @property
    def deposition(self): return self.config.deposition
    @property
    def device(self): return self.config.device
    @property
    def data(self): return self.trainer.data
    @property
    def batch_data(self): return self.trainer.batch_data
    @property
    def batch_idx(self): return self.trainer.batch_idx
    @property
    def model_date(self): return self.trainer.model_date
    @property
    def model_num(self): return self.trainer.model_num
    @property
    def model_submodel(self): return self.trainer.model_submodel
    @property
    def model_str(self): return self.trainer.model_str

class BaseCallBack(ModelStreamLineWithTrainer):
    CB_ORDER : int = 0
    CB_KEY_PARAMS : list[str] = []
    def __init__(self , trainer , turn_off = False) -> None:
        self.bound_with_trainer(trainer)
        self.turn_off : bool = turn_off
        self.__hook_stack = []

        for param in self.CB_KEY_PARAMS:
            assert hasattr(self , param) , f'{self.__class__.__name__} has no attribute {param}'

    def print_info(self , vb_level : int = 2 , **kwargs):
        args = {k:getattr(self , k) for k in self.CB_KEY_PARAMS} | kwargs
        info = f'Callback {self.__class__.__name__}' + '({})'.format(','.join([f'{k}={v}' for k,v in args.items()])) 
        if self.__class__.__doc__: 
            info += f' , {self.__class__.__doc__}'
        Logger.stdout(info , vb_level = vb_level)
        return self

    def __enter__(self): 
        self.__hook_stack.append(self.trace_hook_name())
        self.at_enter(self.__hook_stack[-1])
    def __exit__(self , *args):
        self.at_exit(self.__hook_stack.pop())
    def __bool__(self):
        return not self.turn_off
    def at_enter(self , hook : str , vb_level : int = Proj.vb.max):  
        Logger.stdout(f'{hook} of callback {self.__class__.__name__} start' , vb_level = vb_level)
    def at_exit(self , hook : str , vb_level : int = Proj.vb.max): 
        getattr(self , hook)()
        Logger.stdout(f'{hook} of callback {self.__class__.__name__} end' , vb_level = vb_level)

    def trace_hook_name(self) -> str:
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): 
            env = getattr(env , 'f_back')
        return env.f_code.co_name

    @property
    def model(self): return self.trainer.model

class BasePredictorModel(ModelStreamLineWithTrainer):
    '''a group of ensemble models , of same net structure'''
    AVAILABLE_CALLBACKS = []
    COMPULSARY_CALLBACKS = ['BasicTestResult' , 'DetailedAlphaAnalysis' , 'StatusDisplay']
    
    def __init__(self, *args , **kwargs) -> None:
        self.reset()
        self.model_dict = ModelDict()

    def __call__(self , input : BatchData | torch.Tensor | Any | int | None , *args , **kwargs):
        if isinstance(input , int):
            from src.res.model.data_module import DataModule
            input = DataModule.get_date_batch_data(self.config , input)
            output = self.forward(input , *args , **kwargs)
        elif input is None or len(input) == 0:
            output = None
        else:
            output = self.forward(input , *args , **kwargs)
        return BatchOutput(output)
    
    def __repr__(self): 
        return f'{self.__class__.__name__}(model_full_name={self.model_full_name})'
    
    def multiloss_params(self): return {}

    def reset(self):
        self.trainer : BaseTrainer | Any = None
        self._config : TrainConfig | Any = None
        return self

    def bound_with_config(self , config : TrainConfig):
        assert self.trainer is None , 'Cannot bound with config if bound with trainer first'
        self._config = config
        return self.init_utils()

    def bound_with_trainer(self , trainer : BaseTrainer):
        self.reset()
        self.trainer = trainer
        return self.init_utils()
    
    def init_utils(self):
        self.config.init_utils()
        return self

    @classmethod
    def create_from_trainer(cls , trainer : BaseTrainer):
        return cls().bound_with_trainer(trainer)

    @property
    def config(self):
        return self.trainer.config if self.trainer else self._config
    @property
    def model_full_name(self):
        return f'{self.config.model_name}@{self.model_num}@{self.model_date}@{self.model_submodel}'
    @property
    def model_num(self):
        return self.trainer.model_num if self.trainer else self._model_num
    @property
    def model_date(self):
        return self.trainer.model_date if self.trainer else self._model_date
    @property
    def model_submodel(self):
        return self.trainer.model_submodel if self.trainer else self._model_submodel
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    
    def load_model_file(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when fitting/testing new model'''
        if model_num is not None: 
            self._model_num  = model_num
        else: 
            model_num = self.model_num
        if model_date is not None: 
            self._model_date = model_date
        else: 
            model_date = self.model_date
        if submodel is not None: 
            self._model_submodel = submodel
        else: 
            submodel = self.model_submodel
        assert self.deposition.exists(model_num , model_date , submodel) , (model_num , model_date , submodel)
        return self.deposition.load_model(model_num , model_date , submodel)
    
    @abstractmethod
    def new_model(self , *args , **kwargs):
        '''call when fitting new model'''
        self.optimizer : Any
        return self
    @abstractmethod
    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when testing new model'''
        return self
    @abstractmethod
    def forward(self , batch_data : BatchData | torch.Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
    @abstractmethod
    def fit(self) -> None:
        '''fit the model inside'''
    @abstractmethod
    def collect(self , submodel = 'best' , *args) -> ModelDict: 
        '''collect model params, called before stacking model'''

    def test(self):
        '''test the model inside'''
        for _ in self.trainer.iter_model_submodels():
            self.load_model(submodel=self.model_submodel)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

    def metric_kwargs(self):
        kwargs = {}
        kwargs['pred'] = self.batch_output.pred
        kwargs['label'] = self.batch_data.y
        kwargs['weight'] = self.batch_data.w
        kwargs.update(self.multiloss_params())
        kwargs.update(self.batch_output.other)
        return kwargs
    
    def batch_forward(self) -> None: 
        if self.trainer.batch_idx < self.trainer.batch_resumed or self.trainer.batch_idx >= self.trainer.batch_aftermath: 
            self.batch_output = BatchOutput()
        else:
            self.batch_output = self(self.batch_data)

    def batch_metrics(self) -> None:
        if self.batch_output.is_empty or self.trainer.batch_idx < self.trainer.batch_warm_up: 
            return
        batch_key = self.trainer.batch_dates[self.trainer.batch_idx] if self.status.stage == 'test' else self.trainer.batch_idx
        self.metrics.calculate(self.status.dataset , **self.metric_kwargs()).collect_batch(key = batch_key)

    def batch_backward(self) -> None:
        if self.batch_data.is_empty: 
            return
        assert self.status.dataset == 'train' , self.status.dataset
        self.trainer.on_before_backward()
        self.optimizer.backward(self.metrics.output)
        self.trainer.on_after_backward()

