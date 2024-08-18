import gc , torch
import numpy as np
import torch

from abc import abstractmethod
from torch import Tensor

from typing import Any , Literal , Optional

from ..models import ModelEnsembler
from ...callback import CallBackManager
from ...classes import BaseDataModule , BaseTrainer
from ...util import (Checkpoint , Deposition , Device , Logger , Metrics , TrainConfig ,
                     BufferSpace , Device , Storage)
from ....basic import CONF
from ....data import DataBlockNorm , DataProcessor , ModuleData
from ....func import tensor_standardize_and_weight , BigTimer

class DataModule(BaseDataModule):
    @abstractmethod
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Optional[Tensor]) -> None: ...

    @abstractmethod
    def split_sample(self , stage , valid : Tensor , index0 : Tensor , index1 : Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list]: ...
    
    def __init__(self , config : Optional[TrainConfig] = None , use_data : Literal['fit','predict','both'] = 'fit'):
        '''
        1. load Package of BlockDatas of x , y , norms and index
        2. Setup model_date dataloaders
        3. Buffer dict for dynamic nn's
        '''
        self.config  : TrainConfig = TrainConfig.load() if config is None else config
        self.use_data : Literal['fit','predict','both'] = use_data
        self.device  = Device()
        self.storage = Storage('mem' if self.config['mem_storage'] else 'disk')
        self.buffer  = BufferSpace(self.device)

    def load_data(self):
        self.datas = ModuleData.load(self.data_type_list, self.config['data.labels'], 
                                     fit = self.use_data != 'predict' , predict = self.use_data != 'fit' ,
                                     dtype = self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)
        if self.use_data == 'predict':
            self.model_date_list = self.datas.date[0]
            self.test_full_dates = self.datas.date[1:]
        else:
            self.model_date_list = self.datas.date_within(self.config['beg_date'] , self.config['end_date'] , self.config['interval'])
            self.test_full_dates = self.datas.date_within(self.config['beg_date'] , self.config['end_date'])[1:]

        self.static_prenorm_method = {}
        for mdt in self.data_type_list: 
            method : dict[str,bool] = self.config['data.prenorm'].get(mdt , {})
            method['divlast']  = method.get('divlast' , True) and (mdt in DataBlockNorm.DIVLAST)
            method['histnorm'] = method.get('histnorm', True) and (mdt in DataBlockNorm.HISTNORM)
            if not CONF.SILENT: print(f'Pre-Norming method of [{mdt}] : {method}')
            self.static_prenorm_method[mdt] = method

        self.reset_dataloaders()
        return self
    
    @property
    def data_type_list(self):
        '''get data type list (abbreviation)'''
        return [ModuleData.abbr(data_type) for data_type in self.config.data_type_list]
    
    @staticmethod
    def prepare_data(data_types : Optional[list[str]] = None):
        DataProcessor.main(predict = False , data_types = data_types)
        DataProcessor.main(predict = True , data_types = data_types)

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , 
              param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1 , none_valid = False) -> None:
        if self.use_data == 'predict': stage = 'predict'
        seqlens : dict = {key:param['seqlens'][key] for key in self.data_type_list}
        seqlens.update({k:v for k,v in param.items() if k.endswith('_seq_len')})
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        
        self.stage = stage
        x_keys = self.data_type_list
        y_keys = [k for k in seqlens.keys() if k not in x_keys]
        self.seqs = {k:seqlens.get(k , 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([v for k,v in self.seqs.items() if k in y_keys]) if y_keys else 1
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys]) if x_keys else 1
        self.seq0 = self.seqx + self.seqy - 1

        if stage == 'fit':
            model_date_col = (self.datas.date < model_date).sum()
            step_interval = self.config['input_step_day']
            d0 = max(0 , model_date_col - self.config['skip_horizon'] - self.config['input_span'] - self.seq0)
            d1 = max(0 , model_date_col - self.config['skip_horizon'])
        elif stage in ['predict' , 'test']:
            if stage == 'predict': self.model_date_list = np.array([model_date])
            next_model_date = self.next_model_date(model_date)
            step_interval  = 1

            before_test_dates = self.datas.date[self.datas.date < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::step_interval]
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // step_interval:] if self.seqy > 1 else test_dates[-1:-1]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])
            
            d0 = max(np.where(self.datas.date == test_dates[0])[0][0] - self.seqx + 1 , 0)
            d1 = np.where(self.datas.date == test_dates[-1])[0][0] + 1
        else:
            raise KeyError(stage)

        self.day_len  = d1 - d0
        self.step_len = (self.day_len - self.seqx + 1) // step_interval
        if stage in ['predict' , 'test']: assert self.step_len == len(test_dates) , (self.step_len , len(test_dates))
        self.step_idx = torch.flip(self.day_len - 1 - torch.arange(self.step_len) * step_interval , [0])
        self.date_idx = d0 + self.step_idx
        self.y_secid , self.y_date = self.datas.y.secid , self.datas.y.date[d0:d1]

        x = {k:Tensor(v.values)[:,d0:d1] for k,v in self.datas.x.items()}
        y = Tensor(self.datas.y.values)[:,d0:d1].squeeze(2)[...,:self.labels_n]

        self.y , _ = self.standardize_y(y , None , None , no_weight = True)

        if none_valid:
            w , valid = None , None
            y , _ = self.standardize_y(self.y , None , self.step_idx)
        else:
            valid = self.full_valid_sample(x , self.y , self.step_idx)
            y , w = self.standardize_y(self.y , valid , self.step_idx)

        self.y[:,self.step_idx] = y[:]
        self.static_dataloader(x , y , w , valid)

        gc.collect() 
        torch.cuda.empty_cache()

    def full_valid_sample(self , x_data : dict[str,Tensor] , y : Tensor , index1 : Tensor , **kwargs) -> Tensor:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is divlast
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        valid = self.valid_sample(y , index1) if self.stage == 'train' else torch.ones(len(y),len(index1)).to(torch.bool)
        for k , x in x_data.items(): 
            valid = valid * self.valid_sample(x , index1 , self.seqs[k] , k in DataBlockNorm.DIVLAST)
        for k , x in kwargs.items(): 
            valid = valid * self.valid_sample(x , index1 , self.seqs[k])
        return valid
    
    @staticmethod
    def valid_sample(data : Tensor , index1 : Tensor , rolling_window = 1 , endpoint_nonzero = False):
        '''return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims'''
        start_idx = rolling_window
        assert start_idx > 0 , start_idx
        data = torch.cat([torch.zeros_like(data[:,:start_idx]) , data],dim=1).unsqueeze(2)
        sum_dim = tuple(range(2,data.ndim))
        
        invalid_samp = data[:,index1 + start_idx].isnan().sum(sum_dim)
        for i in range(1 , start_idx): 
            invalid_samp += data[:,index1 - i + start_idx].isnan().sum(sum_dim)

        if endpoint_nonzero: 
            invalid_samp += (data[:,index1 + start_idx] == 0).sum(sum_dim)
        
        return (invalid_samp == 0)
     
    def standardize_y(self , y : Tensor , valid : Optional[Tensor] , index1 : Optional[Tensor] , no_weight = False) -> tuple[Tensor , Optional[Tensor]]:
        '''standardize y and weight'''
        y = y[:,index1].clone() if index1 is not None else y.clone()
        if valid is not None: y.nan_to_num_(0)[~valid] = torch.nan
        return tensor_standardize_and_weight(y , 0 , self.config.weight_scheme(self.stage , no_weight))

    @staticmethod
    def rolling_rotation(x : Tensor , rolling : int , index0 , index1 , dim = 1 , squeeze_out = True) -> Tensor:
        '''rotate [stock , date , inday , feature] to [sample , rolling , inday , feature]'''
        assert x.ndim == 4 , x.ndim
        assert len(index0) == len(index1) , (len(index0) , len(index1))
        try:
            new_x = x.unfold(dim , rolling , 1)[index0 , index1 + 1 - rolling].permute(0,3,1,2) # [stock , rolling , inday , feature]
        except MemoryError:
            new_x = torch.stack([x[index0 , index1 + i + 1 - rolling] for i in range(rolling)],dim=dim)
        if squeeze_out: new_x = new_x.squeeze(-2)
        return new_x
        
    def prenorm(self , x : Tensor, key : str) -> Tensor:
        '''
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        '''
        if self.static_prenorm_method[key]['divlast']:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if self.static_prenorm_method[key]['histnorm']:
            x = x - self.datas.norms[key].avg[-x.shape[-2]:]
            x = x / (self.datas.norms[key].std[-x.shape[-2]:] + 1e-6)
        return x

class TrainerModule(BaseTrainer):
    '''run through the whole process of training'''
    def init_config(self , config_path = None , **kwargs) -> None:
        self.config : TrainConfig = TrainConfig.load(config_path = config_path , do_parser = True , par_args = kwargs)
        self.stage_queue = self.config.stage_queue
    def init_utilities(self , **kwargs) -> None: 
        self.logger     = Logger()
        self.device     : Device = Device()
        self.checkpoint : Checkpoint = Checkpoint(self.config)
        self.deposition : Deposition = Deposition(self.config)
        self.metrics    : Metrics = Metrics(self.config)
        self.callbacks  : CallBackManager= CallBackManager.setup(self)
        self.model      : ModelEnsembler = ModelEnsembler.setup(self)
    
    @property
    def model_param(self): return self.config.Model.params[self.model_num]
    @property
    def model_types(self): return self.config['model.types']
    @property
    def if_transfer(self): return bool(self.config['train.trainer.transfer'])
    @property
    def model_iter(self): return self.deposition.model_iter(self.status.stage , self.data.model_date_list)
    
    def go(self):
        with BigTimer(self.logger.critical , 'Main Process'):
            self.main_process()
        return self