from __future__ import annotations

import numpy as np
import pandas as pd
import gc , torch

from functools import cached_property
from typing import Any , Literal , Callable

from src.proj import CALENDAR , Const, MACHINE , BaseClass
from src.data import PreProcessorTask , ModuleData

from src.func import match_values
from src.res.model.util.core import BatchInput
from src.res.model.util.config import ModelConfig
from src.res.model.util.storage import TorchFileStorage , StoredTorchFileLoader
from src.res.model.util.trainer import BaseTrainer
from .dynamic_buffer import DynamicDataBuffer
from .batch_input_loader import DataloaderParam , BatchInputLoader
from .data_callback import DataCallbacks
from .operations import PrenormOperator , DataOperator

__all__ = ['DataModule']

class DataModule(BaseClass.BoundLogger):
    """
    DataModule for model fitting / testing / predicting
    """
   
    def __init__(
        self , config : ModelConfig | None = None , 
        use_data : Literal['fit','predict','both'] = 'fit' , * , 
        indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.config   : ModelConfig = config or ModelConfig(stage=0)
        self._use_data : Literal['fit','predict','both'] = use_data

    @classmethod
    def initialize(cls , trainer_or_config : BaseTrainer | ModelConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit' , *args , **kwargs):
        if trainer_or_config is None:
            config = ModelConfig(stage=0)
            vb_level = kwargs.pop('vb_level' , 1)
        elif isinstance(trainer_or_config , BaseTrainer):
            config = trainer_or_config.config
            use_data = trainer_or_config.use_data
            vb_level = vb_level = kwargs.pop('vb_level' , trainer_or_config.vb_level + 1)
        elif isinstance(trainer_or_config , ModelConfig):
            config = trainer_or_config
            vb_level = kwargs.pop('vb_level' , 1)
        else:
            raise ValueError(f'Invalid trainer_or_config: {trainer_or_config}')
        data = cls(config , use_data = use_data , *args , vb_level = vb_level , **kwargs)
        if isinstance(trainer_or_config , BaseTrainer):
            for hook in ['on_before_batch_transfer' , 'on_after_batch_transfer']:
                data.register_callbacks(hook , *trainer_or_config.callback.get_implemented_hook_callables(hook))
        return data

    def print_out(self , vb_level : Any = 2 , min_key_len = 30):
        self.logger.stdout_pairs({'Use Data' : self.use_data} , title = 'Module Data Initiated:' , vb_level = vb_level , min_key_len = min_key_len)

    def __repr__(self): 
        keys =  self.input_keys
        if len(keys) >= 5: 
            keys_str = f'[{keys[0]},...,{keys[-1]}({len(keys)})]'
        else:
            keys_str = str(keys)
        return f'{self.__class__.__name__}(model_name={self.config.model_name},use_data={self.use_data},datas={keys_str})'    
    
    def load_data(self):
        '''load prepared data at training begin , only load data once in a fitting'''
        self.datas = ModuleData.initialize(
            self.input_keys_data + self.input_keys_factor ,  self.config.labels , 
            use_data = self._use_data ,
            factor_names = self.config.input_factor_names , 
            factor_start_dt = self.factor_start_dt , 
            factor_end_dt = self.factor_end_dt , 
            filter_secid = self.config.input_filter_secid , 
            filter_date = self.config.input_filter_date , 
            dtype = self.config.precision)

        self.datas.load()
        self.logger.stdout(f'Data loaded , shape: {self.datas.shape}' , vb = 10)
        
        self.config.update_data_param(self.datas.x)
        
        self.set_critical_dates()
        self.loader_dict : dict[str , StoredTorchFileLoader]  = {}
        self.loader_dates : dict[str , list[int]] = {}
        self.loader_param = DataloaderParam()
        self.data_operator = DataOperator(self.config , self.loader_param)
        self.prenorm_operator = PrenormOperator(self.config , self.datas.norms)
        self.labels = self.data_operator.standardize_y(self.datas.y.values.squeeze(2).clone() , no_weight = True)[0]

        if self.empty_x:
            self.logger.alert2(f'DataModule got empty x , fit and test stage will be skipped')
            self.logger.note(f'{self.input_type} input keys: {self.input_keys}' , vb = 1)
            if 'fit' in self.config.queue_of_stages:
                self.config.queue_of_stages.remove('fit')
            if 'test' in self.config.queue_of_stages:
                self.config.queue_of_stages.remove('test')
        return self

    def set_critical_dates(self):
        '''set critical dates for model date list and test full dates'''
        dates = self.datas.date_within(self.beg_date , self.end_date)
        # check if dates is align with CALENDAR
        assert len(dates) > 0 , f'dates is empty: {self.beg_date} , {self.end_date}'
        calendar_dates = CALENDAR.range(min(dates) , max(dates) , 'td')
        if not np.isin(dates , calendar_dates).all() or not np.isin(calendar_dates , dates).all():
            self.logger.error(f'dates is not align with calendar dates!')
            if len(np.setdiff1d(dates , calendar_dates)) > 0:
                self.logger.alert1(f'dates not in calendar dates: {np.setdiff1d(dates , calendar_dates)}' , idt = 1 , vb = 1)
            if len(np.setdiff1d(calendar_dates , dates)) > 0:
                self.logger.alert1(f'calendar dates not in dates: {np.setdiff1d(calendar_dates , dates)}' , idt = 1 , vb = 1)
            if not MACHINE.platform_coding:
                raise ValueError(f'dates is not align with calendar dates!')
        self.data_dates = dates
        
        if self.config.is_null_model:
            # previos month end (use calendar date)
            self.test_full_dates = dates
            self.model_date_list = CALENDAR.td_array(CALENDAR.cd_array(np.unique(dates // 100) * 100 + 1 , -1))
        else:
            self.test_full_dates = dates[1:]
            if self.use_data == 'predict':
                self.model_date_list = dates[:1]
            else:
                self.model_date_list = dates[::self.config.interval]

    def setup(self, stage : Literal['fit' , 'test' , 'predict' , 'retrospective'] , 
              param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1 , **kwargs) -> None:
        """
        setup data module for a given stage
        other kwargs:
            retro_start_date : int | None = None # start date for retrospective data , None means year start of model date
            retro_end_date : int | None = None # end date for retrospective data , None means year end of model date
        """
        if self.setup_new_param(stage , param , model_date , **kwargs):
            self.setup_loader_kwargs()
            self.setup_loader_inputs()
            self.setup_loader_create()

    def setup_new_param(
            self , stage : Literal['fit' , 'test' , 'predict' , 'retrospective'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date : int = -1 , retro_start_date : int | None = None , retro_end_date : int | None = None
        ) -> bool:
        assert self.use_data in ['fit' , 'both'] or stage in ['predict' , 'test' , 'retrospective'] , (self.use_data , stage)
        slens = self.config.seq_lens | param.get('seqlens',{})
        slens = {key:int(val) for key,val in slens.items() if key in self.input_keys}
        slens.update({key:int(val) for key,val in param.items() if key.endswith('_seq_len')})
        
        assert slens , (self.config.seq_lens | param.get('seqlens',{}) , self.input_keys)
        loader_param = DataloaderParam(stage , model_date , slens , retro_start_date , retro_end_date)
        if self.loader_param == loader_param: 
            return False
        else:
            self.loader_param = loader_param
            self.data_operator = DataOperator(self.config , loader_param)
            return True

    def setup_loader_kwargs(self):
        assert all(k > 0 for k in self.seq_lens.values()) , self.seq_lens
        assert all(k > 0 for k in self.seq_steps.values()) , self.seq_steps
        y_keys = [k for k in self.seq_lens.keys() if k not in self.input_keys]
        y_extend = max([self.seq_lens[y] for y in y_keys]) if y_keys else 1
        x_extend = max([self.seq_lens[x] * self.seq_steps[x] for x in self.input_keys]) if self.input_keys else 1
        d_extend = x_extend + y_extend - 1

        match self.stage:
            case 'fit':
                model_date_col = (self.datas.date < self.model_date).sum()
                self.d0 = max(0 , model_date_col - self.config.skip_horizon - self.config.window - d_extend)
                self.d1 = max(0 , model_date_col - self.config.skip_horizon)
            case 'predict' | 'test' | 'retrospective':
                if self.stage == 'retrospective':
                    start_date = self.loader_param.retro_start_date
                    end_date = self.loader_param.retro_end_date
                    possible_dates = self.datas.date
                else:
                    start_date = max(CALENDAR.cd(self.model_date , 1) , CALENDAR.td(self.config.resumed_max_pred_date , -1).as_int())
                    end_date = self.next_model_date(self.model_date)
                    possible_dates = self.test_full_dates
                    
                before_dates = self.datas.date[self.datas.date < min(possible_dates)][-y_extend:]
                possible_dates = np.concatenate([before_dates , possible_dates])
                self.early_test_dates = possible_dates[possible_dates < start_date][-(y_extend-1):] if y_extend > 1 else possible_dates[-1:-1]
                self.model_test_dates = possible_dates[(possible_dates >= start_date) & (possible_dates <= end_date)]
                test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])
                
                if test_dates.size == 0:
                    self.d0 = len(self.datas.date) - x_extend
                    self.d1 = len(self.datas.date) - 1
                else:
                    self.d0 = max(np.where(self.datas.date == test_dates[0])[0][0] - x_extend + 1 , 0)
                    self.d1 = np.where(self.datas.date == test_dates[-1])[0][0] + 1
                test_dates = self.datas.date[self.d0 + x_extend - 1:self.d1]
            case _:
                raise KeyError(self.stage)

        self.step_len = (self.day_len - x_extend + 1) // self.data_step
        if self.step_len < 0:
            self.logger.error( 
                f'Step length is less than 0 ({self.step_len}) , stage: {self.stage} , '
                f'd0: {self.d0} , d1: {self.d1} , day_len: {self.day_len} , ' 
                f'data_len: {len(self.datas.date)} , x_extend: {x_extend} , data_step: {self.data_step}')
            self.logger.print_traceback_stack()
            raise ValueError(f'Step length is less than 0')
        self.step_idx = torch.flip(self.day_len - 1 - torch.arange(self.step_len) * self.data_step , [0])
        self.date_idx = self.d0 + self.step_idx

        if self.stage in ['predict' , 'test']:
            assert self.step_len == len(test_dates) , (self.step_len , len(test_dates))

    def setup_loader_inputs(self):
        '''additional input prepare for hidden input'''
        self.setup_loader_inputs_hidden()

    def setup_loader_inputs_hidden(self):
        '''additional input prepare for hidden input , calculate hiddens and temporary store them'''
        if self.input_type not in ['hidden' , 'combo'] or not self.input_keys_hidden: 
            return
        from src.res.model.model_module.application import ArchivedPredictorModel
        
        hd_dates = self.y_date[self.step_idx]

        for hd_key in self.input_keys_hidden:
            hd_model = ArchivedPredictorModel.from_model_str(hd_key)
            assert hd_model.model_dates.size > 0 , f'hidden model {hd_key} has no model dates'
            assert hd_model.model_dates[0] <= self.model_date , f'hidden model {hd_key} has no model date before {self.model_date}'
            model_date = hd_model.model_dates[hd_model.model_dates <= self.model_date][-1]
            bd_block = hd_model.hidden_block(hd_dates , model_date , align_secid = self.datas.secid , align_date = self.datas.date)
            self.datas.x[hd_key] = bd_block
        self.config.update_data_param(self.datas.x)
    
    def setup_loader_create(self) -> None:
        if self.day_len == 0:
            self.emptry_dataloader()
            return

        x_full = {k:v.values[:,self.d0:self.d1] for k,v in self.datas.x.items()}
        self.y_std = self.labels[:,self.d0:self.d1]

        self.logger.stdout(f'datas shape: {self.datas.shape}')
        self.logger.stdout(f'x_full: {x_full.keys()}')
        self.logger.stdout(f'x_full shapes: {[x.shape for x in x_full.values()]}')
        self.logger.stdout(f'y_std: {self.y_std.shape}')

        x_shapes = [x.shape[:2] for x in x_full.values()]
        assert all(x == self.y_std.shape[:2] for x in x_shapes) , (x_shapes , self.y_std.shape)

        valid_x = x_full if self.config.module_type == 'nn' else {}
        valid_y = self.y_std if self.is_fitting else None

        valid_sampled = self.valid_position(valid_x , valid_y , self.step_idx , all_valid=(self.config.module_type == 'nn'))
        y_sampled , w_sampled = self.data_operator.standardize_y(self.y_std , valid_sampled , self.step_idx)
        # since in fit stage , step_idx can be larger than 1 , different valid and result may occur
        self.y_std[:,self.step_idx] = y_sampled[:]
        self.static_dataloader(x_full , y_sampled , w_sampled , valid_sampled)

        if self.config.gc_collect_each_model:
            gc.collect() 
            torch.cuda.empty_cache()

    def train_dataloader(self)   -> BatchInputLoader: 
        return BatchInputLoader(self.loader_dict['train'] , self , desc = 'Train')
    def val_dataloader(self)     -> BatchInputLoader: 
        return BatchInputLoader(self.loader_dict['valid'] , self , desc = 'Valid')
    def test_dataloader(self)    -> BatchInputLoader: 
        return BatchInputLoader(self.loader_dict['test'] , self , desc = 'Test')
    def predict_dataloader(self) -> BatchInputLoader: 
        return BatchInputLoader(self.loader_dict['predict'] , self , tqdm = False , desc = 'Predict')
    def retrospective_dataloader(self) -> BatchInputLoader: 
        return BatchInputLoader(self.loader_dict['retrospective'] , self , tqdm = False , desc = 'Retrospective')

    @cached_property
    def data_callbacks(self) -> DataCallbacks:
        return DataCallbacks()
    def register_callbacks(self , hook_name : str , *callbacks : Callable):
        self.data_callbacks.register_callbacks(hook_name , *callbacks)
    def on_before_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        return self.data_callbacks.on_before_batch_transfer(batch)
    def on_after_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        return self.data_callbacks.on_after_batch_transfer(batch)
    def transfer_batch_to_device(self , batch : BatchInput , device = None) -> BatchInput:
        if self.config.module_type == 'nn':
            batch = batch.to(self.config.device if device is None else device)
        return batch

    def emptry_dataloader(self) -> None:
        if self.is_fitting:
            self.loader_dict['train'] = StoredTorchFileLoader(self.storage , [] , 'static')
            self.loader_dict['valid'] = StoredTorchFileLoader(self.storage , [] , 'static')
        else:
            self.loader_dict[self.stage] = StoredTorchFileLoader(self.storage , [] , 'static')
            self.loader_dates[self.stage] = []

    def prev_model_date(self , model_date) -> int:
        if self.stage == 'retrospective':
            return model_date - 10000
        prev_dates = self.model_date_list[self.model_date_list < model_date]
        return max(prev_dates) if len(prev_dates) > 0 else -1

    def next_model_date(self , model_date) -> int:
        if self.stage == 'retrospective':
            return model_date + 10000
        late_dates = self.model_date_list[self.model_date_list > model_date]
        return min(late_dates) if len(late_dates) > 0 else max(self.test_full_dates) + 1

    def y_label(self , dates : np.ndarray | list[int]) -> pd.DataFrame:
        labels : list[pd.DataFrame] = []
        for date in dates:
            label = self.label_of_date(date)
            if label.size > 0:
                labels.append(pd.DataFrame({
                    'secid' : self.datas.y.secid, 'date' : date,
                    'label' : label.flatten()
                }).dropna())
        return pd.concat(labels)

    @cached_property
    def labels_np(self) -> np.ndarray:
        return self.labels.cpu().numpy()

    def label_of_date(self , date : int) -> np.ndarray:
        return self.labels_np[:,self.datas.y.date == date][...,0].squeeze()

    def valid_position(self , x : dict[str,torch.Tensor] , y : torch.Tensor | None , index1 : torch.Tensor , all_valid = True) -> torch.Tensor | None:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window (seqlen * step) non-nan , end non-zero if in k is divlast
        others : rolling window non-nan , default as self.seqy
        '''
        valids : list[torch.Tensor] = []
        if x:
            finites = torch.stack([self.data_operator.finite_position(k , v , index1) for k , v in x.items()] , dim = -1)
            valids.append(finites.all(dim=-1) if all_valid else finites.any(dim=-1))
        if y is not None:
            valids.append(self.data_operator.finite_position(None , y, index1))
        if valids:
            return torch.stack(valids , dim = -1).all(dim = -1)
        else:
            return None

    def get_batch_input_of_date(self , date : int) -> BatchInput:
        assert self.stage in ['predict' , 'test' , 'retrospective'] , f'stage should be predict , test or retrospective, but got {self.stage}'
        return self.loader_dict[self.stage][self.loader_dates[self.stage].index(date)]
       
    def static_dataloader(self , x : dict[str,torch.Tensor] , y : torch.Tensor , w : torch.Tensor | None , valid : torch.Tensor | None) -> None:
        '''update loader_dict , save batch_input to f'PATH.batch.joinpath(f'{set_key}.{bnum}.pt')' and later load them'''   
        if valid is None: 
            valid = torch.ones(y.shape[:2] , dtype=torch.bool , device=y.device)
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.data_operator.split_sample(valid , index0 , index1)
        self.storage.del_group(self.stage)
        self.loader_dates[self.stage] = []
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test' , 'predict' , 'retrospective'] , set_key
            shuf_opt = self.config.shuffle_option if set_key == 'train' else 'static'
            batch_keys : list[str] = []
            for bnum , b_i in enumerate(set_samples):
                if b_i.numel() == 0:
                    continue
                assert torch.isin(b_i[:,1] , index1).all() , f'all b_i[:,1] must be in index1'
                index0 , xindex1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1) # here

                b_x = self.batch_data_x(x , index0 , xindex1)
                b_y = self.batch_data_y(y , index0 , yindex1)
                b_w = self.batch_data_y(w , index0 , yindex1)
                b_v = self.batch_data_y(valid , index0 , yindex1)

                batch_input = BatchInput(b_x , b_y , b_w , b_i , b_v , self.y_date , self.y_secid)

                batch_key = f'{set_key}.{bnum}'
                self.storage.save(batch_input , batch_key , group = self.stage)
                batch_keys.append(batch_key)
                if set_key in ['predict' , 'test' , 'retrospective']:
                    self.loader_dates[set_key].append(self.y_date[int(xindex1[0].item())])
                
            self.loader_dict[set_key] = StoredTorchFileLoader(self.storage , batch_keys , shuf_opt)

    def batch_data_x(self , x : dict[str,torch.Tensor] , index0 : torch.Tensor | np.ndarray , index1 : torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        datas = []
        for model_data_type , data in x.items():
            if data[index0,index1].isnan().all():
                self.logger.error(f'x valid_dates: {self.datas.x[model_data_type].valid_dates}')
                self.logger.error(f'date: {self.y_date[index1[0]]} , keys: {x.keys()}')
                self.logger.error(f'seq_lens: {self.seq_lens} , seq_steps: {self.seq_steps}')
                self.logger.error(f'early_test_dates: {self.early_test_dates}')
                raise ValueError(f'Get all nan in {model_data_type} at index {index0} , {index1}')
            data = self.data_operator.rolling_rotation(model_data_type , data , index0 , index1)
            data = self.prenorm_operator.prenorm(model_data_type , data)
            datas.append(data)
        return datas

    def batch_data_y(self , y : torch.Tensor | None, index0 : torch.Tensor | np.ndarray , index1 : torch.Tensor | np.ndarray) -> torch.Tensor | Any:
        if y is None:
            return None
        return y[index0 , index1]

    @property
    def use_data(self) -> Literal['fit','predict','both']:
        return self._use_data if not hasattr(self , 'datas') else self.datas.use_data

    @property
    def stage(self) -> Literal['fit' , 'test' , 'predict' , 'retrospective']:
        return self.loader_param.stage

    @cached_property
    def storage(self):
        return TorchFileStorage(self.config.mem_storage)

    @cached_property
    def buffer(self):
        return DynamicDataBuffer(self.config.device)

    @property
    def is_fitting(self): 
        return self.stage == 'fit'

    @property
    def input_keys(self) -> list[str]:
        input_keys = [key for value in self.config.input_keys_all.values() for key in value]
        if self.config.module_type == 'factor':
            input_keys.append('factor')
        return input_keys

    @property
    def input_keys_all(self) -> dict[str,list[str]]:
        input_keys = {key : [*value] for key , value in self.config.input_keys_all.items()}
        if self.config.module_type == 'factor':
            input_keys['factor'] = ['factor']
        input_keys = {key : value for key , value in input_keys.items() if value}
        assert len(input_keys) > 0 , (self.config.input_keys_all , self.config.module_type)
        return input_keys

    @property
    def input_keys_subkeys(self) -> dict[str,str]:
        try:
            subkeys = {f'{key}.{subkey}' : str(list(self.datas.x[subkey].feature)) for key , value in self.input_keys_all.items() for subkey in value if subkey in self.datas.x}
        except Exception as e:
            self.logger.alert2(f'Error getting input keys subkeys: {e}')
            self.logger.alert2(f'Input keys: {self.input_keys}')
            return {f'{key}.{subkey}' : subkey for key , value in self.input_keys_all.items() for subkey in value}
        return subkeys

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
        return self.config.fitting_step if self.stage in ['fit'] else 1

    @property
    def empty_x(self):
        return self.datas.empty_x and not self.input_keys_hidden

    @property
    def beg_date(self):
        return 19000101 if self.use_data == 'predict' else self.config.beg_date

    @property
    def end_date(self):
        return 99991231 if self.use_data == 'predict' else self.config.end_date

    @property
    def input_type(self) -> Literal['data' , 'hidden' , 'factor' , 'combo']: 
        return self.config.input_type

    @property
    def input_keys_data(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_data_types]

    @property
    def input_keys_factor(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_factor_types]

    @property
    def input_keys_hidden(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_hidden_types]

    @property
    def seq_steps(self) -> dict[str,int]:
        return self.config.seq_steps

    @property
    def min_test_date(self):
        if hasattr(self , 'test_full_dates'):
            return self.test_full_dates.min() if len(self.test_full_dates) > 0 else 99991231
        return ModuleData.min_data_date(self.input_keys_data + self.input_keys_factor , factor_names = self.config.input_factor_names) or 99991231

    @property
    def max_test_date(self):
        if hasattr(self , 'test_full_dates'):
            return self.test_full_dates.max() if len(self.test_full_dates) > 0 else 19000101
        return ModuleData.max_data_date(self.input_keys_data + self.input_keys_factor , factor_names = self.config.input_factor_names) or 19000101

    @property
    def factor_start_dt(self):
        beg_date = self.beg_date
        if self.config.is_null_model and self.config.is_resuming and Const.Model.resume_test:
            beg_date = max(beg_date , self.config.resumed_max_pred_date)
        return CALENDAR.td(beg_date , -1).as_int()

    @property
    def factor_end_dt(self):
        return self.end_date

    @property
    def no_date_to_test(self):
        return self.stage not in ['fit'] and len(self.model_test_dates) == 0

    @staticmethod
    def prepare_data(data_types : list[str] | None = None):
        PreProcessorTask.update(predict = False , data_types = data_types)
        PreProcessorTask.update(predict = True , data_types = data_types)
