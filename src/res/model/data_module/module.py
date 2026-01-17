import gc , torch
import numpy as np
import pandas as pd

from numpy.random import permutation
from torch.utils.data import BatchSampler
from typing import Any , Literal

from src.proj import PATH , Logger , CALENDAR
from src.data import DataBlockNorm , DataPreProcessor , ModuleData , DataBlock
from src.math import tensor_standardize_and_weight , match_values
from src.res.model.util import BaseBuffer , BaseDataModule , BatchData , TrainConfig , MemFileStorage , StoredFileLoader , HiddenPath

from .loader import BatchDataLoader

__all__ = ['DataModule']

def rolling_rotation(x : torch.Tensor , index0 : torch.Tensor | Any , index1 : torch.Tensor | Any , seqlen : int , step = 1 , dim = 1 , squeeze_out = True) -> torch.Tensor:
    '''rotate [stock , date , inday , feature] to [sample , rolling , inday , feature]'''
    assert x.ndim == 4 , x.ndim
    assert len(index0) == len(index1) , (index0 , index1)
    assert index1.max() < x.shape[dim] , (index1.max() , x.shape)
    assert index1.min() >= seqlen * step - 1 , (index1.min() , seqlen , step)
    
    try:
        start = max(0 , index1.min().item() - seqlen * step + 1)
        end = min(x.shape[dim] , index1.max().item() + 1)
        new_index1 = index1 - start + 1 - seqlen * step
        new_x = x[:,start:end].unfold(dim , seqlen * step , 1)[index0 , new_index1].\
            permute(0,3,1,2)[:,step-1::step] # [stock , seqlen (by step) , inday , feature]
    except MemoryError:
        new_x = torch.stack([x[index0 , index1 + (i + 1 - seqlen) * step] for i in range(seqlen)],dim=dim)
    
    assert new_x.shape[1] == seqlen , (new_x.shape[1] , seqlen)
    if squeeze_out: 
        new_x = new_x.squeeze(-2)
    return new_x

class DataModule(BaseDataModule):    
    """
    DataModule for model fitting / testing / predicting
    """
    _config_instance_for_batch_data : dict[TrainConfig,'DataModule'] = {}
    def __init__(self , config : TrainConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit'):
        '''
        1. load Package of BlockDatas of x , y , norms and index
        2. Setup model_date dataloaders
        3. Buffer dict for dynamic nn's
        use_data: 'fit' , 'predict' , 'both' 
            if 'predict' only load recent data
        '''
        self.config   = TrainConfig.default(stage=0) if config is None else config
        self.use_data = use_data
        self.storage  = MemFileStorage(self.config.mem_storage)
        self.buffer   = BaseBuffer(self.device)

    def __repr__(self): 
        keys = self.input_keys
        if len(keys) >= 5: 
            keys_str = f'[{keys[0]},...,{keys[-1]}({len(keys)})]'
        else:
            keys_str = str(keys)
        return f'{self.__class__.__name__}(model_name={self.config.model_name},use_data={self.use_data},datas={keys_str})'

    @property
    def empty_x(self):
        return self.datas.empty_x and not self.input_keys_hidden

    @staticmethod
    def prepare_data(data_types : list[str] | None = None):
        DataPreProcessor.main(predict = False , data_types = data_types)
        DataPreProcessor.main(predict = True , data_types = data_types)
       
    def load_data(self):
        self.datas = ModuleData.load(self.input_keys_data + self.input_keys_factor , 
                                     self.config.model_labels , 
                                     self.config.input_factor_names , 
                                     fit = self.use_data != 'predict' , predict = self.use_data != 'fit' ,
                                     dtype = self.config.precision, 
                                     factor_start_dt = CALENDAR.td(self.beg_date , -1).as_int() , factor_end_dt = self.end_date)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)

        self.set_critical_dates()
        self.parse_prenorm_method()
        self.reset_dataloaders()

        if self.empty_x:
            Logger.alert2(f'DataModule got empty x , fit and test stage will be skipped')
            Logger.note(f'{self.input_type} input keys: {self.input_keys}')
            self.config.stage_queue.remove('fit')
            self.config.stage_queue.remove('test')
        return self

    def set_critical_dates(self):
        '''set critical dates for model date list and test full dates'''
        dates = self.datas.date_within(self.beg_date , self.end_date)
        self.data_dates = dates

        if self.config.module_type in ['factor' , 'db']:
            # previos month end (use calendar date)
            self.test_full_dates = dates
            self.model_date_list = CALENDAR.td_array(CALENDAR.cd_array(np.unique(dates // 100) * 100 + 1 , -1))
        else:
            self.test_full_dates = dates[1:]
            if self.use_data == 'predict':
                self.model_date_list = dates[:1]
            else:
                self.model_date_list = dates[::self.config.model_interval]

    @property
    def beg_date(self):
        return -1 if self.use_data == 'predict' else self.config.beg_date

    @property
    def end_date(self):
        return 99991231 if self.use_data == 'predict' else self.config.end_date
        
    def setup(self, stage : Literal['fit' , 'test' , 'predict' , 'extract'] , 
              param : dict[str,Any] = {'seqlens' : {}} , 
              model_date = -1 , **kwargs) -> None:
        if self.setup_new_param(stage , param , model_date , **kwargs):
            self.setup_input_prepare()
            self.setup_loader_prepare()
            self.setup_loader_create()

    def setup_new_param(
            self , stage : Literal['fit' , 'test' , 'predict' , 'extract'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date = -1 , extract_backward_days = 300 , extract_forward_days = 160
        ) -> bool:
        stage = 'predict' if self.use_data == 'predict' else stage
        if self.input_type == 'db':
            slens = {self.config.model_module: 1}
        else:
            slens = self.config.seq_lens() | param.get('seqlens',{})
            slens = {key:int(val) for key,val in slens.items() if key in self.input_keys}
            slens.update({key:int(val) for key,val in param.items() if key.endswith('_seq_len')})
            
        loader_param = self.LoaderParam(stage , model_date , slens , extract_backward_days , extract_forward_days)
        if self.loader_param == loader_param: 
            return False
        self.loader_param = loader_param
        return True

    def setup_input_prepare(self):
        '''additional input prepare for hidden / db input'''
        self.setup_input_prepare_hidden()
        self.setup_input_prepare_db()

    def setup_input_prepare_hidden(self):
        '''additional input prepare for hidden input , load hidden data if needed'''
        if self.input_type not in ['hidden' , 'combo'] or not self.input_keys_hidden: 
            return
        hidden_max_date : int | Any = None
        hidden_input : dict[str,tuple[int,pd.DataFrame]] = {}
        for hidden_key in self.input_keys_hidden:
            hidden_path = HiddenPath.from_key(hidden_key)
            if hidden_key in hidden_input and hidden_input[hidden_key][0] == hidden_path.latest_hidden_model_date(self.model_date):
                df = hidden_input[hidden_key][1]
            else:
                hidden_model_date , df = hidden_path.get_hidden_df(self.model_date , exact=False)
                hidden_input[hidden_key] = (hidden_model_date , df)
            hidden_max_date = df['date'].max() if hidden_max_date is None else min(hidden_max_date , df['date'].max())

            df = df.drop(columns='dataset' , errors='ignore').set_index(['secid','date'])
            df.columns = [f'{hidden_key}.{col}' for col in df.columns]
            self.datas.x[hidden_key] = DataBlock.from_dataframe(df).align_secid_date(self.datas.secid , self.datas.date)

        assert self.datas.date[self.datas.date < self.next_model_date(self.model_date)][-1] <= hidden_max_date , \
            (self.next_model_date(self.model_date) , hidden_max_date)
        self.config.update_data_param(self.datas.x)

    def setup_input_prepare_db(self):
        '''additional input prepare for hidden input , load hidden data if needed'''
        if self.input_type != 'db': 
            return
        assert self.stage in ['test' , 'predict'] , self.stage
        test_dates = self.test_full_dates
        next_model_date = self.next_model_date(self.model_date)
        test_dates = test_dates[(test_dates > self.model_date) * (test_dates <= next_model_date)]
            
        db_mapping = self.config.db_mapping
        assert db_mapping.name == self.config.model_module , (db_mapping.name , self.config.model_module)
        self.datas.x[db_mapping.name] = db_mapping.load_block(test_dates[0] , test_dates[-1]).align_secid_date(self.datas.secid , self.datas.date)
        self.config.update_data_param(self.datas.x)

    def setup_loader_prepare(self):
        seq_lens = self.seq_lens
        x_keys = self.input_keys
        y_keys = [k for k in seq_lens.keys() if k not in x_keys]
        assert all([seq_lens[xkey] > 0 for xkey in x_keys]) , (seq_lens , x_keys)
        assert all([seq_lens[ykey] > 0 for ykey in y_keys]) , (seq_lens , y_keys)
        y_extend = max([seq_lens[ykey] for ykey in y_keys]) if y_keys else 1
        x_extend = max([seq_lens[xkey] * self.seq_steps[xkey] for xkey in x_keys]) if x_keys else 1
        d_extend = x_extend + y_extend - 1

        if self.stage == 'fit':
            model_date_col = (self.datas.date < self.model_date).sum()
            self.d0 = max(0 , model_date_col - self.config.train_skip_horizon - self.config.model_train_window - d_extend)
            self.d1 = max(0 , model_date_col - self.config.train_skip_horizon)
        elif self.stage in ['predict' , 'test']:
            next_model_date = self.next_model_date(self.model_date)

            before_test_dates = self.datas.date[self.datas.date < min(self.test_full_dates)][-y_extend:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::self.data_step]
            self.early_test_dates = test_dates[test_dates <= self.model_date][-(y_extend-1) // self.data_step:] if y_extend > 1 else test_dates[-1:-1]
            self.model_test_dates = test_dates[(test_dates > self.model_date) * (test_dates <= next_model_date)]
            
            test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])
            
            if test_dates.size == 0:
                self.d0 = len(self.datas.date) - x_extend
                self.d1 = len(self.datas.date) - 1
            else:
                self.d0 = max(np.where(self.datas.date == test_dates[0])[0][0] - x_extend + 1 , 0)
                self.d1 = np.where(self.datas.date == test_dates[-1])[0][0] + 1
            test_dates = self.datas.date[self.d0 + x_extend - 1:self.d1]
        elif self.stage == 'extract':
            model_date_col = (self.datas.date < self.model_date).sum()
            self.d0 = max(0 , model_date_col - self.loader_param.extract_backward_days - d_extend)
            self.d1 = min(max(0 , model_date_col + self.loader_param.extract_forward_days) , len(self.datas.date))
        else:
            raise KeyError(self.stage)

        self.step_len = (self.day_len - x_extend + 1) // self.data_step
        if self.step_len <= 0:
            Logger.alert2(f'Step length is less than 0 , stage: {self.stage} , d0: {self.d0} , d1: {self.d1} , data_len: {len(self.datas.date)} , x_extend: {x_extend} , data_step: {self.data_step}')
            if self.stage in ['predict' , 'test']:
                Logger.alert2(f'Test dates: {test_dates}')
            raise ValueError(f'Step length is less than 0')
        self.step_idx = torch.flip(self.day_len - 1 - torch.arange(self.step_len) * self.data_step , [0])
        self.date_idx = self.d0 + self.step_idx

        if self.stage in ['predict' , 'test']:
            assert self.step_len == len(test_dates) , (self.step_len , len(test_dates))

    @property
    def input_type(self) -> Literal['db' , 'data' , 'hidden' , 'factor' , 'combo']: 
        return self.config.model_input_type
    
    @property
    def input_keys(self) -> list[str]:
        keys = self.input_keys_data + self.input_keys_factor + self.input_keys_hidden
        if self.config.module_type == 'factor':
            keys.append('factor')
        assert len(keys) > 0 or self.config.model_input_type in ['db'] , self.config.model_input_type
        return keys

    @property
    def input_keys_data(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.model_data_types]

    @property
    def input_keys_factor(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.model_factor_types]

    @property
    def input_keys_hidden(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.model_hidden_types]

    @property
    def seq_steps(self) -> dict[str,int]:
        return self.config.seq_steps()

    @property
    def y_secid(self) -> np.ndarray:
        return self.datas.y.secid

    @property
    def y_date(self) -> np.ndarray:
        return self.datas.y.date[self.d0:self.d1]

    def setup_loader_create(self) -> None:
        if self.day_len == 0:
            self.empty_dataloader()
            return

        x_full = {k:torch.Tensor(v.values[:,self.d0:self.d1]) for k,v in self.datas.x.items()}
        y_full = torch.Tensor(self.datas.y.values[:,self.d0:self.d1]).squeeze(2)[...,:self.labels_n]

        # standardized y with step == 1
        self.y_std = self.standardize_y(y_full , None , None , no_weight = True)[0]

        valid_x = x_full if self.config.module_type == 'nn' else {}
        valid_y = self.y_std if self.stage == 'fit' else None
        
        valid_sampled = self.multiple_valid(valid_x , valid_y , self.step_idx , x_all_valid=(self.config.module_type == 'nn'))
        y_sampled , w_sampled = self.standardize_y(self.y_std , valid_sampled , self.step_idx)
            
        # since in fit stage , step_idx can be larger than 1 , different valid and result may occur
        self.y_std[:,self.step_idx] = y_sampled[:]
        self.static_dataloader(x_full , y_sampled , w_sampled , valid_sampled)

        if self.config.gc_collect_each_model:
            gc.collect() 
            torch.cuda.empty_cache()

    def multiple_valid(self , x : dict[str,torch.Tensor] , y : torch.Tensor | None , index1 : torch.Tensor , 
                       x_all_valid = True) -> torch.Tensor | None:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window (seqlen * step) non-nan , end non-zero if in k is divlast
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        valid = None if y is None else self.valid_sample(y,1,index1)
        for k , v in x.items(): 
            new_val = self.valid_sample(v , self.seq_lens[k] * self.seq_steps[k] , index1 , k in DataBlockNorm.DIVLAST , x_all_valid)
            if x_all_valid:
                valid = new_val * (True if valid is None else valid)
            else:
                valid = new_val + (False if valid is None else valid)
        return valid
    
    @staticmethod
    def valid_sample(data : torch.Tensor , rolling_window : int | None = 1 , index1 : torch.Tensor | None = None , 
                     endpoint_nonzero = False , x_all_valid = True):
        '''return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims'''
        if rolling_window is None: 
            rolling_window = 1
        if index1 is None: 
            index1 = torch.arange(data.shape[1])
        assert rolling_window > 0 , rolling_window
        data = torch.cat([torch.zeros_like(data[:,:rolling_window]) , data],dim=1).unsqueeze(2)
        sum_dim = tuple(range(2,data.ndim))
        
        if x_all_valid:
            nans = data[:,index1 + rolling_window].isnan().sum(sum_dim)
            for i in range(1 , rolling_window): 
                nans += data[:,index1 - i + rolling_window].isnan().sum(sum_dim)
            valid = nans == 0
        else:
            finite = data[:,index1 + rolling_window].isfinite().sum(sum_dim)
            for i in range(1 , rolling_window): 
                finite += data[:,index1 - i + rolling_window].isfinite().sum(sum_dim)
            valid = finite > 0

        if endpoint_nonzero: 
            valid *= ((data[:,index1 + rolling_window] == 0).sum(sum_dim) == 0)
        
        return valid
     
    def standardize_y(self , y : torch.Tensor , valid : torch.Tensor | None , index1 : torch.Tensor | None , no_weight = False) -> tuple[torch.Tensor , torch.Tensor | None]:
        '''standardize y and weight'''
        y = y[:,index1].clone() if index1 is not None else y.clone()
        if valid is not None: 
            y.nan_to_num_(0)[~valid] = torch.nan
        return tensor_standardize_and_weight(y , 0 , self.config.weight_scheme(self.loader_param.stage , no_weight))

    def train_dataloader(self)   -> BatchDataLoader: 
        return BatchDataLoader(self.loader_dict['train'] , self , desc = 'Train')
    def val_dataloader(self)     -> BatchDataLoader: 
        return BatchDataLoader(self.loader_dict['valid'] , self , desc = 'Valid')
    def test_dataloader(self)    -> BatchDataLoader: 
        return BatchDataLoader(self.loader_dict['test'] , self , desc = 'Test')
    def predict_dataloader(self) -> BatchDataLoader: 
        return BatchDataLoader(self.loader_dict['predict'] , self , tqdm = False , desc = 'Predict')
    def extract_dataloader(self) -> BatchDataLoader: 
        return BatchDataLoader(self.loader_dict['extract'] , self , desc = 'Extract')
    
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        if self.config.module_type == 'nn':
            batch = batch.to(self.device if device is None else device)
        return batch
    
    def empty_dataloader(self) -> None:
        if self.stage == 'fit':
            self.loader_dict['train'] = StoredFileLoader(self.storage , [] , 'static')
            self.loader_dict['valid'] = StoredFileLoader(self.storage , [] , 'static')
        else:
            self.loader_dict[self.stage] = StoredFileLoader(self.storage , [] , 'static')
       
    def static_dataloader(self , x : dict[str,torch.Tensor] , y : torch.Tensor , w : torch.Tensor | None , valid : torch.Tensor | None) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''   
        if valid is None: 
            valid = torch.ones(y.shape[:2] , dtype=torch.bool , device=y.device)
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(valid , index0 , index1 , self.config.train_sample_method , 
                                         self.config.train_train_ratio , self.config.train_batch_size)
        self.storage.del_group(self.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test' , 'predict' , 'extract'] , set_key
            shuf_opt = self.config.train_shuffle_option if set_key == 'train' else 'static'
            batch_files = [PATH.batch.joinpath(f'{set_key}.{bnum}.pt') for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all() , f'all b_i[:,1] must be in index1'
                index0 , xindex1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                b_x = self.batch_data_x(x , index0 , xindex1)
                b_y = self.batch_data_y(y , index0 , yindex1)
                b_w = self.batch_data_y(w , index0 , yindex1)
                b_v = self.batch_data_y(valid , index0 , yindex1)

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.stage)
            self.loader_dict[set_key] = StoredFileLoader(self.storage , batch_files , shuf_opt)

    def batch_data_x(self , x : dict[str,torch.Tensor] , index0 : torch.Tensor | np.ndarray , index1 : torch.Tensor | np.ndarray) -> list[torch.Tensor]:
        datas = []
        for model_data_type , data in x.items():
            slen = self.seq_lens[model_data_type]
            step = self.seq_steps[model_data_type]
            data = rolling_rotation(data , index0 , index1 , slen , step)
            data = self.prenorm(data , model_data_type)
            datas.append(data)
        return datas

    def batch_data_y(self , y : torch.Tensor | None, index0 : torch.Tensor | np.ndarray , index1 : torch.Tensor | np.ndarray) -> torch.Tensor | Any:
        if y is None:
            return None
        return y[index0 , index1]
        
    def parse_prenorm_method(self):
        self.prenorm_divlast  : list[str] = []
        self.prenorm_histnorm : list[str] = []
        for mdt in self.input_keys: 
            method : dict = self.config.model_data_prenorm.get(mdt , {})
            divlast = method.get('divlast'  , False) and (mdt in DataBlockNorm.DIVLAST)
            histnorm = method.get('histnorm' , True)  and (mdt in DataBlockNorm.HISTNORM)
            if (divlast or histnorm): 
                Logger.success(f'Pre-Norm [{mdt}] : {dict(divlast=divlast, histnorm=histnorm)}' , vb_level = 3)
            if divlast: 
                self.prenorm_divlast.append(mdt)
            if histnorm: 
                self.prenorm_histnorm.append(mdt)

    def prenorm(self , x : torch.Tensor, key : str) -> torch.Tensor:
        '''
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        '''
        if key in self.prenorm_divlast and x.shape[-2] > 1:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if key in self.prenorm_histnorm:
            x = x - self.datas.norms[key].avg[-x.shape[-2]:]
            x = x / (self.datas.norms[key].std[-x.shape[-2]:] + 1e-6)
        return x

    def split_sample(self , valid : torch.Tensor , index0 : torch.Tensor , index1 : torch.Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list[torch.Tensor]]:
        '''
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        '''
        l0 , l1 = valid.shape[:2]
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)
        
        def shuffle_sampling(i , bs = batch_size):
            return [i[p] for p in BatchSampler(permutation(np.arange(len(i))) , bs , drop_last=False)]
        def sequential_sampling(beg , end , posit = pos , valid = valid):
            return [posit[:,j][valid[:,j]] for j in range(beg , end)]
        
        sample_index = {}
        if self.stage == 'fit':
            sep = int(l1 * train_ratio)
            if sample_method == 'total_shuffle':
                pool = torch.tensor(permutation(np.arange(valid.sum().item())))
                sep = int(len(pool) * train_ratio)
                sample_index['train'] = shuffle_sampling(pos[valid][pool[:sep]])
                sample_index['valid'] = shuffle_sampling(pos[valid][pool[sep:]])
            elif sample_method == 'both_shuffle':
                sample_index['train'] = shuffle_sampling(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = shuffle_sampling(pos[:,sep:][valid[:,sep:]])
            elif sample_method == 'train_shuffle':
                sample_index['train'] = shuffle_sampling(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = sequential_sampling(sep , l1)
            else:
                sample_index['train'] = sequential_sampling(0 , sep)
                sample_index['valid'] = sequential_sampling(sep , l1)
        else:
            sample_index[self.stage] = sequential_sampling(0 , l1)
        return sample_index    

    @classmethod
    def get_date_batch_data(cls , config : TrainConfig , date : int , model_num : int = 0) -> BatchData:
        if config not in cls._config_instance_for_batch_data:
            cls._config_instance_for_batch_data[config] = cls(config , 'both').load_data()
        module = cls._config_instance_for_batch_data[config]
        model_param = config.model_param[model_num]
        assert date in module.test_full_dates , f"date {date} not in test_full_dates [{module.test_full_dates.min()}-{module.test_full_dates.max()}]"
        module.setup('predict' , model_param , module.model_date_list[module.model_date_list < date][-1])
        dataloader = module.predict_dataloader()
        return dataloader.of_date(date)
