import gc , torch
import numpy as np
import pandas as pd

from numpy.random import permutation
from torch import Tensor
from torch.utils.data import BatchSampler
from typing import Any , Literal , Optional

from src.basic import CONF , PATH , SILENT , HiddenPath
from src.data import DataBlockNorm , DataProcessor , ModuleData , DataBlock
from src.func import tensor_standardize_and_weight , match_values , index_intersect
from src.model.util import BaseBuffer , BaseDataModule , BatchData , TrainConfig , MemFileStorage , StoredFileLoader

from .loader import BatchDataLoader


class DataModule(BaseDataModule):    
    def __init__(self , config : Optional[TrainConfig] = None , use_data : Literal['fit','predict','both'] = 'fit'):
        '''
        1. load Package of BlockDatas of x , y , norms and index
        2. Setup model_date dataloaders
        3. Buffer dict for dynamic nn's
        '''
        self.config   = TrainConfig.load() if config is None else config
        self.use_data = use_data
        self.storage  = MemFileStorage(self.config.mem_storage)
        self.buffer   = BaseBuffer(self.device)

    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.config.model_name},use_data={self.use_data},datas={self.data_type_list})'

    def load_data(self):
        self.datas = ModuleData.load(self.data_type_list , self.config.model_labels, 
                                     fit = self.use_data != 'predict' , predict = self.use_data != 'fit' ,
                                     dtype = self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)

        if self.use_data == 'predict':
            self.model_date_list = self.datas.date[:1]
            self.test_full_dates = self.datas.date[1:]
        else:
            self.model_date_list = self.datas.date_within(self.config.beg_date , self.config.end_date , self.config.model_interval)
            self.test_full_dates = self.datas.date_within(self.config.beg_date , self.config.end_date)[1:]

        self.parse_prenorm_method()
        self.reset_dataloaders()
        return self

    @property
    def data_type_list(self) -> list[str]:
        '''get data type list (abbreviation)'''
        if self.config.model_input_type == 'data':
            data_type_list = [ModuleData.abbr(data_type) for data_type in self.config.model_data_types]
        else:
            data_type_list = []
        return data_type_list
    
    @property
    def hidden_type_list(self) -> list[str]:
        '''get data type list (abbreviation)'''
        if self.config.model_input_type == 'data':
            hidden_type_list = []
        else:
            hidden_type_list = self.config.model_hidden_types
        return hidden_type_list
    
    @staticmethod
    def prepare_data(data_types : Optional[list[str]] = None):
        DataProcessor.main(predict = False , data_types = data_types)
        DataProcessor.main(predict = True , data_types = data_types)
        
    def setup(self, stage : Literal['fit' , 'test' , 'predict' , 'extract'] , 
              param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1 , extract_backward_days = 300 , extract_forward_days = 160) -> None:
        loader_param = self.setup_param_parsing(stage , param , model_date , extract_backward_days , extract_forward_days)
        if self.loader_param == loader_param: return
        self.loader_param = loader_param

        if self.config.model_input_type == 'data':
            self.setup_data_prepare()
        else:
            self.setup_hidden_prepare()

        self.setup_loader_create()

    def setup_param_parsing(
            self , stage : Literal['fit' , 'test' , 'predict' , 'extract'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date = -1 , extract_backward_days = 300 , extract_forward_days = 160):
        if self.use_data == 'predict': stage = 'predict'

        if self.config.model_input_type == 'data':
            seqlens : dict = {key:param['seqlens'][key] for key in self.data_type_list}
            seqlens.update({k:v for k,v in param.items() if k.endswith('_seq_len')})
        else:
            seqlens : dict = {key:1 for key in self.hidden_type_list}

        loader_param = self.LoaderParam(stage , model_date , seqlens , extract_backward_days , extract_forward_days)
        return loader_param
    
    def setup_hidden_prepare(self) -> None:
        model_date = self.loader_param.model_date
        
        self.seqs = {key:1 for key in self.hidden_type_list}
        self.seq0 = self.seqx = self.seqy = 1

        hidden_max_date : int | Any = None
        self.hidden_input : dict[str,tuple[int,pd.DataFrame]] = {}
        for hidden_key in self.hidden_type_list:
            hidden_path = HiddenPath.from_key(hidden_key)
            if hidden_key in self.hidden_input and self.hidden_input[hidden_key][0] == hidden_path.latest_hidden_model_date(model_date):
                df = self.hidden_input[hidden_key][1]
            else:
                hidden_model_date , df = hidden_path.get_hidden_df(model_date , exact=False)
                self.hidden_input[hidden_key] = (hidden_model_date , df)
            hidden_max_date = df['date'].max() if hidden_max_date is None else min(hidden_max_date , df['date'].max())

            df = df.drop(columns='dataset' , errors='ignore').set_index(['secid','date'])
            df.columns = [f'{hidden_key}.{col}' for col in df.columns]
            self.datas.x[hidden_key] = DataBlock.from_dataframe(df).align_secid_date(self.datas.secid , self.datas.date)

        assert self.datas.date[self.datas.date < self.next_model_date(model_date)][-1] <= hidden_max_date , \
            (self.next_model_date(model_date) , hidden_max_date)

    def setup_data_prepare(self) -> None:
        seqlens = self.loader_param.seqlens

        x_keys = self.data_type_list
        y_keys = [k for k in seqlens.keys() if k not in x_keys]
        self.seqs = {k:seqlens.get(k , 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([v for k,v in self.seqs.items() if k in y_keys]) if y_keys else 1
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys]) if x_keys else 1
        self.seq0 = self.seqx + self.seqy - 1

    def setup_loader_create(self):
        stage , model_date = self.loader_param.stage , self.loader_param.model_date
        if stage == 'fit':
            model_date_col = (self.datas.date < model_date).sum()
            data_step = self.config.train_data_step
            d0 = max(0 , model_date_col - self.config.train_skip_horizon - self.config.model_train_window - self.seq0)
            d1 = max(0 , model_date_col - self.config.train_skip_horizon)
        elif stage in ['predict' , 'test']:
            if stage == 'predict': self.model_date_list = np.array([model_date])
            next_model_date = self.next_model_date(model_date)
            data_step  = 1

            before_test_dates = self.datas.date[self.datas.date < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::data_step]
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // data_step:] if self.seqy > 1 else test_dates[-1:-1]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            
            test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])
            
            d0 = max(np.where(self.datas.date == test_dates[0])[0][0] - self.seqx + 1 , 0)
            d1 = np.where(self.datas.date == test_dates[-1])[0][0] + 1
            test_dates = self.datas.date[d0 + self.seqx - 1:d1]

        elif stage == 'extract':
            model_date_col = (self.datas.date < model_date).sum()
            data_step = 1
            d0 = max(0 , model_date_col - self.loader_param.extract_backward_days - self.seq0)
            d1 = min(max(0 , model_date_col + self.loader_param.extract_forward_days) , len(self.datas.date))
        else:
            raise KeyError(stage)

        self.day_len  = d1 - d0
        self.step_len = (self.day_len - self.seqx + 1) // data_step
        if stage in ['predict' , 'test']: assert self.step_len == len(test_dates) , (self.step_len , len(test_dates))
        self.step_idx = torch.flip(self.day_len - 1 - torch.arange(self.step_len) * data_step , [0])
        self.date_idx = d0 + self.step_idx
        self.y_secid , self.y_date = self.datas.y.secid , self.datas.y.date[d0:d1]

        x = {k:Tensor(v.values)[:,d0:d1] for k,v in self.datas.x.items()}
        y = Tensor(self.datas.y.values)[:,d0:d1].squeeze(2)[...,:self.labels_n]

        # standardized y with step == 1
        self.y = self.standardize_y(y , None , None , no_weight = True)[0]

        valid_x = x if self.config.module_type == 'nn' else {}
        valid_y = self.y if stage == 'fit' else None
        
        valid = self.multiple_valid(valid_x , valid_y , self.step_idx , self.seqs , x_all_valid=(self.config.module_type == 'nn'))
        y , w = self.standardize_y(self.y , valid , self.step_idx)
            
        # since in fit stage , step_idx can be larger than 1 , different valid and result may occur
        self.y[:,self.step_idx] = y[:]
        self.static_dataloader(x , y , w , valid)

        gc.collect() 
        torch.cuda.empty_cache()

    def multiple_valid(self , x : dict[str,Tensor] , y : Tensor | None , index1 : Tensor , 
                       rolling_windows : dict[str,int] = {} , x_all_valid = True) -> Tensor | None:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is divlast
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        valid = None if y is None else self.valid_sample(y,1,index1)
        for k , v in x.items(): 
            new_val = self.valid_sample(v , rolling_windows.get(k,1) , index1 , k in DataBlockNorm.DIVLAST , x_all_valid)
            if x_all_valid:
                valid = new_val * (True if valid is None else valid)
            else:
                valid = new_val + (False if valid is None else valid)
        return valid
    
    @staticmethod
    def valid_sample(data : Tensor , rolling_window : int | None = 1 , index1 : Tensor | None = None , 
                     endpoint_nonzero = False , x_all_valid = True):
        '''return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims'''
        if rolling_window is None: rolling_window = 1
        if index1 is None: index1 = torch.arange(data.shape[1])
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
     
    def standardize_y(self , y : Tensor , valid : Optional[Tensor] , index1 : Optional[Tensor] , no_weight = False) -> tuple[Tensor , Optional[Tensor]]:
        '''standardize y and weight'''
        y = y[:,index1].clone() if index1 is not None else y.clone()
        if valid is not None: y.nan_to_num_(0)[~valid] = torch.nan
        return tensor_standardize_and_weight(y , 0 , self.config.weight_scheme(self.loader_param.stage , no_weight))

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
        
    def parse_prenorm_method(self):
        prenorm_keys = self.data_type_list if self.config.model_input_type == 'data' else self.hidden_type_list
        self.prenorm_divlast  : dict[str,bool] = {}
        self.prenorm_histnorm : dict[str,bool] = {}
        for mdt in prenorm_keys: 
            method : dict = self.config.model_data_prenorm.get(mdt , {})
            new_method = {
                'divlast' : method.get('divlast'  , False) and (mdt in DataBlockNorm.DIVLAST) ,
                'histnorm': method.get('histnorm' , True)  and (mdt in DataBlockNorm.HISTNORM) ,
            }
            if not SILENT: print(f'Pre-Norming method of [{mdt}] : {new_method}')
            self.prenorm_divlast[mdt]  = new_method['divlast']
            self.prenorm_histnorm[mdt] = new_method['histnorm']

    def prenorm(self , x : Tensor, key : str) -> Tensor:
        '''
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        '''
        if self.prenorm_divlast[key] and x.shape[-2] > 1:
            x = x / (x.select(-2,-1).unsqueeze(-2) + 1e-6)
        if self.prenorm_histnorm[key]:
            x = x - self.datas.norms[key].avg[-x.shape[-2]:]
            x = x / (self.datas.norms[key].std[-x.shape[-2]:] + 1e-6)
        return x

    def train_dataloader(self)   -> BatchDataLoader: return BatchDataLoader(self.loader_dict['train'] , self)
    def val_dataloader(self)     -> BatchDataLoader: return BatchDataLoader(self.loader_dict['valid'] , self)
    def test_dataloader(self)    -> BatchDataLoader: return BatchDataLoader(self.loader_dict['test'] , self)
    def predict_dataloader(self) -> BatchDataLoader: return BatchDataLoader(self.loader_dict['predict'] , self)
    def extract_dataloader(self) -> BatchDataLoader: return BatchDataLoader(self.loader_dict['extract'] , self)
    
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        if self.config.module_type == 'nn':
            batch = batch.to(self.device if device is None else device)
        return batch
        
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Optional[Tensor]) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        if valid is None: valid = torch.ones(y.shape[:2] , dtype=torch.bool , device=y.device)
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(valid , index0 , index1 , self.config.train_sample_method , 
                                         self.config.train_train_ratio , self.config.train_batch_size)
        self.storage.del_group(self.loader_param.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test' , 'predict' , 'extract'] , set_key
            shuf_opt = self.config.train_shuffle_option if set_key == 'train' else 'static'
            batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all()
                i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                b_x = [self.prenorm(self.rolling_rotation(x[mdt],self.seqs[mdt],i0,i1) , mdt) for mdt in x.keys()]
                b_y = y[i0 , yindex1]
                b_w = None if w is None else w[i0 , yindex1]
                b_v = valid[i0 , yindex1]

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.loader_param.stage)
            self.loader_dict[set_key] = StoredFileLoader(self.storage , batch_files , shuf_opt)

    def split_sample(self , valid : Tensor , index0 : Tensor , index1 : Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list]:
        '''
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        '''
        stage = self.loader_param.stage
        l0 , l1 = valid.shape[:2]
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)
        
        def shuffle_sampling(i , bs = batch_size):
            return [i[p] for p in BatchSampler(permutation(np.arange(len(i))) , bs , drop_last=False)]
        def sequential_sampling(beg , end , posit = pos , valid = valid):
            return [posit[:,j][valid[:,j]] for j in range(beg , end)]
        
        sample_index = {}
        if stage == 'fit':
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
            sample_index[stage] = sequential_sampling(0 , l1)
        return sample_index    
