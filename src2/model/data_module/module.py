import gc , torch
import numpy as np
import pandas as pd

from numpy.random import permutation
from torch import Tensor
from torch.utils.data import BatchSampler
from typing import Any , Literal , Optional

from .loader import BatchDataLoader
from ..util import BatchData , TrainConfig , MemFileStorage , StoredFileLoader
from ..util.classes import BaseBuffer , BaseDataModule
from ...basic import CONF , PATH
from ...data import DataBlockNorm , DataProcessor , ModuleData , DataBlock
from ...func import tensor_standardize_and_weight , match_values , index_intersect

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

    def load_data(self):
        self.datas = ModuleData.load(self.data_type_list , self.config.model_labels, 
                                     fit = self.use_data != 'predict' , predict = self.use_data != 'fit' ,
                                     dtype = self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)

        if self.use_data == 'predict':
            self.model_date_list = self.datas.date[0]
            self.test_full_dates = self.datas.date[1:]
        else:
            self.model_date_list = self.datas.date_within(self.config.beg_date , self.config.end_date , self.config.model_interval)
            self.test_full_dates = self.datas.date_within(self.config.beg_date , self.config.end_date)[1:]

        self.static_prenorm_method = {}
        for mdt in self.data_type_list: 
            method : dict[str,bool] = self.config.model_data_prenorm.get(mdt , {})
            method['divlast']  = method.get('divlast' , True) and (mdt in DataBlockNorm.DIVLAST)
            method['histnorm'] = method.get('histnorm', True) and (mdt in DataBlockNorm.HISTNORM)
            if not CONF.SILENT: print(f'Pre-Norming method of [{mdt}] : {method}')
            self.static_prenorm_method[mdt] = method

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
    
    @staticmethod
    def prepare_data(data_types : Optional[list[str]] = None):
        DataProcessor.main(predict = False , data_types = data_types)
        DataProcessor.main(predict = True , data_types = data_types)
        
    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , 
              param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1) -> None:
        if self.config.model_input_type == 'data':
            self.setup_with_data(stage , param , model_date)
        else:
            self.setup_with_hidden(stage , param , model_date)

    def setup_parse_param(
            self , stage : Literal['fit' , 'test' , 'predict'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date = -1):
        if self.use_data == 'predict': stage = 'predict'
        self.stage = stage

        if self.config.model_input_type == 'data':
            seqlens : dict = {key:param['seqlens'][key] for key in self.data_type_list}
            seqlens.update({k:v for k,v in param.items() if k.endswith('_seq_len')})
        else:
            seqlens : dict = {'hidden':1}

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        return stage , model_date , seqlens
    
    def setup_process_dataloader(self , x , y):
        # standardized y with step == 1
        self.y = self.standardize_y(y , None , None , no_weight = True)[0]
        valid = self.full_valid_sample(x , self.y , self.step_idx) if self.stage == 'fit' else None
        # since in fit stage , step_idx can be larger than 1 , different valid and result may occur
        y , w = self.standardize_y(self.y , valid , self.step_idx)
        self.y[:,self.step_idx] = y[:]
        self.static_dataloader(x , y , w , valid)

        gc.collect() 
        torch.cuda.empty_cache()

    def setup_with_hidden(
            self , stage : Literal['fit' , 'test' , 'predict'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date = -1) -> None:
        
        stage , model_date , seqlens = self.setup_parse_param(stage , param , model_date)
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        self.seqs = {'hidden':1}
        self.seq0 = self.seqx = self.seqy = 1

        hidden_dates : list[np.ndarray] = []
        hidden_df : pd.DataFrame | Any = None
        ds_list = ['train' , 'valid'] if stage == 'fit' else ['test' , 'predict']
        for hidden_key in self.config.model_hidden_types:
            model_name , model_num , model_type = hidden_key.split('.')
            hidden_path = PATH.hidden.joinpath(model_name , f'hidden.{model_num}.{model_type}.{model_date}.feather')
            df = pd.read_feather(hidden_path)
            df = df[df['dataset'].isin(ds_list)].drop(columns='dataset').set_index(['secid','date'])
            hidden_dates.append(df.index.get_level_values('date').unique().to_numpy())
            df.columns = [f'{hidden_key}.{col}' for col in df.columns]
            hidden_df = df if hidden_df is None else hidden_df.join(df , how='outer')

        stage_date = index_intersect(hidden_dates)[0]
        if self.stage != 'fit':
            stage_date = index_intersect([stage_date , self.test_full_dates])[0]
        self.day_len = len(stage_date)
        self.step_len = len(stage_date)
        self.date_idx , self.step_idx = torch.arange(self.day_len) , torch.arange(self.day_len)

        y_aligned = self.datas.y.align_date(stage_date , inplace=False)
        self.y_secid , self.y_date = y_aligned.secid , y_aligned.date

        if stage in ['predict' , 'test']:
            self.model_test_dates = stage_date
            self.early_test_dates = stage_date[:0]

        x = {'hidden':DataBlock.from_dataframe(hidden_df).align_secid_date(self.y_secid , self.y_date).as_tensor().values}
        y = Tensor(y_aligned.values).squeeze(2)[...,:self.labels_n]

        self.setup_process_dataloader(x , y)

    def setup_with_data(
            self , stage : Literal['fit' , 'test' , 'predict'] , 
            param : dict[str,Any] = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
            model_date = -1) -> None:
        
        stage , model_date , seqlens = self.setup_parse_param(stage , param , model_date)
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        x_keys = self.data_type_list
        y_keys = [k for k in seqlens.keys() if k not in x_keys]
        self.seqs = {k:seqlens.get(k , 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([v for k,v in self.seqs.items() if k in y_keys]) if y_keys else 1
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys]) if x_keys else 1
        self.seq0 = self.seqx + self.seqy - 1

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

        self.setup_process_dataloader(x , y)

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

    def train_dataloader(self)   -> BatchDataLoader: return BatchDataLoader(self.loader_dict['train'] , self)
    def val_dataloader(self)     -> BatchDataLoader: return BatchDataLoader(self.loader_dict['valid'] , self)
    def test_dataloader(self)    -> BatchDataLoader: return BatchDataLoader(self.loader_dict['test'] , self)
    def predict_dataloader(self) -> BatchDataLoader: return BatchDataLoader(self.loader_dict['predict'] , self)
    
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        if self.config.module_type == 'nn':
            batch = batch.to(self.device if device is None else device)
        return batch
        
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Optional[Tensor]) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        if valid is None: valid = torch.ones(y.shape[:2] , dtype=torch.bool , device=y.device)
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(self.stage , valid , index0 , index1 , self.config.train_sample_method , 
                                         self.config.train_train_ratio , self.config.train_batch_size)
        self.storage.del_group(self.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = self.config.train_shuffle_option if set_key == 'train' else 'static'
            batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all()
                i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                b_x = [self.prenorm(self.rolling_rotation(x[mdt],self.seqs[mdt],i0,i1) , mdt) for mdt in x.keys()]
                b_y = y[i0 , yindex1]
                b_w = None if w is None else w[i0 , yindex1]
                b_v = valid[i0 , yindex1]

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.stage)
            self.loader_dict[set_key] = StoredFileLoader(self.storage , batch_files , shuf_opt)

    def split_sample(self , stage , valid : Tensor , index0 : Tensor , index1 : Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list]:
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
            # test dataloader should have the same length as dates, so no filtering of val[:,j].sum() > 0
            sample_index['test'] = sequential_sampling(0 , l1)
        return sample_index    

    