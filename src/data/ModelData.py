import gc , os
import numpy as np
import torch

from dataclasses import dataclass , field
from torch import Tensor
from typing import Any , Callable , Literal , Optional

from .BlockData import DataBlock , DataBlockNorm
from ..util import Device , Storage , DataloaderStored , BatchData , TrainConfig
from ..func.basic import tensor_standardize_and_weight , match_values

from ..environ import DIR

class ModelData:
    '''
    A class to store relavant training data
    '''
    def __init__(self , config : Optional[TrainConfig] = None , if_train : bool = True):
        self.config : TrainConfig = TrainConfig.load() if config is None else config
        self.if_train : bool = if_train

        self.load_model_data()
        self.reset_dataloaders()
        self.reset_buffer()

    def load_model_data(self):
        '''load torch pack of BlockDatas of x , y , norms and index'''
        self.device  = Device()
        self.storage = Storage(self.config.mem_storage)
        self.datas = DataPack.load_pack(self.data_type_list, self.config.labels, self.if_train, self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)
        self.model_date_list = self.datas.date_within(self.config.beg_date    , self.config.end_date , self.config.interval)
        self.test_full_dates = self.datas.date_within(self.config.beg_date + 1, self.config.end_date)

        self.static_prenorm_method = {}
        for mdt in self.data_type_list: 
            method = self.config.model_data_prenorm.get(mdt , {})
            method['divlast']  = method.get('divlast' , True) and (self.abbr(mdt) in ['day'])
            method['histnorm'] = method.get('histnorm', True) and (self.datas.norms.get(mdt) is not None)
            print(f'Pre-Norming method of [{mdt}] : {method}')
            self.static_prenorm_method[mdt] = method

    def reset_dataloaders(self): 
        '''Reset dataloaders and dataloader_param'''
        self.dataloaders = {}
        self.dataloader_param = ()
        gc.collect() 
        torch.cuda.empty_cache()   

    def reset_buffer(self): 
        '''Reset buffer , buffer_init , buffer_proc'''
        self.buffer: dict = {}
        self.buffer_init = self.define_buffer_init(self.config.buffer_type , **self.config.buffer_param) 
        self.buffer_proc = self.define_buffer_proc(self.config.buffer_type , **self.config.buffer_param)  

    @property
    def data_type_list(self):
        return self.type_list_abbr(self.config.data_type_list)

    @staticmethod
    def abbr(data_type): return DataBlock.data_type_abbr(data_type)

    @classmethod
    def type_list_abbr(cls , model_data_type):
        '''get data type list (abbreviation)'''
        if isinstance(model_data_type , str): model_data_type = model_data_type.split('+')
        return [cls.abbr(tp) for tp in model_data_type]
    
    def get_dataloader_param(self , loader_type , model_date = -1 , param = {} , namespace = None):
        '''get function params of dataloader: loader_type , model_date , seqlens'''
        if namespace is not None: model_date , param = namespace.model_date , namespace.param
        seqlens = param['seqlens']
        if self.config.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        return loader_type , model_date , seqlens

    def create_dataloader(self , loader_type , model_date , seqlens):
        '''Create train/valid/test dataloaders'''
        if self.dataloader_param == (loader_type , model_date , seqlens): return NotImplemented
        self.dataloader_param = loader_type , model_date , seqlens

        assert loader_type in ['train' , 'test'] , loader_type
        assert model_date > 0 , model_date

        self.loader_type = loader_type
        y_keys , x_keys = [k for k in seqlens.keys() if k in ['hist_loss','hist_preds','hist_labels']] , self.data_type_list
        self.seqs = {k:(seqlens[k] if k in seqlens.keys() else 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([1]+[v for k,v in self.seqs.items() if k in y_keys])
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys])
        self.seq0 = self.seqx + self.seqy - 1

        if self.loader_type == 'train':
            model_date_col = (self.datas.date < model_date).sum()    
            d0 = max(0 , model_date_col - self.config.skip_horizon - self.config.input_span - self.seq0)
            d1 = max(0 , model_date_col - self.config.skip_horizon)
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx) // self.config.input_step_day
            # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
            #  (You can probably work around this by making a copy of your array  with array.copy().) 
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * self.config.input_step_day).copy() 
            self.date_idx = d0 + self.step_idx
        else:
            if self.if_train:
                if model_date == self.model_date_list[-1]:
                    next_model_date = self.config.end_date + 1
                else:
                    next_model_date = self.model_date_list[self.model_date_list > model_date][0]
            else:
                self.model_date_list = [model_date]
                next_model_date = max(self.test_full_dates) + 1
            test_step  = self.config.test_step_day
            before_test_dates = self.datas.date[self.datas.date < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::test_step]
            
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // test_step:] if self.seqy > 1 else test_dates[-1:-1]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            
            _cal_test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])

            d0 = max(np.where(self.datas.date == _cal_test_dates[0])[0][0] - self.seqx + 1 , 0)
            d1 = np.where(self.datas.date == _cal_test_dates[-1])[0][0] + 1
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx + 1) // test_step + (0 if self.day_len % test_step == 0 else 1)
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * test_step).copy() 
            self.date_idx = d0 + self.step_idx

        x = {k:v.values[:,d0:d1] for k,v in self.datas.x.items()}
        self.y , _ = self.process_y_data(self.datas.y.values[:,d0:d1].squeeze(2)[...,:self.labels_n] , None , no_weight = True)
        self.y_secid , self.y_date = self.datas.y.secid , self.datas.y.date[d0:d1]

        self.buffer.update(self.buffer_init(self))
        self.nonnan_sample = self.cal_nonnan_sample(x, self.y, loader_type = loader_type , 
                                                    **{k:v for k,v in self.buffer.items() if k in self.seqs.keys()})
        y_step , w_step = self.process_y_data(self.y , self.nonnan_sample)
        self.y[:,self.step_idx] = y_step[:]
        
        self.buffer.update(self.buffer_proc(self))
        self.buffer = self.device(self.buffer) # type: ignore

        index = self.data_sampling(self.nonnan_sample)
        self.static_dataloader(x , y_step , w_step , index , self.nonnan_sample)

        gc.collect() 
        torch.cuda.empty_cache()
    
    def cal_nonnan_sample(self , x : dict , y : Tensor, loader_type : Literal['train','test'] , **kwargs):
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is 'day'
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        idx = self.step_idx
        valid_sample = self.nonnan_sample_pos(y,idx) if loader_type == 'train' else True
        for k,v in x.items():      valid_sample *= self.nonnan_sample_pos(v,idx,self.seqs[k],self.abbr(k) in ['day'])
        for k,v in kwargs.items(): valid_sample *= self.nonnan_sample_pos(v,idx,self.seqs[k])
        return valid_sample > 0

    @staticmethod
    def nonnan_sample_pos(data : torch.Tensor , index1 : np.ndarray , rolling_window = 1 , endpoint_nonzero = False):
        '''return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims'''
        # if index1 is None: index1 = np.arange(rolling_window - 1 , data.shape[1])
        data = torch.cat([torch.zeros_like(data[:,:rolling_window]) , data],dim=1).unsqueeze(2)
        sum_dim = tuple(range(2,data.ndim))
        
        invalid_samp = data[:,index1 + rolling_window].isnan().sum(sum_dim)
        if endpoint_nonzero: invalid_samp += (data[:,index1 + rolling_window] == 0).sum(sum_dim)
        for i in range(rolling_window - 1): invalid_samp += data[:,index1 + rolling_window - i - 1].isnan().sum(sum_dim)
        return (invalid_samp == 0)
     
    def process_y_data(self , y , nonnan_sample , no_weight = False) -> tuple[Tensor , Optional[Tensor]]:
        '''standardize y and weight'''
        weight_scheme = None if no_weight else self.config.Train.weight_scheme.get(self.loader_type.lower() , 'equal')
        if nonnan_sample is None:
            y_new = y
        else:
            y_new = torch.rand(*nonnan_sample.shape , *y.shape[2:])
            y_new[:] = y[:,self.step_idx].nan_to_num(0)
            y_new[nonnan_sample == 0] = torch.nan
        y_new , w_new = tensor_standardize_and_weight(y_new , 0 , weight_scheme)
        return y_new , w_new
        
    def data_sampling(self , nonnan_sample) -> dict[str,list|Tensor]:
        '''
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        '''
        sample_method = self.config.Train.sample_method
        train_ratio   = self.config.Train.train_ratio
        assert sample_method in ['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] , sample_method

        def _shuffle_sampling(ii , batch_size = self.config.batch_size):
            pool = np.random.permutation(np.arange(len(ii)))
            return [ii[pos] for pos in torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)]

        pos = nonnan_sample
        shp = nonnan_sample.shape
        pij = torch.zeros(shp[0] , shp[1] , 2 , dtype = torch.int)
        pij[:,:,0] = torch.arange(shp[0] , dtype = torch.int).reshape(-1,1) 
        pij[:,:,1] = torch.tensor(self.step_idx)

        sample_index = {}
        if self.loader_type == 'train':
            dtrain = int(shp[1] * train_ratio)
            if sample_method == 'total_shuffle':
                pool = np.random.permutation(np.arange(pos.sum()))
                train_samples = int(len(pool) * train_ratio)
                ii_train = pij[pos][pool[:train_samples]]
                ii_valid = pij[pos][pool[train_samples:]]
                sample_index['train'] = _shuffle_sampling(ii_train)
                sample_index['valid'] = _shuffle_sampling(ii_valid)
            elif sample_method == 'both_shuffle':
                ii_train = pij[:,:dtrain][pos[:,:dtrain]]
                ii_valid = pij[:,dtrain:][pos[:,dtrain:]]
                sample_index['train'] = _shuffle_sampling(ii_train)
                sample_index['valid'] = _shuffle_sampling(ii_valid)
            elif sample_method == 'train_shuffle':
                ii_train = pij[:,:dtrain][pos[:,:dtrain]]
                sample_index['train'] = _shuffle_sampling(ii_train)
                sample_index['valid'] = [pij[:,j][pos[:,j]] for j in range(dtrain , shp[1]) if pos[:,j].sum() > 0]
            else:
                sample_index['train'] = [pij[:,j][pos[:,j]] for j in range(0      , dtrain) if pos[:,j].sum() > 0]
                sample_index['valid'] = [pij[:,j][pos[:,j]] for j in range(dtrain , shp[1]) if pos[:,j].sum() > 0]
        else:
            # test dataloader should have the same length as dates, so no filtering of pos[:,j].sum() > 0
            sample_index['test'] = [pij[:,j][pos[:,j]] for j in range(0 , shp[1])]

        return sample_index
        
    def static_dataloader(self , x , y , w , sample_index , nonnan_sample) -> None:
        '''
        update dataloaders dict(set_name = ['train' , 'valid']), 
        save batch_data to f'{DIR.model}/{model_name}/{set_name}_batch_data' and later load them
        '''
        shuffle_option = self.config.Train.shuffle_option
        self.storage.del_group(self.loader_type)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = shuffle_option if set_key == 'train' else 'static'
            pbar_opt = self.config.verbosity >= 10
            batch_files = [f'{DIR.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , batch_i in enumerate(set_samples):
                assert torch.isin(batch_i[:,1] , torch.tensor(self.step_idx)).all()
                i0 , i1 , yi1 = batch_i[:,0] , batch_i[:,1] , match_values(self.step_idx , batch_i[:,1])
                batch_x = []
                for mdt in x.keys():
                    data = self.selected_rolling_window(x[mdt] , self.seqs[mdt] , i0 , i1 , dim = 1 , squeeze_out = True)
                    batch_x.append(self.prenorm(data , mdt))

                batch_data = BatchData(x = batch_x , y = y[i0,yi1] , w = None if w is None else w[i0,yi1] ,
                                       i = batch_i , nonnan = nonnan_sample[i0,yi1])
                self.storage.save(batch_data, batch_files[bnum] , group = self.loader_type)

            self.dataloaders[set_key] = DataloaderStored(self.storage , batch_files , self.device , shuf_opt , pbar_opt)

    @staticmethod
    def selected_rolling_window(x : Tensor , rw , index0 , index1 , dim = 1 , squeeze_out = True) -> Tensor:
        assert x.ndim == 4 , x.ndim
        assert len(index0) == len(index1) , (len(index0) , len(index1))
        try:
            x_rw = x.unfold(dim , rw , 1)[index0 , index1 + 1 - rw].permute(0,3,1,2)
        except MemoryError:
            x_rw = torch.stack([x[index0 , index1+i+1-rw] for i in range(rw)],dim=dim)
        if squeeze_out: x_rw = x_rw.squeeze(-2)
        return x_rw
        
    def prenorm(self , x : Tensor, key : str) -> Tensor:
        '''
        return panel_normalized x
        1.divlast: divide by the last value, get seq-mormalized x
        2.histnorm: normalized by history avg and std
        '''
        if self.static_prenorm_method[key]['divlast']:
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-6
        if self.static_prenorm_method[key]['histnorm']:
            x -= self.datas.norms[key].avg[-x.shape[-2]:]
            x /= self.datas.norms[key].std[-x.shape[-2]:] + 1e-6
        return x

    @staticmethod
    def define_buffer_init(key , **param) -> Callable:
        # first param of wrapper is container, which represent self in ModelData
        if key == 'tra':
            def tra_wrapper(self_container , *args, **kwargs):
                buffer = dict()
                if param['tra_num_states'] > 1:
                    hist_loss_shape = list(self_container.y.shape)
                    hist_loss_shape[2] = param['tra_num_states']
                    buffer['hist_labels'] = self_container.y
                    buffer['hist_preds'] = torch.randn(hist_loss_shape)
                    buffer['hist_loss']  = (buffer['hist_preds'] - buffer['hist_labels'].nan_to_num(0)).square()
                return buffer
            return tra_wrapper
        else:
            def none_wrapper(*args, **kwargs): return {}
            return none_wrapper
        
    @staticmethod
    def define_buffer_proc(key , **param) -> Callable:
        # first param of wrapper is container, which represent self in ModelData
        if key == 'tra':
            def tra_wrapper(self_container , *args, **kwargs):
                buffer = dict()
                if param['tra_num_states'] > 1:
                    buffer['hist_loss']  = (self_container.buffer['hist_preds'] - 
                                            self_container.buffer['hist_labels'].nan_to_num(0)).square()
                return buffer
            return tra_wrapper
        else:
            def none_wrapper(*args, **kwargs): return {}
            return none_wrapper
        
@dataclass
class DataPack:
    x : dict[str,DataBlock]
    y : DataBlock
    norms : dict[str,DataBlockNorm]
    secid : np.ndarray
    date  : np.ndarray

    def __post_init__(self) -> None:
        pass

    def align_ylabels(self , y_labels = None) -> None:
        if y_labels is not None: self.y.align_feature(y_labels)

    def date_within(self , start , end , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def load_pack(cls , data_type_list , y_labels = None , if_train=True , dtype = torch.float):
        if dtype is None: dtype = torch.float
        if isinstance(dtype , str): dtype = getattr(torch , dtype)
        if if_train: 
            last_date = max(DataBlock.load_dict(DataBlock.block_path('y'))['date'])
            torch_pack_code = '+'.join(data_type_list)
            torch_pack = f'{DIR.torch_pack}/{torch_pack_code}.{last_date}.pt'
        else:
            torch_pack = 'no_torch_pack'

        if os.path.exists(torch_pack):
            print(f'use {torch_pack}')
            data = cls(**torch.load(torch_pack))
        else:
            data_type_list = ['y' , *data_type_list]
            
            blocks = DataBlock.load_keys(data_type_list, if_train , alias_search=True,dtype = dtype)
            norms  = DataBlockNorm.load_keys(data_type_list, if_train , alias_search=True,dtype = dtype)

            y : DataBlock = blocks[0]
            x : dict[str,DataBlock] = {ModelData.abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
            norms = {ModelData.abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
            secid , date = blocks[0].secid , blocks[0].date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = cls(x , y , norms , secid , date)
            if if_train: 
                os.makedirs(os.path.dirname(torch_pack) , exist_ok=True)
                torch.save(data.__dict__ , torch_pack , pickle_protocol = 4)

        if y_labels is not None:  data.y.align_feature(y_labels)
        return data