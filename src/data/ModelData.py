import gc , os
import numpy as np
import pandas as pd

import torch

from dataclasses import dataclass , field
from typing import Literal , Any

from .BlockData import DataBlock , DataBlockNorm
from ..util import Device , Storage , DataloaderStored , BatchData
from ..func.basic import tensor_standardize_and_weight , match_values

from ..environ import DIR

class ModelData():
    """
    A class to store relavant training data , includes:
    1. Parameters: train_params , compt_params , model_data_type
    2. Datas: x_data , y_data , x_norm , index(secid , date)
    3. Dataloader : yield x , y of training samples , create new ones if necessary
    """
    def __init__(self , data_type_list , config = {} ,  if_train = True , **kwargs):
        # self.config = train_config() if config is None else config
        self.kwarg = self.ModelDataKwargs()
        self.kwarg.assign(config , **kwargs)
        self.data_type_list = self._type_list(data_type_list)

        self.if_train = if_train
        self.load_model_data()
        self.reset_dataloaders()
        if self.kwarg.tra_model and self.kwarg.buffer_type == 'tra':
            buffer_type = 'tra'
        else:
            buffer_type = None
        self.buffer: dict = {}
        self.buffer_init = self.define_buffer_init(buffer_type , **self.kwarg.buffer_param) 
        self.buffer_proc = self.define_buffer_proc(buffer_type , **self.kwarg.buffer_param)  

    @dataclass
    class ModelDataKwargs:
        device : Any = None
        labels : Any = None
        precision : Any = torch.float32
        mem_storage : bool = True
        beg_date : int = 20170101
        end_date : int = 20991231
        interval : int = 120
        skip_horizon : int = 20
        input_span   : int = 2400
        buffer_param : dict = field(default_factory=dict)
        weight_scheme: dict = field(default_factory=dict)
        num_output   : int = 1
        train_ratio  : float = 0.8
        sample_method  : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential'
        shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static'
        batch_size     : int = 10000
        input_step_day : int = 5
        test_step_day  : int = 1
        verbosity    : int = 5
        tra_model    : bool = False
        buffer_type  : str = 'tra'
        buffer_param : dict = field(default_factory=dict)

        def assign(self , config = {} , **kwargs):
            for key in self.__dict__.keys(): 
                value = None
                if kwargs.get(key):
                    value = kwargs[key]
                else:
                    if key == 'num_output':
                        value = config.get('MODEL_PARAM',{}).get('num_output')
                    elif key == 'weight_scheme':
                        value = config.get('train_params',{}).get('criterion',{}).get('weight')
                    elif key in ['train_ratio','sample_method']:
                        value = config.get('train_params',{}).get('dataloader',{}).get(key)
                    else:
                        value = config.get(key)
                if value is not None: setattr(self , key , value)

    def load_model_data(self):
        self.prenorming_method = {}
        self.x_data , self.y_data , self.norms , self.index = \
            self._load_data_pack(self.data_type_list, self.kwarg.labels, self.if_train, self.kwarg.precision)
        # self.x_data , self.y_data , self.norms , self.index = load_old_valid_data()
        # self.date , self.secid = self.index

        self.labels_n = min(self.y_data.shape[-1] , max(self.kwarg.num_output) if isinstance(self.kwarg.num_output,(list,tuple)) else self.kwarg.num_output)
        self.model_date_list = self.index[1][(self.index[1] >= self.kwarg.beg_date) & (self.index[1] <= self.kwarg.end_date)][::self.kwarg.interval]
        self.test_full_dates = self.index[1][(self.index[1] >  self.kwarg.beg_date) & (self.index[1] <= self.kwarg.end_date)]

    def reset_dataloaders(self):        
        """
        Reset dataloaders and dataloader_param
        """
        self.dataloaders = {}
        self.dataloader_param = ()
        self.device  = Device(self.kwarg.device)
        self.storage = Storage(self.kwarg.mem_storage)

        gc.collect() 
        torch.cuda.empty_cache()   
    
    def get_dataloader_param(self , loader_type , process_name = '' , model_date = -1 , param = {} , namespace = None):
        if namespace is not None:
            process_name , model_date , param = namespace.process_name , namespace.model_date , namespace.param
        seqlens = param['seqlens']
        if self.kwarg.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        
        return process_name , loader_type , model_date , seqlens

    def create_dataloader(self , *dataloader_param):
        """
        Create train/valid dataloaders, used recurrently
        """
        if self.dataloader_param == dataloader_param: return NotImplemented
        self.dataloader_param = process_name , loader_type , model_date , seqlens = dataloader_param

        assert loader_type in ['train' , 'test'] , loader_type
        assert process_name in [loader_type , 'instance'] , (process_name,loader_type)
        assert model_date > 0 , model_date

        gc.collect() 
        torch.cuda.empty_cache()

        self.loader_type = loader_type
        self.process_name = process_name
        y_keys , x_keys = [k for k in seqlens.keys() if k in ['hist_loss','hist_preds','hist_labels']] , self.data_type_list
        self.seqs = {k:(seqlens[k] if k in seqlens.keys() else 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([1]+[v for k,v in self.seqs.items() if k in y_keys])
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys])
        self.seq0 = self.seqx + self.seqy - 1

        if self.loader_type == 'train':
            model_date_col = (self.index[1] < model_date).sum()    
            d0 = max(0 , model_date_col - self.kwarg.skip_horizon - self.kwarg.input_span - self.seq0)
            d1 = max(0 , model_date_col - self.kwarg.skip_horizon)
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx) // self.kwarg.input_step_day
            # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
            #  (You can probably work around this by making a copy of your array  with array.copy().) 
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * self.kwarg.input_step_day).copy() 
            self.date_idx = d0 + self.step_idx
        else:
            if self.if_train:
                if model_date == self.model_date_list[-1]:
                    next_model_date = self.kwarg.end_date + 1
                else:
                    next_model_date = self.model_date_list[self.model_date_list > model_date][0]
            else:
                self.model_date_list = [model_date]
                next_model_date = max(self.test_full_dates) + 1
            test_step  = (1 if self.process_name == 'instance' else self.kwarg.test_step_day)
            before_test_dates = self.index[1][self.index[1] < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::test_step]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // test_step:] if self.seqy > 1 else test_dates[-1:-1]
            _cal_test_dates = np.concatenate([self.early_test_dates , self.model_test_dates])
    
            d0 = max(np.where(self.index[1] == _cal_test_dates[0])[0][0] - self.seqx + 1 , 0)
            d1 = np.where(self.index[1] == _cal_test_dates[-1])[0][0] + 1
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx + 1) // test_step + (0 if self.day_len % test_step == 0 else 1)
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * test_step).copy() 
            self.date_idx = d0 + self.step_idx

        x = {k:v.values[:,d0:d1] for k,v in self.x_data.items()}
        self.y , _ = self.process_y_data(self.y_data.values[:,d0:d1].squeeze(2)[...,:self.labels_n] , None , no_weight = True)
        self.y_secid , self.y_date = self.y_data.secid , self.y_data.date[d0:d1]

        self.buffer.update(self.buffer_init(self))
        self.nonnan_sample = self.cal_nonnan_sample(x, self.y, **{k:v for k,v in self.buffer.items() if k in self.seqs.keys()})
        y_step , w_step = self.process_y_data(self.y , self.nonnan_sample)
        self.y[:,self.step_idx] = y_step[:]
        
        self.buffer.update(self.buffer_proc(self))
        self.buffer = self.device(self.buffer) # type: ignore

        index = self.data_sampling(self.nonnan_sample)
        self.static_dataloader(x , y_step , w_step , index , self.nonnan_sample)

        gc.collect() 
        torch.cuda.empty_cache()
    
    def cal_nonnan_sample(self , x , y , **kwargs):
        """
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is 'day'
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        """
        valid_sample = True
        if self.if_train: valid_sample = self._nonnan_sample_sub(y)
        for k , v in x.items():
            valid_sample *= self._nonnan_sample_sub(v , self.seqs[k] , DataBlock.data_type_abbr(k) in ['day'])
        for k , v in kwargs.items():
            valid_sample *= self._nonnan_sample_sub(v , self.seqs[k])
        return valid_sample > 0

    def _nonnan_sample_sub(self , data , rolling_window = 1 , endpoint_nonzero = False , index1 = None):
        """
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan
        y : exact point non-nan 
        """
        if index1 is None: index1 = self.step_idx # np.arange(rolling_window - 1 , data.shape[1])
        data = data.unsqueeze(2)
        index_pad = index1 + rolling_window
        data_pad = torch.cat([torch.zeros_like(data)[:,:rolling_window] , data],dim=1)
        sum_dim = tuple(np.arange(data.dim())[2:])
        
        invalid_samp = data_pad[:,index_pad].isnan().sum(sum_dim)
        if endpoint_nonzero: 
            invalid_samp += (data_pad[:,index_pad] == 0).sum(sum_dim)
        for i in range(rolling_window - 1):
            invalid_samp += data_pad[:,index_pad - i - 1].isnan().sum(sum_dim)
        return (invalid_samp == 0)
     
    def process_y_data(self , y , nonnan_sample , no_weight = False):
        if no_weight:
            weight_scheme = None 
        else:
            weight_scheme = self.kwarg.weight_scheme.get(self.loader_type.lower() , 'equal')
        if nonnan_sample is None:
            y_new = y
        else:
            y_new = torch.rand(*nonnan_sample.shape , *y.shape[2:])
            y_new[:] = y[:,self.step_idx].nan_to_num(0)
            y_new[nonnan_sample == 0] = torch.nan
        y_new , w_new = tensor_standardize_and_weight(y_new , 0 , weight_scheme)
        return y_new , w_new
        
    def data_sampling(self , nonnan_sample):
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        """
        assert self.kwarg.sample_method in ['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] , self.kwarg.sample_method

        def _shuffle_sampling(ii , batch_size = self.kwarg.batch_size):
            pool = np.random.permutation(np.arange(len(ii)))
            return [ii[pos] for pos in torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)]

        pos = nonnan_sample
        shp = nonnan_sample.shape
        pij = torch.zeros(shp[0] , shp[1] , 2 , dtype = torch.int)
        pij[:,:,0] = torch.arange(shp[0] , dtype = torch.int).reshape(-1,1) 
        pij[:,:,1] = torch.tensor(self.step_idx)

        sample_index = {}
        if self.loader_type == 'train':
            dtrain = int(shp[1] * self.kwarg.train_ratio)
            if self.kwarg.sample_method == 'total_shuffle':
                pool = np.random.permutation(np.arange(pos.sum()))
                train_samples = int(len(pool) * self.kwarg.train_ratio)
                ii_train = pij[pos][pool[:train_samples]]
                ii_valid = pij[pos][pool[train_samples:]]
                sample_index['train'] = _shuffle_sampling(ii_train)
                sample_index['valid'] = _shuffle_sampling(ii_valid)
            elif self.kwarg.sample_method == 'both_shuffle':
                ii_train = pij[:,:dtrain][pos[:,:dtrain]]
                ii_valid = pij[:,dtrain:][pos[:,dtrain:]]
                sample_index['train'] = _shuffle_sampling(ii_train)
                sample_index['valid'] = _shuffle_sampling(ii_valid)
            elif self.kwarg.sample_method == 'train_shuffle':
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
        
    def static_dataloader(self , x , y , w , sample_index , nonnan_sample):
        """
        update dataloaders dict(set_name = ['train' , 'valid']), 
        save batch_data to f'{DIR.model}/{model_name}/{set_name}_batch_data' and later load them
        """
        self.storage.del_group(self.loader_type)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = self.kwarg.shuffle_option if set_key == 'train' else 'static'
            pbar_opt = self.kwarg.verbosity >= 10
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
    def selected_rolling_window(x , rw , index0 , index1 , dim = 1 , squeeze_out = True):
        assert x.ndim == 4 , x.ndim
        assert len(index0) == len(index1) , (len(index0) , len(index1))
        try:
            x_rw = x.unfold(dim , rw , 1)[index0 , index1 + 1 - rw].permute(0,3,1,2)
        except MemoryError:
            x_rw = torch.stack([x[index0 , index1+i+1-rw] for i in range(rw)],dim=dim)
        if squeeze_out: x_rw = x_rw.squeeze(-2)
        return x_rw
        
    def prenorm(self , x , key , static = True):
        """
        return panel_normalized x
        1.for ts-cols , divide by the last value, get seq-mormalized x
        2.for seq-mormalized x , normalized by history avg and std
        """
        if not static or self.prenorming_method.get(key) is None:
            prenorming_method: list[bool] = [DataBlock.data_type_abbr(key) in ['day'] , self.norms.get(key) is not None]
            if static:
                self.prenorming_method.update({key:prenorming_method})
                print(f'Pre-Norming method of [{key}] : [endpoint_division({prenorming_method[0]}) , history_standardize({prenorming_method[1]})]')
        else:
            prenorming_method: list[bool] = self.prenorming_method[key]
        
        if prenorming_method[0]:
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-6
        if prenorming_method[1]:
            x -= self.norms[key].avg[-x.shape[-2]:]
            x /= self.norms[key].std[-x.shape[-2]:] + 1e-6

        return x

    @staticmethod
    def define_buffer_init(key , **param):
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
    def define_buffer_proc(key , **param):
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
    
    @classmethod
    def _load_data_pack(cls , data_type_list , y_labels = None , if_train=True,dtype = torch.float):
        if dtype is None: dtype = torch.float
        data_type_list = cls._type_list(data_type_list)
        data = cls._load_torch_pack(data_type_list , y_labels , if_train)
        if isinstance(data , str):
            path_torch_pack = data
            if isinstance(dtype , str): dtype = getattr(torch , dtype)
            data_type_list = ['y' , *data_type_list]
            
            blocks = DataBlock.load_keys(data_type_list, if_train , alias_search=True,dtype = dtype)
            norms  = DataBlockNorm.load_keys(data_type_list, if_train , alias_search=True,dtype = dtype)

            y : DataBlock = blocks[0]
            if y_labels is not None: 
                ifeat = np.concatenate([np.where(y.feature == label)[0] for label in y_labels])
                y.update(values = y.values[...,ifeat] , feature = y.feature[ifeat])
                assert np.array_equal(y_labels , y.feature) , (y_labels , y.feature)

            x : dict[str,DataBlock] = {DataBlock.data_type_abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
            norms = {DataBlock.data_type_abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
            secid , date = blocks[0].secid , blocks[0].date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x':x,'y':y,'norms':norms,'secid':secid,'date':date}
            if if_train: DataBlock.save_dict(data , path_torch_pack)

        x, y, norms, secid, date = data['x'], data['y'], data['norms'], data['secid'], data['date']
        return x , y , norms , (secid , date)

    @classmethod
    def _load_torch_pack(cls , data_type_list , y_labels , if_train=True):
        if not if_train: return 'no_torch_pack'
        last_date = max(DataBlock.load_dict(DataBlock.block_path('y'))['date'])
        path_torch_pack = f'{DIR.torch_pack}/{cls._modal_data_code(data_type_list , y_labels)}.{last_date}.pt'

        if os.path.exists(path_torch_pack):
            print(f'use {path_torch_pack}')
            return torch.load(path_torch_pack)
        else:
            return path_torch_pack

    @staticmethod
    def _type_list(model_data_type):
        if isinstance(model_data_type , str): model_data_type = model_data_type.split('+')
        return [DataBlock.data_type_abbr(tp) for tp in model_data_type]

    @staticmethod
    def _modal_data_code(type_list , y_labels):
        xtype = '+'.join([DataBlock.data_type_abbr(tp) for tp in type_list])
        ytype = 'ally' if y_labels is None else '+'.join([DataBlock.data_type_abbr(tp) for tp in y_labels])
        return '+'.join([xtype , ytype])