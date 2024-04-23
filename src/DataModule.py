import gc , os
import numpy as np
import torch

from dataclasses import dataclass , field

from numpy.random import permutation
from torch import Tensor
from torch.utils.data import BatchSampler
from tqdm import tqdm
from typing import Any , Literal , Optional

from .data.PreProcess import pre_process
from .data.BlockData import DataBlock , DataBlockNorm
from .util import DataHook , Device , DataloaderStored , Storage , TrainConfig
from .func import tensor_standardize_and_weight , match_values

from .environ import DIR

def _abbr(data_type : str): return DataBlock.data_type_abbr(data_type)

@dataclass
class BatchData:
    '''custom data component of a batch'''
    x       : Tensor | tuple[Tensor] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor 
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: self.x = self.x[0]
        
    def to(self , device = None): return self.__class__(**{k:self.send_to(v , device) for k,v in self.__dict__.items()})
    def cpu(self):  return self.__class__(**{k:self.send_to(v , 'cpu') for k,v in self.__dict__.items()})
    def cuda(self): return self.__class__(**{k:self.send_to(v , 'cuda') for k,v in self.__dict__.items()})
    @property
    def is_empty(self): return len(self.y) == 0
    @classmethod
    def send_to(cls , obj , des : Any | Literal['cpu' , 'cuda']) -> Any:
        if obj is None: return None
        elif isinstance(obj , Tensor):
            if des == 'cpu': return obj.cpu()
            elif des == 'cuda': return obj.cuda()
            elif callable(des): return des(obj) 
            else: return obj.to(des)
        elif isinstance(obj , (list , tuple)):
            return type(obj)([cls.send_to(o , des) for o in obj])
        else: raise TypeError(obj)

class DataModule:
    '''A class to store relavant training data'''
    def __init__(self , config : Optional[TrainConfig] = None , predict : bool = False):
        '''
        1. load Package of BlockDatas of x , y , norms and index
        2. Setup model_date dataloaders
        3. Buffer dict for dynamic nn's
        '''
        self.config : TrainConfig = TrainConfig.load() if config is None else config
        self.predict : bool = predict

        self.device  = Device()
        self.storage = Storage('mem' if self.config.mem_storage else 'disk')

        self.datas = self.DataInterface.load(self.data_type_list, self.config.labels, self.predict, self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)
        self.model_date_list = self.datas.date_within(self.config.beg_date    , self.config.end_date , self.config.interval)
        self.test_full_dates = self.datas.date_within(self.config.beg_date + 1, self.config.end_date)

        self.static_prenorm_method = {}
        for mdt in self.data_type_list: 
            method = self.config.model_data_prenorm.get(mdt , {})
            method['divlast']  = method.get('divlast' , True) and (mdt in ['day'])
            method['histnorm'] = method.get('histnorm', True) and (self.datas.norms.get(mdt) is not None)
            print(f'Pre-Norming method of [{mdt}] : {method}')
            self.static_prenorm_method[mdt] = method

        self.reset_dataloaders()
        self.buffer = self.BufferSpace(self.config.buffer_type , self.config.buffer_param , self.device)

    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict , self.loader_param = {} , ()
    
    @property
    def data_type_list(self):
        '''get data type list (abbreviation)'''
        return [_abbr(data_type) for data_type in self.config.data_type_list]
    
    @staticmethod
    def prepare_data():
        '''prepare data for fit / test / predict'''
        pre_process(False)
        pre_process(True)

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , param = {} , model_date = -1) -> None:
        '''Create train/valid/test dataloaders if necessary'''
        if self.predict: stage = 'predict'
        
        seqlens : dict = param['seqlens']
        if self.config.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        if self.loader_param == (stage , model_date , seqlens): return

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        
        self.stage = stage
        self.loader_param = stage , model_date , seqlens

        x_keys = self.data_type_list
        y_keys = [k for k in seqlens.keys() if k in ['hist_loss','hist_preds','hist_labels']]
        self.seqs = {k:seqlens.get(k , 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([v for k,v in self.seqs.items() if k in y_keys]) if y_keys else 1
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys]) if x_keys else 1
        self.seq0 = self.seqx + self.seqy - 1

        if stage == 'fit':
            model_date_col = (self.datas.date < model_date).sum()
            step_interval = self.config.input_step_day
            d0 = max(0 , model_date_col - self.config.skip_horizon - self.config.input_span - self.seq0)
            d1 = max(0 , model_date_col - self.config.skip_horizon)
        elif stage in ['predict' , 'test']:
            next_model_date = max(self.test_full_dates) + 1
            if stage == 'predict':
                self.model_date_list = np.array([model_date])
            elif model_date != self.model_date_list[-1]:
                next_model_date = self.model_date_list[self.model_date_list > model_date][0]
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

        # record y to self to perform buffer_init
        self.y , _ = self.standardize_y(y , None , None , no_weight = True)
        self.buffer.process('setup' , self)

        valid = self.full_valid_sample(x , self.y , self.step_idx , **self.buffer.get(y_keys))
        y , w = self.standardize_y(self.y , valid , self.step_idx)

        self.y[:,self.step_idx] = y[:]
        self.buffer.process('update' , self)

        self.static_dataloader(x , y , w , valid)

        gc.collect() 
        torch.cuda.empty_cache()

    def prev_model_date(self , model_date):
        prev_dates = [d for d in self.model_date_list if d < model_date]
        return max(prev_dates) if prev_dates else -1
    
    def train_dataloader(self):
        return self.LoaderWrapper(self , self.loader_dict['train'] , self.device , self.config.verbosity)
    def val_dataloader(self):
        return self.LoaderWrapper(self , self.loader_dict['valid'] , self.device , self.config.verbosity)
    def test_dataloader(self):
        return self.LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def predict_dataloader(self):
        return self.LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        return batch.to(self.device if device is None else device)

    def full_valid_sample(self , x_data : dict[str,Tensor] , y : Tensor , index1 : Tensor , **kwargs) -> Tensor:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is 'day'
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        valid = self.valid_sample(y , index1) if self.stage == 'train' else torch.ones(len(y),len(index1)).to(torch.bool)
        for k , x in x_data.items(): valid *= self.valid_sample(x , index1 , self.seqs[k] , k in ['day'])
        for k , x in kwargs.items(): valid *= self.valid_sample(x , index1 , self.seqs[k])
        return valid
    
    @staticmethod
    def valid_sample(data : Tensor , index1 : Tensor , rolling_window = 1 , endpoint_nonzero = False):
        '''return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims'''
        data = torch.cat([torch.zeros_like(data[:,:rolling_window]) , data],dim=1).unsqueeze(2)
        sum_dim = tuple(range(2,data.ndim))
        
        invalid_samp = data[:,index1 + rolling_window].isnan().sum(sum_dim)
        for i in range(rolling_window - 1): invalid_samp += data[:,index1 + rolling_window - i - 1].isnan().sum(sum_dim)
        if endpoint_nonzero: invalid_samp += (data[:,index1 + rolling_window] == 0).sum(sum_dim)
        
        return (invalid_samp == 0)
     
    def standardize_y(self , y : Tensor , valid : Optional[Tensor] , index1 : Optional[Tensor] , no_weight = False) -> tuple[Tensor , Optional[Tensor]]:
        '''standardize y and weight'''
        if valid is not None:
            assert index1 is not None , index1
            y = y[:,index1].clone().nan_to_num(0)
            y[valid == 0] = torch.nan
        return tensor_standardize_and_weight(y , 0 , self.config.weight_scheme(self.stage , no_weight))
        
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Tensor) -> None:
        '''update loader_dict , save batch_data to f'{DIR.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(self.stage , valid , index0 , index1 , self.config.sample_method , 
                                         self.config.train_ratio , self.config.batch_size)
        self.storage.del_group(self.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = self.config.shuffle_option if set_key == 'train' else 'static'
            batch_files = [f'{DIR.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all()
                i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(index1 , b_i[:,1])

                b_x = [self.prenorm(self.rolling_rotation(x[mdt],self.seqs[mdt],i0,i1) , mdt) for mdt in x.keys()]
                b_y = y[i0 , yindex1]
                b_w = None if w is None else w[i0 , yindex1]
                b_v = valid[i0 , yindex1]

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.stage)
            self.loader_dict[set_key] = DataloaderStored(self.storage , batch_files , shuf_opt)

    @staticmethod
    def split_sample(stage , valid : Tensor , index0 : Tensor , index1 : Tensor ,
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
                pool = permutation(np.arange(valid.sum().item()))
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
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-6
        if self.static_prenorm_method[key]['histnorm']:
            x -= self.datas.norms[key].avg[-x.shape[-2]:]
            x /= self.datas.norms[key].std[-x.shape[-2]:] + 1e-6
        return x

    class BufferSpace:
        '''dynamic buffer space for some module to use (tra), can be updated at each batch / epoch '''
        def __init__(self , buffer_key : str | None , buffer_param : dict = {} , device : Optional[Device] = None , always_on_device = True) -> None:
            self.key = buffer_key
            self.param = buffer_param
            self.device = device
            self.always = always_on_device
            self.contents : dict[str,Any] = {}

            self.register_setup()
            self.register_update()

        def update(self , new = None):
            if new is not None: 
                if self.always and self.device is not None: new = self.device(new)
                self.contents.update(new)
            return self
        
        def __getitem__(self , key): return self.contents[key]
        def __setitem__(self , key , value): self.contents[key] = value

        def get(self , keys , default = None , keep_none = True):
            if hasattr(keys , '__len__'):
                result = {k:self.contents.get(k , default) for k in keys}
                if not keep_none: result = {k:v for k,v in result.items() if v is not None}
            else:
                result = self.contents.get(keys , default)
            if not self.always and self.device is not None: result = self.device(result)
            return result

        def process(self , stage : Literal['setup' , 'update'] , data_module):
            new = getattr(self , f'{stage}_wrapper')(data_module)
            if new is not None: 
                if self.always and self.device is not None: new = self.device(new)
                self.contents.update(new)
            return self
        
        def register_setup(self) -> None:
            # first param of wrapper is container, which represent self in ModelData
            if self.key == 'tra':
                def tra_wrapper(self_container , *args, **kwargs):
                    buffer = dict()
                    if self.param['tra_num_states'] > 1:
                        hist_loss_shape = list(self_container.y.shape)
                        hist_loss_shape[2] = self.param['tra_num_states']
                        buffer['hist_labels'] = self_container.y
                        buffer['hist_preds'] = torch.randn(hist_loss_shape)
                        buffer['hist_loss']  = (buffer['hist_preds'] - buffer['hist_labels'].nan_to_num(0)).square()
                    return buffer
                self.setup_wrapper = tra_wrapper
            else:
                self.setup_wrapper = self.none_wrapper
            
        def register_update(self) -> None:
            # first param of wrapper is container, which represent self in ModelData
            if self.key == 'tra':
                def tra_wrapper(self_container , *args, **kwargs):
                    buffer = dict()
                    if self.param['tra_num_states'] > 1:
                        buffer['hist_loss']  = (self_container.buffer['hist_preds'] - 
                                                self_container.buffer['hist_labels'].nan_to_num(0)).square()
                    return buffer
                self.update_wrapper = tra_wrapper
            else:
                self.update_wrapper = self.none_wrapper
            
        @staticmethod
        def none_wrapper(*args, **kwargs): return {}

    @dataclass
    class DataInterface:
        '''load datas / norms / index'''
        x : dict[str,DataBlock]
        y : DataBlock
        norms : dict[str,DataBlockNorm]
        secid : np.ndarray
        date  : np.ndarray

        def date_within(self , start , end , interval = 1) -> np.ndarray:
            return self.date[(self.date >= start) & (self.date <= end)][::interval]
        
        @classmethod
        def load(cls , data_type_list , y_labels = None , predict=False , dtype = torch.float , save_upon_loading = True):
            if dtype is None: dtype = torch.float
            if isinstance(dtype , str): dtype = getattr(torch , dtype)
            if predict: 
                torch_pack = 'no_torch_pack'
            else:
                last_date = max(DataBlock.load_dict(DataBlock.block_path('y'))['date'])
                torch_pack_code = '+'.join(data_type_list)
                torch_pack = f'{DIR.torch_pack}/{torch_pack_code}.{last_date}.pt'

            if os.path.exists(torch_pack):
                print(f'use {torch_pack}')
                data = cls(**torch.load(torch_pack))
            else:
                data_type_list = ['y' , *data_type_list]
                
                blocks = DataBlock.load_keys(data_type_list, predict , alias_search=True,dtype = dtype)
                norms  = DataBlockNorm.load_keys(data_type_list, predict , alias_search=True,dtype = dtype)

                y : DataBlock = blocks[0]
                x : dict[str,DataBlock] = {_abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
                norms = {_abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
                secid , date = blocks[0].secid , blocks[0].date

                assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

                data = cls(x , y , norms , secid , date)
                if not predict and save_upon_loading: 
                    os.makedirs(os.path.dirname(torch_pack) , exist_ok=True)
                    torch.save(data.__dict__ , torch_pack , pickle_protocol = 4)

            if y_labels is not None:  data.y.align_feature(y_labels)
            return data
    class LoaderWrapper:
        '''wrap loader to impletement DataModule Callbacks'''
        def __init__(self , data_module , raw_loader , device , verbosity = 0) -> None:
            self.data_module = data_module
            self.device = device
            self.verbosity = verbosity
            self.loader = raw_loader
            self.display_text = None

        def __len__(self):  return len(self.loader)
        def __getitem__(self , i : int): return self.process(list(self.loader)[i] , i)

        def __iter__(self):
            for batch_i , batch_data in enumerate(self.loader):
                yield self.process(batch_data , batch_i)        
        
        def process(self , batch_data : BatchData , batch_i : int) -> BatchData:
            batch_data = DataHook.on_before_batch_transfer(self.data_module , batch_data , batch_i)
            batch_data = DataHook.transfer_batch_to_device(self.data_module , batch_data , self.device , batch_i)
            batch_data = DataHook.on_after_batch_transfer(self.data_module , batch_data , batch_i)
            return batch_data

        def init_tqdm(self , text = ''):
            if self.verbosity >= 10:
                self.text = text
                self.loader = tqdm(self.loader , total=len(self.loader))
            return self

        def display(self , **kwargs):
            if isinstance(self.text , str) and isinstance(self.loader , tqdm): 
                self.loader.set_description(self.text.format(**kwargs))
