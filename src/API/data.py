import gc
import numpy as np
import torch

from abc import abstractmethod
from numpy.random import permutation
from torch import Tensor
from torch.utils.data import BatchSampler
from typing import Any , Iterator , Literal , Optional

from ..classes import BaseDataModule , BatchData , BoosterData
from ..data import DataBlockNorm , DataProcessor , ModuleData , DataUpdater
from ..environ import PATH , CONF
from ..func import tensor_standardize_and_weight , match_values
from ..util import BufferSpace , DataloaderStored , Device , LoaderWrapper , Storage , TrainConfig

class DataAPI:
    @staticmethod
    def update(): 
        '''
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        '''
        DataUpdater.main()

    @staticmethod
    def prepare_train_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        DataProcessor.main(True)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        NetDataModule.prepare_data()

class _DataModule(BaseDataModule):
    @abstractmethod
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Tensor) -> None: ...

    @abstractmethod
    def split_sample(self , stage , valid : Tensor , index0 : Tensor , index1 : Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list]: ...
    
    def __init__(self , config : Optional[TrainConfig] = None , predict : bool = False):
        '''
        1. load Package of BlockDatas of x , y , norms and index
        2. Setup model_date dataloaders
        3. Buffer dict for dynamic nn's
        '''
        self.config  : TrainConfig = TrainConfig.load() if config is None else config
        self.predict : bool = predict
        self.device  = Device()
        self.storage = Storage('mem' if self.config.mem_storage else 'disk')

    def load_data(self):
        self.datas = ModuleData.load(self.data_type_list, self.config.labels, self.predict, self.config.precision)
        self.config.update_data_param(self.datas.x)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)
        if self.predict:
            self.model_date_list = self.datas.date[0]
            self.test_full_dates = self.datas.date[1:]
        else:
            self.model_date_list = self.datas.date_within(self.config.beg_date    , self.config.end_date , self.config.interval)
            self.test_full_dates = self.datas.date_within(self.config.beg_date + 1, self.config.end_date)

        self.static_prenorm_method = {}
        for mdt in self.data_type_list: 
            method : dict[str,bool] = self.config.model_data_prenorm.get(mdt , {})
            method['divlast']  = method.get('divlast' , True) and (mdt in DataBlockNorm.DIVLAST)
            method['histnorm'] = method.get('histnorm', True) and (mdt in DataBlockNorm.HISTNORM)
            print(f'Pre-Norming method of [{mdt}] : {method}')
            self.static_prenorm_method[mdt] = method

        self.reset_dataloaders()
        self.buffer = BufferSpace(self.config.buffer_type , self.config.buffer_param , self.device)
        return self
    
    @property
    def data_type_list(self):
        '''get data type list (abbreviation)'''
        return [ModuleData.abbr(data_type) for data_type in self.config.data_type_list]
    
    @staticmethod
    def prepare_data():
        DataProcessor.main(predict = False)
        DataProcessor.main(predict = True)

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , param = {'seqlens' : {'day': 30 , '30m': 30 , 'dms': 30}} , model_date = -1) -> None:
        if self.predict: stage = 'predict'
        seqlens : dict = param['seqlens']
        if self.config.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        
        self.stage = stage

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

    def full_valid_sample(self , x_data : dict[str,Tensor] , y : Tensor , index1 : Tensor , **kwargs) -> Tensor:
        '''
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in k is divlast
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        '''
        valid = self.valid_sample(y , index1) if self.stage == 'train' else torch.ones(len(y),len(index1)).to(torch.bool)
        for k , x in x_data.items(): valid *= self.valid_sample(x , index1 , self.seqs[k] , k in DataBlockNorm.DIVLAST)
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
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-6
        if self.static_prenorm_method[key]['histnorm']:
            x -= self.datas.norms[key].avg[-x.shape[-2]:]
            x /= self.datas.norms[key].std[-x.shape[-2]:] + 1e-6
        return x
    
class NetDataModule(_DataModule):
    def train_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['train'] , self.device , self.config.verbosity)
    def val_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['valid'] , self.device , self.config.verbosity)
    def test_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def predict_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        return batch.to(self.device if device is None else device)

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , param = {'seqlens' : {'day': 30 , '30m': 30 , 'dms': 30}} , model_date = -1) -> None:
        if self.predict: stage = 'predict'
        seqlens : dict = param['seqlens']
        if self.config.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        
        self.stage = stage

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
        
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Tensor) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(self.stage , valid , index0 , index1 , self.config.sample_method , 
                                         self.config.train_ratio , self.config.batch_size)
        self.storage.del_group(self.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = self.config.shuffle_option if set_key == 'train' else 'static'
            batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all()
                i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                b_x = [self.prenorm(self.rolling_rotation(x[mdt],self.seqs[mdt],i0,i1) , mdt) for mdt in x.keys()]
                b_y = y[i0 , yindex1]
                b_w = None if w is None else w[i0 , yindex1]
                b_v = valid[i0 , yindex1]

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.stage)
            self.loader_dict[set_key] = DataloaderStored(self.storage , batch_files , shuf_opt)

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

    
class BoosterDataModule(_DataModule):
    '''for boosting such as algo.boost.lgbm, create booster'''
    def train_dataloader(self) -> Iterator[BoosterData]: return self.loader_dict['train']
    def val_dataloader(self) -> Iterator[BoosterData]:   return self.loader_dict['valid']
    def test_dataloader(self) -> Iterator[BoosterData]:  return self.loader_dict['test']
    def predict_dataloader(self) -> Iterator[BoosterData]: return self.loader_dict['test']
        
    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , param = {'seqlens' : {'day': 30 , '30m': 30 , 'dms': 30}} , model_date = -1) -> None:
        if self.predict: stage = 'predict'
        seqlens : dict = param['seqlens']
        if self.config.tra_model: seqlens.update(param.get('tra_seqlens',{}))
        if self.loader_param == (stage , model_date , seqlens): return
        self.loader_param = stage , model_date , seqlens

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 and seqlens , (stage , model_date , seqlens)
        
        self.stage = stage

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

        # record y to self to perform buffer_init
        self.y , _ = self.standardize_y(y , None , None , no_weight = True)

        y , _ = self.standardize_y(self.y , None , self.step_idx)
        self.y[:,self.step_idx] = y[:]
        self.static_dataloader(x , y)

        gc.collect() 
        torch.cuda.empty_cache()

    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(y)) , self.step_idx
        sample_index = self.split_sample(self.stage , index0 , index1 , self.config.train_ratio)
        self.storage.del_group(self.stage)
        assert len(x) == 1 , len(x)
        mdt0 , x0 = [(k,v) for k,v in x.items()][0]
        for set_key , set_samples in sample_index.items():
            if set_key in ['train' , 'valid']:
                bb_x , bb_y , bb_d = [] , [] , []
                for bnum , b_i in enumerate(set_samples):
                    assert torch.isin(b_i[:,1] , index1).all()
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    b_x = self.prenorm(self.rolling_rotation(x0,self.seqs[mdt0],i0,i1) , mdt0) # [n_stock x seq_len x n_feat]
                    b_y = y[i0 , yindex1] # [n_stock x num_output]
                    assert b_x.shape[:2] == (len(self.y_secid) , self.seqs[mdt0]) , (b_x.shape , (len(self.y_secid) , self.seqs[mdt0]))
                    assert b_y.shape == (len(self.y_secid) , 1) , (b_y.shape , (len(self.y_secid) , 1))
                    assert all(yindex1 == yindex1[0]) , yindex1
                    b_x = b_x.reshape(len(self.y_secid),1,-1)

                    bb_x.append(b_x)
                    bb_y.append(b_y)
                    bb_d.append(self.y_date[index1[yindex1[0]]])
                bb_x = torch.concat(bb_x , dim = 1)
                bb_y = torch.concat(bb_y , dim = 1)
                bb_d = np.array(bb_d)
                bnum = 0
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt']
                self.storage.save(BoosterData(bb_x , bb_y , self.y_secid , bb_d) , batch_files[bnum] , group = self.stage)
            elif set_key == 'test':
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
                for bnum , b_i in enumerate(set_samples):
                    assert torch.isin(b_i[:,1] , index1).all()
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    b_x = self.prenorm(self.rolling_rotation(x0,self.seqs[mdt0],i0,i1) , mdt0) # [n_stock x seq_len x n_feat]
                    b_y = y[i0 , yindex1] # [n_stock x num_output]
                    assert b_x.shape[:2] == (len(self.y_secid) , self.seqs[mdt0]) , (b_x.shape , (len(self.y_secid) , self.seqs[mdt0]))
                    assert b_y.shape == (len(self.y_secid) , 1) , (b_y.shape , (len(self.y_secid) , 1))
                    assert all(yindex1 == yindex1[0]) , yindex1
                    b_x = b_x.reshape(len(self.y_secid),1,-1)
                    dates = np.array([self.y_date[index1[yindex1[0]]]])
                    self.storage.save(BoosterData(b_x , b_y , self.y_secid , dates) , batch_files[bnum] , group = self.stage)
            else:
                raise KeyError(set_key)
            self.loader_dict[set_key] = DataloaderStored(self.storage , batch_files)

    @staticmethod
    def split_sample(stage , index0 : Tensor , index1 : Tensor , train_ratio   : float = 0.8) -> dict[str,list]:
        l0 , l1 = len(index0) , len(index1)
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)

        def sequential_sampling(beg , end , posit = pos): return [posit[:,j] for j in range(beg , end)]
        
        sample_index = {}
        if stage == 'fit':
            # must be sequential
            sep = int(l1 * train_ratio)
            sample_index['train'] = sequential_sampling(0 , sep)
            sample_index['valid'] = sequential_sampling(sep , l1)
        else:
            # test dataloader should have the same length as dates, so no filtering of val[:,j].sum() > 0
            sample_index['test'] = sequential_sampling(0 , l1)
        return sample_index