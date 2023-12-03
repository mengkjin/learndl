
import torch
import gc , random
import numpy as np
from function import *
from environ import get_config , cuda
from my_utils import versatile_storage
from gen_data import load_trading_data
from torch.utils.data.dataset import IterableDataset , Dataset

storage_loader = versatile_storage()
num_states = 3

class ModelData():
    """
    A class to store relavant training data , includes:
    1. Parameters: train_params , compt_params , model_data_type
    2. Datas: x_data , y_data , norm_param , index_stock , index_date
    3. Dataloader : yield x , y of training samples , create new ones if necessary
    """
    def __init__(self , model_data_type , config = None):
        CONFIG = get_config()
        if config is not None: CONFIG.update(config)
        storage_loader.activate(CONFIG['STORAGE_TYPE'])
        try:  
            self.precision = getattr(torch , CONFIG['PRECISION'])
        except:
            self.precision = torch.float
        self.model_data_type = model_data_type
        self.data_type_list  = self.model_data_type.split('+')
        self.x_data , self.y_data , self.norm_param , (self.index_stock , self.index_date) = load_trading_data(model_data_type , self.precision)
        self.stock_n , self.all_day_len = self.y_data.shape[:2]
        self.labels_n = min(self.y_data.shape[-1] , (lambda x:max(x) if isinstance(x,(list,tuple)) else x)(CONFIG['MODEL_PARAM']['num_output']))
        self.feat_dims = {mdt:v.shape[-1] for mdt,v in self.x_data.items()}

        self.model_date_list = self.index_date[(self.index_date >= CONFIG['BEG_DATE']) & (self.index_date <= CONFIG['END_DATE'])][::CONFIG['INTERVAL']]
        self.test_full_dates = self.index_date[(self.index_date > CONFIG['BEG_DATE']) & (self.index_date <= CONFIG['END_DATE'])]
        
        self.input_step = CONFIG['INPUT_STEP_DAY']
        self.test_step  = CONFIG['TEST_STEP_DAY']
        self.reset_dataloaders()

        self.buffer = {}
        self.buffer_init = None
        self.buffer_process = None
        self.CONFIG = CONFIG
        rmdir([f'./data/{k}_batch_path' for k in ['train' , 'valid' , 'test']] , remake_dir = True)

    def reset_dataloaders(self):        
        """
        Reset dataloaders and dataloader_param
        """
        self.dataloaders = {}
        self.dataloader_param = ()
        gc.collect() , torch.cuda.empty_cache()        
    
    def create_dataloader(self , process_name , style , model_date , seqlens):
        """
        Create train/valid dataloaders, used recurrently
        """
        assert style in ['train' , 'test']
        assert process_name in [style , 'instance']
        self.dataloader_param = (process_name , style , model_date , seqlens)
        gc.collect() , torch.cuda.empty_cache()

        self.dataloader_style = style
        self.process_name = process_name
        y_keys , x_keys = [k for k in seqlens.keys() if k in ['hist_loss','hist_preds','hist_labels']] , self.data_type_list
        self.seqs = {k:(seqlens[k] if k in seqlens.keys() else 1) for k in y_keys + x_keys}
        assert all([v > 0 for v in self.seqs.values()]) , self.seqs
        self.seqy = max([v for k,v in self.seqs.items() if k in y_keys])
        self.seqx = max([v for k,v in self.seqs.items() if k in x_keys])
        self.seq0 = self.seqx + self.seqy - 1

        if self.dataloader_style == 'train':
            model_date_col = (self.index_date < model_date).sum()    
            d0 = max(0 , model_date_col - self.CONFIG['SKIP_HORIZON'] - self.CONFIG['INPUT_SPAN'] - self.seq0)
            d1 = max(0 , model_date_col - self.CONFIG['SKIP_HORIZON'])
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx) // self.input_step
            # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
            #  (You can probably work around this by making a copy of your array  with array.copy().) 
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * self.input_step).copy() 
            self.date_idx = d0 + self.step_idx
        else:
            if model_date == self.model_date_list[-1]:
                next_model_date = self.CONFIG['END_DATE'] + 1
            else:
                next_model_date = self.model_date_list[self.model_date_list > model_date][0]
            test_step  = (1 if self.process_name == 'instance' else self.test_step)
            before_test_dates = self.index_date[self.index_date < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::test_step]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // test_step:]

            d0 = np.where(self.index_date == self.early_test_dates[0])[0][0] - self.seqx + 1
            d1 = np.where(self.index_date == self.model_test_dates[-1])[0][0] + 1
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx + 1) // test_step + (0 if self.day_len % test_step == 0 else 1)
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * self.test_step).copy() 
            self.date_idx = d0 + self.step_idx

        # data_func = lambda x:torch.nn.functional.pad(x[:,d0:d1] , (0,0,0,0,0,self.seq0-self.input_step,0,0) , value=np.nan)
        DataFunc = lambda d:d[:,d0:d1]
        x = {k:DataFunc(v) for k,v in self.x_data.items()}
        self.y = self.process_y_data(DataFunc(self.y_data).squeeze(2)[:,:,:self.labels_n] , None , no_weight = True)
        if self.buffer_init is not None: self.buffer.update(self.buffer_init(self))

        self.nonnan_sample = self.cal_nonnan_sample(x, self.y, **{k:v for k,v in self.buffer.items() if k in self.seqs.keys()})
        index = self.sample_index(self.nonnan_sample)
        y_step , w_step = self.process_y_data(self.y , self.nonnan_sample)
        self.y[:,self.step_idx] = y_step[:]
        if self.buffer_process is not None: self.buffer.update(self.buffer_process(self))

        self.buffer = cuda(self.buffer)
        self.static_dataloader(x , y_step , w_step , index , self.nonnan_sample)

        gc.collect() , torch.cuda.empty_cache()
        
    def buffer_functions(self , 
                         init = None , 
                         process = None ,
                         ):
        if init is not None: self.buffer_init = init
        if process is not None: self.buffer_process = process
    
    def cal_nonnan_sample(self , x , y , **kwargs):
        """
        return non-nan sample position (with shape of stock_n * step_len) the first 2 dims
        x : rolling window non-nan , end non-zero if in self.CONFIG['DATATYPE']['trade']
        y : exact point non-nan 
        others : rolling window non-nan , default as self.seqy
        """
        valid_sample = self._nonnan_sample_sub(y)
        for k , v in x.items():
            valid_sample *= self._nonnan_sample_sub(v , self.seqs[k] , k in self.CONFIG['DATATYPE']['trade'])
        for k , v in kwargs.items():
            valid_sample *= self._nonnan_sample_sub(v , self.seqs[k])
        return valid_sample > 0

    def _nonnan_sample_sub(self , data , rolling_window = 1 , endpoint_nonzero = False , index1 = None):
        """
        return non-nan sample position (with shape of stock_n * step_len) the first 2 dims
        x : rolling window non-nan
        y : exact point non-nan 
        """
        if index1 is None: index1 = self.step_idx # np.arange(rolling_window - 1 , data.shape[1])
        data = data.unsqueeze(2)
        index_pad = (self.step_idx + rolling_window)
        data_pad = torch.cat([torch.zeros_like(data)[:,:rolling_window] , data],dim=1)
        sum_dim = tuple(np.arange(data.dim())[2:])
        invalid_samp = data_pad[:,index_pad].isnan().sum(sum_dim)
        if endpoint_nonzero: invalid_samp += (data_pad[:,index_pad] == 0).sum(sum_dim)
        for i in range(rolling_window - 1):
            invalid_samp += data_pad[:,index_pad - i - 1].isnan().sum(sum_dim)
        return (invalid_samp == 0)
        
    def sample_index(self , nonnan_sample = None):
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:stock_n * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non
        """
        shp = nonnan_sample.shape
        if self.dataloader_style == 'train':
            ii_stock_wise = np.arange(shp[0] * shp[1])[nonnan_sample.flatten()]
            ii_time_wise  = np.arange(shp[0] * shp[1]).reshape(shp[1] , shp[0]).transpose().flatten()[ii_stock_wise]
            train_samples = int(len(ii_stock_wise) * self.CONFIG['TRAIN_PARAM']['dataloader']['train_ratio'])
            random.seed(self.CONFIG['TRAIN_PARAM']['dataloader']['random_seed'])
            if self.CONFIG['TRAIN_PARAM']['dataloader']['random_tv_split']:
                random.shuffle(ii_stock_wise)
                ii_train , ii_valid = ii_stock_wise[:train_samples] , ii_stock_wise[train_samples:]
            else:
                early_samples = ii_time_wise < sorted(ii_time_wise)[train_samples]
                ii_train , ii_valid = ii_stock_wise[early_samples] , ii_stock_wise[early_samples == 0]
            random.shuffle(ii_train) , random.shuffle(ii_valid)

        ipos = torch.zeros(shp[0] , shp[1] , 2 , dtype = int) # i_row (sec) , i_col_x (end)
        ipos[:,:,0] = torch.tensor(np.arange(shp[0] , dtype = int)).reshape(-1,1) 
        ipos[:,:,1] = torch.tensor(self.step_idx)
        ipos = ipos.reshape(-1 , ipos.shape[-1])

        if self.dataloader_style == 'train':
            i_train , i_valid = (ipos[ii_train] , ipos[ii_valid])
            return {'train': i_train, 'valid': i_valid} 
        else:
            return {'test': ipos} 
     
    def process_y_data(self , y , nonnan_sample , no_weight = False):
        weight_scheme = None if no_weight else self.CONFIG[f'WEIGHT_{self.dataloader_style.upper()}']
        if nonnan_sample is None:
            y_new = y
        else:
            y_new = torch.rand(*nonnan_sample.shape , *y.shape[2:])
            y_new[:] = y[:,self.step_idx].nan_to_num(0)
            y_new[nonnan_sample == 0] = np.nan
        y_new , w_new = tensor_standardize_and_weight(y_new , 0 , weight_scheme)
        return y_new if no_weight else (y_new , w_new)
    
    def static_dataloader(self , x , y , w , index , nonnan_sample):
        """
        1. update dataloaders dict(set_name = ['train' , 'valid']), save batch_data to './model/{model_name}/{set_name}_batch_data' and later load them
        """
        # init i (row , col position) and y (labels) matrix
        set_iter = ['train' , 'valid'] if self.dataloader_style == 'train' else ['test']
        storage_loader.del_group(self.dataloader_style)
        loaders = dict()
        #self.x = x
        #self.y = y
        #self.w = w
        #self.index = index
        for set_name in set_iter:
            if self.dataloader_style == 'train':
                set_i = index[set_name]
                batch_sampler = torch.utils.data.BatchSampler(range(len(set_i)) , self.CONFIG['BATCH_SIZE'] , drop_last = False)
            else:
                set_i = [index[set_name][index[set_name][:,1] == i1] for i1 in index[set_name][:,1].unique()]
                batch_sampler = range(len(set_i))

            batch_file_list = []
            for batch_num , pos in enumerate(batch_sampler):
                batch_file_list.append(f'./data/{set_name}_batch_path/{set_name}.{batch_num}.pt')
                batch_i = set_i[pos]
                assert torch.isin(batch_i[:,1] , torch.tensor(self.step_idx)).all()
                i0 , i1 , yi1 = batch_i[:,0] , batch_i[:,1] , match_values(self.step_idx , batch_i[:,1])
                batch_x = []
                batch_y = y[i0,yi1]
                batch_w = None if w is None else w[i0,yi1]
                for mdt in x.keys():
                    batch_x.append(self._norm_x(torch.cat([x[mdt][i0,i1+i+1-self.seqs[mdt]] for i in range(self.seqs[mdt])],dim=1),mdt))
                batch_x = batch_x[0] if len(batch_x) == 1 else tuple(batch_x)
                batch_nonnan = nonnan_sample[i0,yi1]
                batch_data = {'x':batch_x,'y':batch_y,'w':batch_w,'nonnan':batch_nonnan,'i':batch_i}
                storage_loader.save(batch_data, batch_file_list[-1] , group = self.dataloader_style)
            loaders[set_name] = dataloader_saved(batch_file_list)
        self.dataloaders.update(loaders)
        
    def _norm_x(self , x , key):
        """
        return panel_normalized x
        1.for ts-cols , divide by the last value, get seq-mormalized x
        2.for seq-mormalized x , normalized by history avg and std
        """
        if key in self.CONFIG['DATATYPE']['trade']:
            x /= x.select(-2,-1).unsqueeze(-2)
            x -= self.norm_param[key]['avg'][-x.shape[-2]:]
            x /= self.norm_param[key]['std'][-x.shape[-2]:] + 1e-4
        else:
            pass
        return x
            
class dataloader_saved:
    """
    class of saved dataloader , retrieve batch_data from './model/{model_name}/{set_name}_batch_data'
    """
    def __init__(self, batch_file_list):
        self.batch_file_list = batch_file_list
    def __len__(self):
        return len(self.batch_file_list)
    def __iter__(self):
        for batch_file in self.batch_file_list: 
            yield cuda(storage_loader.load(batch_file))
    
class Mydataset(Dataset):
    def __init__(self, data1 , label , weight = None) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
            self.weight = weight
    def __len__(self):
        return len(self.data1)
    def __getitem__(self , ii):
        if self.weight is None:
            return self.data1[ii], self.label[ii]
        else:
            return self.data1[ii], self.label[ii], self.weight[ii]

class MyIterdataset(IterableDataset):
    def __init__(self, data1 , label) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
    def __len__(self):
        return len(self.data1)
    def __iter__(self):
        for ii in range(len(self.data1)):
            yield self.data1[ii], self.label[ii]
    