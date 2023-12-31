
import torch
import gc , random , math , psutil , time , copy
import numpy as np
from .DataTank import DataTank
from .DataUpdater import get_db_path , get_db_file
from ..function.basic import *
from ..function.date import *
from ..util.environ import get_config , cuda , DIR_data
from ..util.basic import versatile_storage , timer
from torch.utils.data.dataset import IterableDataset , Dataset

DIR_block      = f'{DIR_data}/block_data'
DIR_hist_norm  = f'{DIR_data}/hist_norm'
DIR_torchpack  = f'{DIR_data}/torch_pack'
save_block_method = 'npz'
    
storage_loader = versatile_storage()

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
            
class Mydataloader_basic:
    def __init__(self, x_set , y_set , batch_size = 1, num_worker = 0, set_name = '', batch_num = None):
        self.dataset = Mydataset(x_set, y_set)
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataloader = torch.utils.data.DataLoader(self.dataset , batch_size = batch_size , num_workers = num_worker)
        self.set_name = set_name
        self.batch_num = math.ceil(len(y_set)/batch_size)
    def __iter__(self):
        for d in self.dataloader: 
            yield d

class Mydataloader_saved:
    def __init__(self, set_name , batch_num , batch_folder):
        self.set_name = set_name
        self.batch_num = batch_num
        self.batch_folder = batch_folder
        self.batch_path = [f'{self.batch_folder}/{self.set_name}.{ii}.pt' for ii in range(self.batch_num)]
    def __iter__(self):
        for ii in range(self.batch_num): 
            yield torch.load(self.batch_path[ii])
            
class ModelData():
    """
    A class to store relavant training data , includes:
    1. Parameters: train_params , compt_params , model_data_type
    2. Datas: x_data , y_data , x_norm , index(secid , date)
    3. Dataloader : yield x , y of training samples , create new ones if necessary
    """
    def __init__(self , model_data_type , config = None):
        self.CONFIG = get_config()
        if config is not None: self.CONFIG.update(config)
        storage_loader.activate(self.CONFIG['STORAGE_TYPE'])
        self.data_type_list = _type_list(model_data_type)
        self.x_data , self.y_data , self.norms , self.index = load_model_data(self.data_type_list , self.CONFIG['LABELS'] , self.CONFIG['PRECISION'])
        self.date , self.secid = self.index

        num_out = self.CONFIG['MODEL_PARAM']['num_output']
        self.labels_n = min(self.y_data.shape[-1] , max(num_out) if isinstance(num_out,(list,tuple)) else num_out)
        self.feat_dims = {mdt:v.shape[-1] for mdt,v in self.x_data.items()}

        _beg , _end , _int = self.CONFIG['BEG_DATE'] , self.CONFIG['END_DATE'] , self.CONFIG['INTERVAL']
        self.model_date_list = self.index[1][(self.index[1] >= _beg) & (self.index[1] <= _end)][::_int]
        self.test_full_dates = self.index[1][(self.index[1] >  _beg) & (self.index[1] <= _end)]
        
        self.input_step = self.CONFIG['INPUT_STEP_DAY']
        self.test_step  = self.CONFIG['TEST_STEP_DAY']
        self.reset_dataloaders()

        self.buffer = {}
        self.buffer_functions()
        rmdir([f'./data/minibatch/{k}' for k in ['train' , 'valid' , 'test']] , remake_dir = True)

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
            model_date_col = (self.index[1] < model_date).sum()    
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
            before_test_dates = self.index[1][self.index[1] < min(self.test_full_dates)][-self.seqy:]
            test_dates = np.concatenate([before_test_dates , self.test_full_dates])[::test_step]
            self.model_test_dates = test_dates[(test_dates > model_date) * (test_dates <= next_model_date)]
            self.early_test_dates = test_dates[test_dates <= model_date][-(self.seqy-1) // test_step:]

            d0 = np.where(self.index[1] == self.early_test_dates[0])[0][0] - self.seqx + 1
            d1 = np.where(self.index[1] == self.model_test_dates[-1])[0][0] + 1
            self.day_len  = d1 - d0
            self.step_len = (self.day_len - self.seqx + 1) // test_step + (0 if self.day_len % test_step == 0 else 1)
            self.step_idx = np.flip(self.day_len - 1 - np.arange(0 , self.step_len) * self.test_step).copy() 
            self.date_idx = d0 + self.step_idx

        # data_func = lambda x:torch.nn.functional.pad(x[:,d0:d1] , (0,0,0,0,0,self.seq0-self.input_step,0,0) , value=np.nan)
        DataFunc = lambda d:d[:,d0:d1]
        x = {k:DataFunc(v.values) for k,v in self.x_data.items()}
        self.y = self.process_y_data(DataFunc(self.y_data.values).squeeze(2)[:,:,:self.labels_n] , None , no_weight = True)
        if self.buffer_init is not None: self.buffer.update(self.buffer_init(self))

        self.nonnan_sample = self.cal_nonnan_sample(x, self.y, **{k:v for k,v in self.buffer.items() if k in self.seqs.keys()})
        y_step , w_step = self.process_y_data(self.y , self.nonnan_sample)
        self.y[:,self.step_idx] = y_step[:]
        if self.buffer_proc is not None: self.buffer.update(self.buffer_proc(self))

        self.buffer = cuda(self.buffer)

        #index = self.sample_index(self.nonnan_sample)
        #self.static_dataloader(x , y_step , w_step , index , self.nonnan_sample)

        index = self.sample_index2(self.nonnan_sample)
        self.static_dataloader2(x , y_step , w_step , index , self.nonnan_sample)

        gc.collect() , torch.cuda.empty_cache()
        
    def buffer_functions(self):
        self.buffer_type  = self.CONFIG['buffer_type']
        self.buffer_param = self.CONFIG['buffer_param']
        if self.CONFIG['TRA_switch'] and self.buffer_type == 'tra':
            self.buffer_type == None
        self.buffer_init = buffer_init(self.buffer_type , **self.buffer_param) 
        self.buffer_proc = buffer_proc(self.buffer_type , **self.buffer_param)
    
    def cal_nonnan_sample(self , x , y , **kwargs):
        """
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
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
        return non-nan sample position (with shape of len(index[0]) * step_len) the first 2 dims
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
    
    def sample_index(self , nonnan_sample = None):
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
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
                batch_file_list.append(f'./data/minibatch/{set_name}/{set_name}.{batch_num}.pt')
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
        
    def sample_index2(self , nonnan_sample = None):
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        """

        train_ratio = self.CONFIG['TRAIN_PARAM']['dataloader']['train_ratio']
        sample_method = self.CONFIG['TRAIN_PARAM']['dataloader']['sample_method']
        assert sample_method in ['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] , sample_method
        batch_size = self.CONFIG['BATCH_SIZE']
        
        random.seed(self.CONFIG['TRAIN_PARAM']['dataloader']['random_seed'])

        shp = nonnan_sample.shape
        ipos = torch.zeros(shp[0] , shp[1] , 2 , dtype = int)
        ipos[:,:,0] = torch.tensor(np.arange(shp[0] , dtype = int)).reshape(-1,1) 
        ipos[:,:,1] = torch.tensor(self.step_idx)

        if self.dataloader_style == 'train':
            sample_index = {'train': [] , 'valid': []} 
            train_dates = int(shp[1] * train_ratio)

            if sample_method == 'total_shuffle':
                iipos = ipos[nonnan_sample]
                train_samples = int(len(iipos) * train_ratio)
                pool = np.arange(len(iipos))
                random.shuffle(pool)
                ii_train , ii_valid = iipos[:train_samples] , iipos[train_samples:]

                pool = np.arange(len(ii_train))
                random.shuffle(pool)
                batch_sampler = torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)
                for pos in batch_sampler: sample_index['train'].append(ii_train[pos])

                pool = np.arange(len(ii_valid))
                random.shuffle(pool)
                batch_sampler = torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)
                for pos in batch_sampler: sample_index['valid'].append(ii_valid[pos])

            elif sample_method == 'both_shuffle':
                ii_train = ipos[:,:train_dates][nonnan_sample[:,:train_dates]]
                pool = np.arange(len(ii_train))
                random.shuffle(pool)
                batch_sampler = torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)
                for pos in batch_sampler: sample_index['train'].append(ii_train[pos])

                ii_valid = ipos[:,train_dates:][nonnan_sample[:,train_dates:]]
                pool = np.arange(len(ii_valid))
                random.shuffle(pool)
                batch_sampler = torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)
                for pos in batch_sampler: sample_index['valid'].append(ii_valid[pos])

            elif sample_method == 'train_shuffle':
                ii_train = ipos[:,:train_dates][nonnan_sample[:,:train_dates]]
                pool = np.arange(len(ii_train))
                random.shuffle(pool)
                batch_sampler = torch.utils.data.BatchSampler(pool , batch_size , drop_last=False)
                for pos in batch_sampler: sample_index['train'].append(ii_train[pos])

                for pos in range(train_dates , shp[1]): sample_index['valid'].append(ipos[:,pos][nonnan_sample[:,pos]])

            else:
                for pos in range(0 , train_dates): sample_index['train'].append(ipos[:,pos][nonnan_sample[:,pos]])

                for pos in range(train_dates , shp[1]): sample_index['valid'].append(ipos[:,pos][nonnan_sample[:,pos]])
        else:
            sample_index = {'test': []} 
            for pos in range(shp[1]): sample_index['test'].append(ipos[:,pos][nonnan_sample[:,pos]])

        return sample_index
        
    def static_dataloader2(self , x , y , w , sample_index , nonnan_sample):
        """
        1. update dataloaders dict(set_name = ['train' , 'valid']), save batch_data to './model/{model_name}/{set_name}_batch_data' and later load them
        """
        # init i (row , col position) and y (labels) matrix
        storage_loader.del_group(self.dataloader_style)
        loaders = dict()
        for set_name in sample_index.keys():
            batch_file_list = []
            for batch_num , batch_i in enumerate(sample_index[set_name]):
                batch_file_list.append(f'./data/minibatch/{set_name}/{set_name}.{batch_num}.pt')
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
        if key in self.norms[key]:
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-4
            x -= self.norms[key]['avg'][-x.shape[-2]:]
            x /= self.norms[key]['std'][-x.shape[-2]:] + 1e-4
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

def buffer_init(key , **param):
    # first param of wrapper is container, which represent self in ModelData
    if key == 'tra':
        def wrapper(self_container , *args, **kwargs):
            buffer = dict()
            if param['tra_num_states'] > 1:
                hist_loss_shape = list(self_container.y.shape)
                hist_loss_shape[2] = param['tra_num_states']
                buffer['hist_labels'] = self_container.y
                buffer['hist_preds'] = torch.randn(hist_loss_shape)
                buffer['hist_loss']  = (buffer['hist_preds'] - buffer['hist_labels']).square()
            return buffer
    else:
        wrapper = None
    return wrapper

def buffer_proc(key , **param):
    # first param of wrapper is container, which represent self in ModelData
    if key == 'tra':
        def wrapper(self_container , *args, **kwargs):
            buffer = dict()
            if param['tra_num_states'] > 1:
                buffer['hist_loss']  = (self_container.buffer['hist_preds'] - 
                                        self_container.buffer['hist_labels']).square()
            return buffer
    else:
        wrapper = None
    return wrapper

class DataBlock():
    def __init__(self , values = None , secid = None , date = None , feature = None) -> None:
        self.initiate = False
        if values is not None:
            self._init_attr(values , secid , date , feature)

    def _init_attr(self , values , secid , date , feature):
        if values is None: 
            self._clear_attr()
            return NotImplemented
        self.initiate = True
        if isinstance(feature , str): feature = [feature]
        if values.ndim == 3: values = values[:,:,None]
        assert values.shape == (len(secid),len(date),values.shape[2],len(feature))
        self.values  = values
        self.secid   = secid
        self.date    = date
        self.feature = feature
        self.shape   = self.values.shape

    def _clear_attr(self):
        self.initiate = False
        for attr_name in ['values' , 'secid' , 'date' , 'feature' , 'shape']:
            if hasattr(self,attr_name): delattr(self,attr_name)

    def __repr__(self):
        if self.initiate:
            return '\n'.join([
                'initiated ' + str(self.__class__) ,
                f'values shape {self.values.shape}'
            ])
        else:
            return 'uninitiate ' + str(self.__class__) 
    
    def update(self , **kwargs):
        valid_keys = np.intersect1d(['values','secid','date','feature'],list(kwargs.keys()))
        [setattr(self,k,kwargs[k]) for k in valid_keys]
        self.shape  = self.values.shape

    def from_dtank(self , dtank , inner_path , 
                   start_dt = None , end_dt = None , 
                   feature = None , **kwargs):
        portal = dtank.get_object(inner_path)
        if portal is None: return NotImplemented

        date = np.array(list(portal.keys())).astype(int)
        if start_dt is not None: date = date[date >= start_dt]
        if end_dt   is not None: date = date[date <= end_dt]
        if len(date) == 0: return NotImplemented

        datas = {str(d):dtank.read_data1D([inner_path , str(d)],feature).to_kline() for d in date}
        secid , p_s0 , p_s1 = index_union([data.secid for data in datas.values()])
        date    = np.array(list(datas.keys())).astype(int)
        feature , _ , p_f1 = index_intersect([data.feature for data in datas.values()])
        new_shape = [len(secid),len(date),len(feature)]
        if datas[str(date[0])].values.ndim == 3:
            new_shape.insert(2 , datas[str(date[0])].values.shape[1])
        newdata = np.full(tuple(new_shape) , np.nan , dtype = float)
        for i,(k,v) in enumerate(datas.items()):
            newdata[p_s0[i],i,:] = v.values[p_s1[i]][...,p_f1[i]]
        return newdata , secid , date , feature
    
    def from_db(self , db_key , inner_path , start_dt = None , end_dt = None , feature = None , **kwargs):
        db_path = get_db_path(db_key)
        dtanks = [DataTank(os.path.join(db_path,fn),'r') for fn in os.listdir(db_path)]
        datas = [self.from_dtank(dtank,inner_path,start_dt,end_dt,feature) for dtank in dtanks]
        [dtank.close() for dtank in dtanks]

        if len(datas) == 0: 
            pass
        elif len(datas) == 1:
            self._init_attr(*datas[0])
        else:
            secid , p_s0 , p_s1 = index_union([data[1] for data in datas])
            date  , p_d0 , p_d1 = index_union([data[2] for data in datas])
            feature , _  , p_f1 = index_intersect([data[3] for data in datas])
            new_shape = [len(secid),len(date),datas[0][0].shape[2],len(feature)]
            newdata = np.full(tuple(new_shape) , np.nan , dtype = float)
            for i , data in enumerate(datas):
                tmp = newdata[p_s0[i]]
                tmp[:,p_d0[i]] = data[0][p_s1[i]][:,p_d1[i]][...,p_f1[i]]
                newdata[p_s0[i]] = tmp[:]
            self._init_attr(newdata , secid , date , feature)
        return self
    
    def merge_others(self , others : list):
        if len(others) == 0: return NotImplemented
        blocks = [self,*others] if self.initiate else others
        if len(blocks) == 1:
            b = blocks[0]
            self._init_attr(b.values , b.secid , b.date , b.feature)
            return self
            
        values = [blk.values for blk in blocks]
        secid  , p_s0 , p_s1 = index_union([blk.secid for blk in blocks])
        date   , p_d0 , p_d1 = index_union([blk.date for blk in blocks])
        l1 = len(np.unique(np.concatenate([blk.feature for blk in blocks])))
        l2 = sum([len(blk.feature) for blk in blocks])
        distinct_feature = (l1 == l2)

        for i , data in enumerate(values):
            newdata = np.full((len(secid),len(date),*data.shape[2:]) , np.nan)
            tmp = newdata[p_s0[i]]
            tmp[:,p_d0[i]] = data[p_s1[i]][:,p_d1[i]]
            newdata[p_s0[i]] = tmp
            values[i] = newdata

        if distinct_feature:
            feature = np.concatenate([blk.feature for blk in blocks])
            newdata = np.concatenate(values , axis = -1)
        else:
            feature, p_f0 , p_f1 = index_union([blk.feature for blk in blocks])
            newdata = np.full((*newdata[0].shape[:-1],len(feature)) , np.nan , dtype = float)
            for i , data in enumerate(values):
                newdata[...,p_f0[i]] = data[...,p_f1[i]]
        self._init_attr(newdata , secid , date , feature)
        return self
    
    def save(self , file_path , start_dt = None , end_dt = None):
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        _save_dict_data(data , file_path)
    
    def load(self , file_path):
        self._init_attr(**_load_dict_data(file_path))
        return self
    
    def to(self , asTensor = None, dtype = None):
        if asTensor: self.values = torch.Tensor(self.values)
        if dtype: self.values = self.values.to(dtype)

def _save_dict_data(data , file_path):
    if file_path is None: return NotImplemented
    os.makedirs(os.path.dirname(file_path) , exist_ok=True)
    if file_path.endswith('.npz'):
        np.savez_compressed(file_path , **data)
    elif file_path.endswith('.pt'):
        torch.save(data , file_path , pickle_protocol = 4)
    else:
        raise Exception(file_path)
    
def _load_dict_data(file_path , keys = None):
    if file_path.endswith('.npz'):
        file = np.load(file_path)
    elif file_path.endswith('.pt'):
        file = torch.load(file_path)
    else:
        raise Exception(file_path)
    if keys is None: 
        keys = file.keys()
    else:
        keys = np.intersect1d(keys , list(file.keys()))
    return {k:file[k] for k in keys}

def _alias_path(file_path , _alias = lambda x:[x , f'trade_{x}' , x.replace('trade_','')]):
    dirname , basename = os.path.dirname(file_path) , os.path.basename(file_path)
    for _basename in _alias(basename):
        if os.path.exists(f'{dirname}/{_basename}'): 
            return f'{dirname}/{_basename}'
    raise Exception(f'No file of key {basename} in {dirname}')

def load_blocks(file_paths , 
                fillna = 'guess' , 
                intersect_secid = True ,
                union_date = True , 
                start_dt = None , end_dt = None ,
                dtype = torch.float):
    _guess = lambda ls,excl:[os.path.basename(x).lower().startswith(excl) == 0 for x in ls]
    if fillna == 'guess':
        exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m')
        fillna = np.array(_guess(file_paths , exclude_list))
    elif fillna is None or isinstance(fillna , bool):
        fillna = np.repeat(fillna , len(file_paths))
    else:
        assert len(file_paths) == len(fillna) , (len(file_paths) , len(fillna))
    
    with timer(f'Load  {len(file_paths)} DataBlocks') as t:
        blocks = [DataBlock().load(path) for path in file_paths]

    with timer(f'Align {len(file_paths)} DataBlocks') as t:
        newsecid = newdate = None
        if intersect_secid: 
            newsecid,p_s0,p_s1 = index_intersect([blk.secid for blk in blocks])
        if union_date: 
            newdate ,p_d0,p_d1 = index_union([blk.date for blk in blocks] , start_dt , end_dt)
        
        for i , blk in enumerate(blocks):
            secid = newsecid if newsecid is not None else blk.secid
            date  = newdate  if newdate  is not None else blk.date
            if blk.shape[:2] != (len(secid),len(date)): # secid/date alter
                values = np.full((len(secid),len(date),*blk.shape[2:]) , np.nan)
                if newsecid is None:
                    values[:,p_d0[i]] = blk.values[:,p_d1[i]]
                elif newdate is None:
                    values[p_s0[i]] = blk.values[p_s1[i]]
                else:
                    tmp = values[p_s0[i]]
                    tmp[:,p_d0[i]] = blk.values[p_s1[i]][:,p_d1[i]]
                    values[p_s0[i]] = tmp
            else:
                values = blk.values

            date_slice = np.repeat(True , len(date))
            if start_dt is not None: date_slice[date < start_dt] = False
            if end_dt   is not None: date_slice[date > end_dt]   = False
            values , date = values[:,date_slice] , date[date_slice]

            if fillna[i]: values = forward_fillna(values , axis = 1)
            blk.update(values = values , secid = secid , date = date)
            blk.to(asTensor = True , dtype = dtype)
    return blocks

def load_norms(file_paths , dtype = None):
    norms = []
    for path in file_paths:
        if not os.path.exists(path):
            norms.append(None)
        else:
            data = _load_dict_data(path)
            avg , std = torch.Tensor(data['avg']) , torch.Tensor(data['std'])
            if dtype: avg , std = avg.to(dtype) , std.to(dtype)
            norms.append({'avg':avg,'std':std})
    return norms
    
def load_model_data(data_type_list , y_labels = None , dtype = torch.float):
    data = load_torch_pack(data_type_list , y_labels)
    if isinstance(data , str):
        path_torch_pack = data
        if isinstance(dtype , str): dtype = getattr(torch , dtype)
        data_type_list = ['y' , *data_type_list]

        block_paths = [(path_block_data(k,alias_search=True)) for k in data_type_list]
        norm_paths  = [(path_norm_data(k,alias_search=True)) for k in data_type_list]
        
        blocks = load_blocks(block_paths ,dtype = dtype)
        norms  = load_norms(norm_paths ,dtype = dtype)

        y = blocks[0]
        if y_labels is not None: 
            ifeat = np.concatenate([np.where(y.feature == label)[0] for label in y_labels])
            y.update(values = y.values[...,ifeat] , feature = y.feature[ifeat])
            assert np.array_equal(y_labels , y.feature) , (y_labels , y.feature)

        x = {_type_abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
        norms = {_type_abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
        secid , date = blocks[0].secid , blocks[0].date

        assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

        data = {'x':x,'y':y,'norms':norms,'secid':secid,'date':date}
        _save_dict_data(data , path_torch_pack)

    x, y, norms, secid, date = data['x'], data['y'], data['norms'], data['secid'], data['date']
    return x , y , norms , (secid , date)

def load_torch_pack(data_type_list , y_labels):
    last_date = max(_load_dict_data(path_block_data('y'))['date'])
    path_torch_pack = f'{DIR_torchpack}/{_modal_data_code(data_type_list , y_labels)}.{last_date}.pt'

    if os.path.exists(path_torch_pack):
        print(f'use {path_torch_pack}')
        return torch.load(path_torch_pack)
    else:
        return path_torch_pack

def _type_abbr(key):
    if (key.startswith('trade_') and len(key)>6):
        return key[6:]
    elif key.startswith(('rtn_lag','res_lag')):
        return f'{key[:3]}{sum([int(s) for s in key[7:].split("_")])}'
    else:
        return key

def _type_list(model_data_type):
    if isinstance(model_data_type , str): model_data_type = model_data_type.split('+')
    return [_type_abbr(tp) for tp in model_data_type]

def _modal_data_code(type_list , y_labels):
    xtype = '+'.join([_type_abbr(tp) for tp in type_list])
    ytype = 'ally' if y_labels is None else '+'.join([_type_abbr(tp) for tp in y_labels])
    return '+'.join([xtype , ytype])

def _astype(data , dtype):
    if isinstance(data , dict):
        return {k:_astype(v,dtype) for k,v in data.items()}
    elif isinstance(data , (list,tuple)):
        return type(data)([_astype(v,dtype) for v in data])
    else:
        return data.to(dtype)

def block_mask(data_block , mask = True , after_ipo = 91 , **kwargs):
    if not mask: return data_block
    assert isinstance(data_block , DataBlock) , type(data_block)

    with DataTank(get_db_file(get_db_path('information')) , 'r') as info:
        desc = info.read_dataframe('stock/description')
    desc = desc[desc['secid'] > 0].loc[:,['secid','list_dt','delist_dt']]
    if len(np.setdiff1d(data_block.secid , desc['secid'])) > 0:
        add_df = pd.DataFrame({
                 'secid':np.setdiff1d(data_block.secid , desc['secid']) ,
                 'list_dt':21991231 , 'delist_dt':21991231})
        desc = pd.concat([desc,add_df]).reset_index(drop=True)

    desc = desc.sort_values('list_dt',ascending=False).drop_duplicates(subset=['secid'],keep='first').set_index('secid') 
    secid , date = data_block.secid , data_block.date
    
    list_dt = desc.loc[secid , 'list_dt'] 
    list_dt[list_dt < 0] = 21991231
    list_dt = date_offset(list_dt , after_ipo , astype = int)

    delist_dt = desc.loc[secid , 'delist_dt'] 
    delist_dt[delist_dt < 0] = 21991231

    mask = np.stack([(date <= l) + (date >= d) for l,d in zip(list_dt,delist_dt)],axis = 0) 
    data_block.values[mask] = np.nan
    return data_block

def block_process(data_block , process_method = 'default' , feature = [] , **kwargs):
    np.seterr(invalid='ignore')
    assert isinstance(data_block , DataBlock) , type(data_block)
    if process_method == 'default':
        process_method = 'order'
    if 'adj' in process_method and 'adjfactor' in data_block.feature:
        price_feat = np.intersect1d(['close', 'high', 'low', 'open', 'vwap'] , data_block.feature)
        ifeat = np.where(np.isin(data_block.feature,price_feat))[0]
        iadj  = np.where(data_block.feature == 'adjfactor')[0]
        data_block.values[...,ifeat] = np.multiply(data_block.values[...,ifeat],data_block.values[...,iadj])
        ifeat  = np.where(data_block.feature != 'adjfactor')[0]
        data_block.update(values = data_block.values[...,ifeat] , feature = data_block.feature[ifeat])
    if 'order' in process_method:
        raw_order = feature
        raw_order = [o for o in raw_order if o in data_block.feature]
        raw_order += [o for o in data_block.feature if o not in raw_order]
        ifeat = np.array([raw_order.index(f) for f in data_block.feature])
        data_block.update(values = data_block.values[...,ifeat] , feature = data_block.feature[ifeat])
    
    np.seterr(invalid='warn')
    return data_block

def block_hist_norm(data_block , key , save_path = None , 
                    start_dt = None , end_dt = 20161231 , 
                    step_day = 5 , tol = 1e-6 , **kwargs):
    if not key.startswith(('x_trade','trade','day','15m','min','30m','60m')): 
        return NotImplemented

    maxday = {
        'trade_day' : 60 ,
        'others'    : 1 ,
    }
    maxday = maxday[key] if key in maxday.keys() else maxday['others']

    date_slice = np.repeat(True , len(data_block.date))
    if start_dt is not None: date_slice[data_block.date < start_dt] = False
    if end_dt   is not None: date_slice[data_block.date > end_dt]   = False

    secid = data_block.secid
    date  = data_block.date
    feat  = data_block.feature
    inday = data_block.values.shape[2]

    len_step = len(date[date_slice]) // step_day
    len_bars = maxday * inday

    x = torch.tensor(data_block.values[:,date_slice])
    pad_array = (0,0,0,0,maxday,0,0,0)
    x = torch.nn.functional.pad(x , pad_array , value = np.nan)
    
    avg_x = torch.zeros(len_bars , len(feat))
    std_x = torch.zeros(len_bars , len(feat))

    x_endpoint = x.shape[1]-1 + step_day * np.arange(-len_step + 1 , 1)
    x_div = torch.ones(len(secid) , len_step , 1 , len(feat)).to(x)
    re_shape = (*x_div.shape[:2] , -1)
    if key == 'trade_day':
        # day : divide by endpoint
        x_div.copy_(x[:,x_endpoint,-1:])
    else:
        # Xmin day : price divide by preclose , other divide by day sum
        x_div.copy_(x[:,x_endpoint].sum(dim=2 , keepdim=True))
        price_feat = [f for f in ['close', 'high', 'low', 'open', 'vwap'] if f in feat]
        if len(price_feat) > 0:
            x_div[...,np.isin(feat , price_feat)] = x[:,x_endpoint-1,-1:][...,feat == price_feat[0]]

    nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
    nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
    for i in range(maxday):
        nan_sample += x[:,x_endpoint-i].reshape(*re_shape).isnan().any(dim=-1)

    for i in range(maxday):
        vijs = ((x[:,x_endpoint - maxday+1 + i]) / (x_div + tol))[nan_sample == 0]
        avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
        std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

    assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
    
    data = {'avg' : avg_x , 'std' : std_x}
    _save_dict_data(data , save_path)
    return data

def path_block_data(data_name , method = save_block_method , alias_search = False):
    if data_name.lower() == 'y': return f'{DIR_block}/Y.{method}'
    path = (DIR_block+'/X_{}.'+method)
    return _alias_search(path , data_name) if alias_search else path.format(data_name)
    
def path_norm_data(data_name , method = save_block_method , alias_search = False):
    if data_name.lower() == 'y': return f'{DIR_hist_norm}/Y.{method}'
    path = (DIR_hist_norm+'/X_{}.'+method)
    return _alias_search(path , data_name) if alias_search else path.format(data_name)

def _alias_search(path , key):
    alias_list = [key , f'trade_{key}' , key.replace('trade_','')]
    for alias in alias_list:
        if os.path.exists(path.format(alias)): 
            return path.format(alias)
    raise path.format(key)
    
    

