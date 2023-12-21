
import torch
import gc , random , math , time , copy
import numpy as np
from ..functional.func import *
from ..util.environ import get_config , cuda , DIR_data
from ..util.basic import versatile_storage
from torch.utils.data.dataset import IterableDataset , Dataset

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
    2. Datas: x_data , y_data , norm_param , index_stock , index_date
    3. Dataloader : yield x , y of training samples , create new ones if necessary
    """
    def __init__(self , model_data_type , config = None):
        self.CONFIG = get_config()
        if config is not None: self.CONFIG.update(config)
        storage_loader.activate(self.CONFIG['STORAGE_TYPE'])
        try:  
            self.precision = getattr(torch , self.CONFIG['PRECISION'])
        except:
            self.precision = torch.float
        self.model_data_type = model_data_type
        self.data_type_list  = self.model_data_type.split('+')
        self.x_data , self.y_data , self.norm_param , (self.index_stock , self.index_date) = load_trading_data(model_data_type , self.precision)
        self.stock_n , self.all_day_len = self.y_data.shape[:2]
        if isinstance(self.CONFIG['MODEL_PARAM']['num_output'],(list,tuple)):
            self.labels_n = min(self.y_data.shape[-1] , max(self.CONFIG['MODEL_PARAM']['num_output']))
        else:
            self.labels_n = min(self.y_data.shape[-1] , self.CONFIG['MODEL_PARAM']['num_output'])
        self.feat_dims = {mdt:v.shape[-1] for mdt,v in self.x_data.items()}

        _beg , _end , _int = self.CONFIG['BEG_DATE'] , self.CONFIG['END_DATE'] , self.CONFIG['INTERVAL']
        self.model_date_list = self.index_date[(self.index_date >= _beg) & (self.index_date <= _end)][::_int]
        self.test_full_dates = self.index_date[(self.index_date > _beg) & (self.index_date <= _end)]
        
        self.input_step = self.CONFIG['INPUT_STEP_DAY']
        self.test_step  = self.CONFIG['TEST_STEP_DAY']
        self.reset_dataloaders()

        self.buffer = {}
        self.buffer_functions()
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
        if self.buffer_proc is not None: self.buffer.update(self.buffer_proc(self))

        self.buffer = cuda(self.buffer)
        self.static_dataloader(x , y_step , w_step , index , self.nonnan_sample)

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
            x /= x.select(-2,-1).unsqueeze(-2) + 1e-4
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

def load_trading_data(model_data_type , dtype = torch.float):
    t0 = time.time()

    DIR_block  = f'{DIR_data}/block_data'
    data_type = get_config('data_type')['DATATYPE']
    path_yfunc = lambda x:f'{DIR_block}/Y.npz'
    path_xfunc = lambda x:f'{DIR_block}/X_{x}.npz'
    path_xnorm = f'{DIR_block}/X_normdict.pt'

    data_type_list = model_data_type.split('+')
    paths = {'y_data':path_yfunc(),**{mdt:path_xfunc(mdt) for mdt in data_type_list}}
    paths_factor = {mdt:path_xfunc(mdt) for mdt in data_type_list if mdt in data_type['factor']}
    data_dict = block_data_align(list(paths.values()) , list(paths_factor.values()))
    
    y_data = torch.tensor(data_dict[paths['y_data']]['values'])
    x_data = {torch.tensor(data_dict[paths[mdt]]['values']) for mdt in data_type_list}
    row = data_dict[paths['y_data']]['index']['secid']
    col = data_dict[paths['y_data']]['index']['date']
    
    # norm_param
    x_norm = {k:v for k,v in torch.load(path_xnorm).items()}

    # check
    assert all([d.shape[0] == y_data.shape[0] == len(row) for mdt,d in x_data.items()])
    assert all([d.shape[1] == y_data.shape[1] == len(col) for mdt,d in x_data.items()])
    
    x_data = _astype(x_data , dtype)
    y_data = _astype(y_data , dtype)
    x_norm = _astype(x_norm , dtype)
    return x_data , y_data , x_norm , (row , col)

def _astype(data , dtype):
    if isinstance(data , dict):
        return {k:_astype(v,dtype) for k,v in data.items()}
    elif isinstance(data , (list,tuple)):
        return type(data)([_astype(v,dtype) for v in data])
    else:
        return data.to(dtype)

def block_data_values(file_path):
    data = np.load(file_path)
    return data['values']

def block_data_index(file_path):
    data = np.load(file_path)
    return {k:data[k] for k in ['secid' , 'date' , 'feature']}

def block_data_align(paths , paths_forward_fillna = None , 
                     start_dt = None , end_dt = None , align_dim = True):
    """
    concatenate block data of paths , 
    secid : intersect of index(['secid']) of [*paths , *paths_forward_fillna]
    date : union of index(['date']) between start_dt and end_dt
    feature : union if concat
    paths_forard_fillna : datas not for everyday
    """
    paths = np.union1d(paths , paths_forward_fillna)
    indexes = {p:block_data_index(p) for p in paths}

    assert np.all([('secid' in v) for v in indexes.values()])
    assert np.all([('date' in v) for v in indexes.values()])
    assert np.all([('feature' in v) for v in indexes.values()])

    l_secid = [v['secid'] for v in indexes.values()]
    l_date  = [v['date']  for v in indexes.values()]
    max_dim = max([len(v) for v in indexes.values()])
    secid, pos_secid, pos_secid_old = index_intersect(l_secid)
    date , pos_date , pos_date_old  = index_union(l_date, start_dt , end_dt)
    
    data_dict = {}
    for i,(p,old_index) in enumerate(indexes.items()):
        old_values = block_data_values(p)
        assert old_values.shape[:2] == (len(old_index['secid']) , len(old_index['date']))
        assert old_values.shape[-1] == len(old_index['feature'])
        new_values = np.full((len(secid),len(date),*old_values.shape[2:]), np.nan)

        tmp = new_values[pos_secid[i]]
        tmp[:,pos_date[i]] = old_values[pos_secid_old[i]][:,pos_date_old[i]]
        new_values[pos_secid[i]] = tmp
        new_index = {'secid':secid,'date':date}
        if p in paths_forward_fillna: new_values = forward_fillna(new_values,axis=1)
        if align_dim and (len(new_values.shape) < max_dim):
            assert (len(new_values.shape),max_dim)==(3,4), (len(new_values.shape),max_dim)
            new_values = new_values.reshape(*new_values.shape[:2],1,-1)
            new_index.update({'feature':old_index['feature']})
        else:
            [new_index.update({k:v}) for k,v in old_values if k not in ['secid','date']]
        data_dict.update({p:{'values':new_values,'index':new_index}})
    return data_dict

def merge_data1d(data_dict , to_data_block = True):
    if len(data_dict) == 0: return None
    secid , p_s0 , p_s1 = index_union([data.secid for data in data_dict.values()])
    date    = np.array(list(data_dict.keys())).astype(int)
    feature , _ , p_f1 = index_intersect([data.feature for data in data_dict.values()])
    newdata = np.full((len(secid),len(date),len(feature)) , np.nan , dtype = float)
    for i,(k,v) in enumerate(data_dict.items()):
        newdata[p_s0[i],i,:] = v.values[p_s1[i]][:,p_f1[i]]
    if to_data_block:
        return DataBlock(newdata , secid , date , feature)
    else:
        return newdata , (secid , date , feature)

def merge_block(blocks):
    if len(blocks) == 0: return None
    if len(blocks) == 1: return blocks[0]
    if isinstance(blocks[0] , DataBlock): 
        return DataBlock.merge_others(blocks)
    
    values = [b[0] for b in blocks]
    secid , p_s0 , p_s1 = index_union([b[1][0] for b in blocks])
    date  , p_d0 , p_d1 = index_union([b[1][1] for b in blocks])
    l1 = len(np.unique(np.concatenate([b[1][2] for b in blocks])))
    l2 = sum([len(b[1][2]) for b in blocks])
    distinct_feature = (l1 == l2)

    for i , data in enumerate(values):
        newdata = np.full((len(secid),len(date),*data.shape[2:]),np.nan)
        tmp = newdata[p_s0[i]]
        tmp[:,p_d0[i]] = data[p_s1[i]][:,p_d1[i]]
        newdata[p_s0[i]] = tmp
        values[i] = newdata

    if distinct_feature:
        feature = np.concatenate([b[1][2] for b in blocks])
        newdata = np.concatenate(values , axis = -1)
    else:
        feature, p_f0 , p_f1 = index_union([b[1][2] for b in blocks])
        newdata = np.full((*newdata[0].shape[:-1],len(feature)) , np.nan , dtype = float)
        for i , data in enumerate(values):
            newdata[...,p_f0[i]] = data[...,p_f1[i]]
    return newdata , (secid , date , feature)

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

    def from_dtank(self , dtank , file , start_dt = None , end_dt = None , feature = None , **kwargs):
        date = np.array(list(dtank.get_object(file).keys())).astype(int)
        if start_dt is not None: date = date[date >= start_dt]
        if end_dt   is not None: date = date[date <= end_dt]
        if len(date) == 0: return NotImplemented
        datas = {str(d):dtank.read_data1D([file,str(d)],feature).to_kline() for d in date}
        secid , p_s0 , p_s1 = index_union([data.secid for data in datas.values()])
        date    = np.array(list(datas.keys())).astype(int)
        feature , _ , p_f1 = index_intersect([data.feature for data in datas.values()])
        new_shape = [len(secid),len(date),len(feature)]
        if datas[str(date[0])].values.ndim == 3:
            new_shape.insert(2 , datas[str(date[0])].values.shape[1])
        newdata = np.full(tuple(new_shape) , np.nan , dtype = float)
        for i,(k,v) in enumerate(datas.items()):
            newdata[p_s0[i],i,:] = v.values[p_s1[i]][...,p_f1[i]]
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
    
    def save_npz(self , file_path):
        index_vals = {k:getattr(self,k) for k in ['secid' , 'date' , 'feature']}
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        np.savez_compressed(file_path , values = self.values , **index_vals)
    
    def read_npz(self , file_path):
        data = np.load(file_path)
        index  = {k:data[k] for k in ['secid' , 'date' , 'feature']}
        self._init_attr(data['values'] , **index)
        return self
    
    def read_multiple_npz(self , file_paths , 
                          forward_fillna  = 'guess' , 
                          intersect_secid = True ,
                          union_date = True , 
                          start_dt = None , end_dt = None):
        _guess = lambda ls,excl:[os.path.basename(x).lower().startswith(excl) == 0 for x in ls]
        if forward_fillna == 'guess':
            exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m')
            forward_fillna = np.array(_guess(file_paths , exclude_list))
        elif forward_fillna is None or isinstance(forward_fillna , bool):
            forward_fillna = np.repeat(forward_fillna , len(file_paths))
        else:
            assert len(file_paths) == len(forward_fillna) , (len(file_paths) , len(forward_fillna))

        portals = [np.load(p) for p in file_paths]
        indexes = [block_data_index(p) for p in file_paths]
        secid = date = None
        if intersect_secid: secid,p_s0,p_s1 = index_intersect([idx['secid'] for idx in indexes])
        if union_date: date,p_d0,p_d1 = index_union([idx['date'] for idx in indexes] , start_dt , end_dt)

        block_dict = {}
        for i , (portal , idx) in enumerate(zip(portals , indexes)):
            values = portal['values']
            if secid is not None: idx['secid'] = secid
            if date  is not None: idx['date']  = date
            if values.shape[:2] != (len(idx['secid']),len(idx['date'])): # no secid/date alter
                values = np.full((len(idx['secid']),len(idx['date']),*values.shape[2:]) , np.nan)
                if secid is None:
                    values[:,p_d0[i]] = portal['values'][:,p_d1[i]]
                elif date is None:
                    values[p_s0[i]] = portal['values'][p_s1[i]]
                else:
                    tmp = values[p_s0[i]]
                    tmp[:,p_d0[i]] = portal['values'][p_s1[i]][:,p_d1[i]]
                    values[p_s0[i]] = tmp

            date_slice = np.repeat(True , len(idx['date']))
            if start_dt is not None: date_slice[idx['date'] < start_dt] = False
            if end_dt   is not None: date_slice[idx['date'] > end_dt]   = False
            values , idx['date'] = values[:,date_slice] , idx['date'][date_slice]

            if forward_fillna[i]: values = forward_fillna(values , axis = 1)

            block_dict.update({file_paths[i]:DataBlock(values , **idx)})
        return block_dict
