import h5py
import pandas as pd
import numpy as np
import random, string , os , time
import inspect , traceback

def index_intersect(idxs , min_value = None , max_value = None):
    new_idx = None
    for idx in idxs:
        if new_idx is None or idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.intersect1d(new_idx , idx)
    if min_value is not None: new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: new_idx = new_idx[new_idx <= max_value]
    new_idx = np.sort(new_idx)
    inter   = [None if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple([None if v is None else v[1] for v in inter])
    pos_old = tuple([None if v is None else v[2] for v in inter])
    return new_idx , pos_new , pos_old

def index_union(idxs , min_value = None , max_value = None):
    new_idx = None
    for idx in idxs:
        if new_idx is None or idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    if min_value is not None: new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: new_idx = new_idx[new_idx <= max_value]
    inter   = [None if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple([None if v is None else v[1] for v in inter])
    pos_old = tuple([None if v is None else v[2] for v in inter])
    return new_idx , pos_new , pos_old

def _update_rcv(old_rowcolval , upd_rowcolval , replace = False , all_within = False , all_without = False):
    old_row , old_col , old_val = old_rowcolval
    upd_row , upd_col , upd_val = upd_rowcolval
    if upd_row is None:  upd_row = old_row
    if upd_col is None:  upd_col = old_col
    assert old_val.shape == (len(old_row) , len(old_col))
    assert upd_val.shape == (len(upd_row) , len(upd_col))
    if replace: return upd_row , upd_col , upd_val

    in_row , in_col = np.isin(upd_row , old_row) , np.isin(upd_col , old_col)
    if all_within:  assert all(all(in_row) , all(in_col))
    if all_without: assert not any(any(in_row) , any(in_col))

    new_row, pos_row, _ = index_union([old_row , upd_row])  
    new_col, pos_col, _ = index_union([old_col , upd_col]) 
    new_val = np.full((len(new_row),len(new_col)) , np.nan)
    for r, c, v in zip(pos_row , pos_col , [old_val , upd_val]): 
        if r is not None and c is not None: new_val[r][:,c] = v[:]
    assert new_val.shape == (len(new_row) , len(new_col))
    return new_row, new_col, new_val

class Data1D():
    # Sec-Feature
    def __init__(self , secid = None , feature = None , values = None , src = None) -> None:
        if isinstance(src , (dict,h5py.Group)):
            secid , feature , values = src['secid'][:] , src['feature'][:].astype(str) , src['values'][:]
        elif isinstance(src , pd.DataFrame):
            if 'secid' in src.index.names: src = src.reset_index()
            secid = src['secid'] 
            feature = src.columns.values[np.isin(src.columns.values, ['secid']) == 0]
            values = src.loc[:,feature].values
        self.init_attr(secid , feature , values)

    def __repr__(self):
        return '\n'.join([
            str(self.__class__) ,
            f'secid len ({len(self.secid)}): {self.secid.__repr__()}' ,
            f'feature len ({len(self.feature)}): {self.feature.__repr__()}' ,
            f'values shape {self.values.shape}'
        ])
    
    def __eq__(self , other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return all([np.array_equal(getattr(self,obj),getattr(other,obj)) for obj in ['secid','feature','values']])

    def init_attr(self , secid , feature , values):
        assert values.shape == (len(secid) , len(feature))
        self.secid, self.feature, self.values = np.array(secid), np.array(feature), np.array(values)

    def replace_values(self , values): 
        old_rcv = (self.secid,self.feature,self.values)
        upd_rcv = (None,None,values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,replace=True))

    def alter_feature(self , new_feature , new_values): 
        old_rcv = (self.secid,self.feature,self.values)
        upd_rcv = (None,new_feature,new_values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,all_within=True))

    def add_feature(self , new_feature , new_values): 
        old_rcv = (self.secid,self.feature,self.values)
        upd_rcv = (None,new_feature,new_values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,all_without=True))

    def slice(self , secid = None , feature = None):
        if secid is not None:
            self.secid = self.secid[secid]
            self.values = self.values[secid]
        if feature is not None:
            self.feature = self.feature[feature]
            self.values = self.values[:,feature]
        return self
    
    def to_dataframe(self):
        return pd.DataFrame(self.values,index=pd.Index(self.secid,name='secid'),columns=self.feature)
    
    def to_kline(self):
        usecid = np.unique(self.secid)
        if len(usecid) != len(self.secid): 
            lookfor = ['minute' , 'inday']
            assert any(np.isin(lookfor , self.feature)) , self.feature
            inday_idx = np.isin(self.feature , lookfor)
            inday  = self.values[:,inday_idx][:,0].astype(int)
            values = self.values[:,inday_idx == 0].astype(int)
            lsecid , linday = len(np.unique(self.secid)) , len(np.unique(inday))
            feature = self.feature[inday_idx == 0]
            if len(values) == lsecid * linday:
                values = values.reshape(lsecid , linday , -1)
                secid  = self.secid[::linday]
            else:
                new_values = np.zeros((lsecid , linday , len(feature)))
                secid  = np.array(sorted(np.unique(self.secid)))
                for mm in sorted(np.unique(inday)):
                    msec , mpos = self.secid[inday == mm] , inday == mm
                    if np.array_equal(secid , msec):
                        new_values[:,mm] = self.values[mpos]
                    else:
                        i_sec = np.intersect1d(secid , msec , return_indices=True)[1]
                        new_values[i_sec,mm] = values[mpos , 1:]
                values = new_values
            values = self._kline_fillna(values , feature)
            return Data1DKline(secid , feature, values)
        else:
            return Data1DKline(self.secid , self.feature, self.values[:,None])

    def _kline_fillna(self , kline , feature):
        if np.isnan(kline).sum() == 0: return kline
        for i , f in enumerate(feature):
            if f == ['close','open','high','low','vwap']:
                arr = kline[:,:,i]
                mask = np.isnan(arr)
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                idx = np.maximum.accumulate(idx, axis=1, out=idx)
                out = arr[np.arange(idx.shape[0])[:,None], idx]
                kline[:,:,i] = out
            else:
                kline[:,:,i] = np.nan_to_num(kline[:,:,i])
        return kline
    
class Data1DKline():
    # Sec-Feature
    def __init__(self , secid = None , feature = None , values = None) -> None:
        self.init_attr(secid , feature , values)

    def __repr__(self):
        return '\n'.join([
            str(self.__class__) ,
            f'secid len ({len(self.secid)}): {self.secid.__repr__()}' ,
            f'feature len ({len(self.feature)}): {self.feature.__repr__()}' ,
            f'values shape {self.values.shape}'
        ])
    
    def __eq__(self , other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return all([np.array_equal(getattr(self,obj),getattr(other,obj)) for obj in ['secid','feature','values']])

    def init_attr(self , secid , feature , values):
        assert values.ndim == 3 , values.shape
        assert values.shape[0]  == len(secid)  , (values.shape , len(secid))
        assert values.shape[-1] == len(feature), (values.shape , len(feature))
        self.secid, self.feature, self.values = np.array(secid), np.array(feature), np.array(values)


class DataDSF():
    # Date-Sec-Feature
    def __init__(self , date = None , datas = None , src = None , rand_new = False) -> None:
        if rand_new:
            date  = np.random.randint(0,10) + np.arange(np.random.randint(10,15))
            _secid = np.random.randint(0,10) + np.arange(np.random.randint(15,20))
            _feature = [f'value{i}' for i in range(np.random.randint(2,5))]
            datas = [Data1D(_secid , _feature , np.random.randn(len(_secid) , len(_feature))) for d in date]
        elif isinstance(src , (dict , h5py.Group)):
            if np.isin(['__date__','__secid__','__feature__','__values__'],list(src.keys())).all():
                date = src['__date__']
                datas = [{
                    'secid' : src['__secid__'] ,
                    'feature' : src['__feature__'] ,
                    'values' : src['__values__'][i]
                } for i,d in enumerate(date)]
            else:
                date , datas = list(src.keys()) , src.values()
        self.init_attr(date , datas)

    def __repr__(self):
        d0 = self.data[str(self.date[0])]
        return '\n'.join([
            str(self.__class__) ,
            f'date len ({len(self.date)}): {self.date.__repr__()}' , 
            f'secid[0] len ({len(d0.secid)}): {d0.secid.__repr__()}' ,
            f'feature[0] len ({len(d0.feature)}): {d0.feature.__repr__()}' ,
            f'values[0] shape {d0.values.shape}'
        ])
    
    def __eq__(self , other):
        if self.__class__ != other.__class__:
            return NotImplemented
        if not np.array_equal(self.date , other.date):
            return False
        for k in self.date:
            if self.data[str(k)] != other.data[str(k)]: return False
        return True

    def _data1d(self , data):
        return data if isinstance(data , Data1D) else Data1D(src=data)

    def init_attr(self , date , datas):
        assert len(date) == len(datas)
        self.date = date
        self.data = {str(k):self._data1d(d) for k,d in zip(date , datas)}
    
    def add_date(self , date , data):
        assert date not in self.date , date
        self.date = np.concatenate([self.date , [date]])
        self.data[str(date)] = self._data1d(data)

    def replace_date(self , date , data):
        assert date in self.date , date
        self.data[str(date)] = self._data1d(data)

    def to_FDS(self):
        date = self.date
        secid, pos_secid, _ = index_union([d.secid for d in self.data.values()])
        feat , pos_feat , _ = index_union([d.feature for d in self.data.values()])
        datas = self.data.values()
        all_values = np.full((len(date),len(secid),len(feat)) , np.nan)
        for i_date , (data , i_sec , i_feat) in enumerate(zip(datas , pos_secid , pos_feat)):
            tmp = all_values[i_date,i_sec,:]
            tmp[:,i_feat] = data.values[:]
            all_values[i_date,i_sec,:] = tmp
        return DataFDS(feat , [Data1F(date,secid,all_values[:,:,i]) for i in range(len(feat))])

class Data1F():
    # Date-Sec
    def __init__(self , date = None , secid = None , values = None , src = None) -> None:
        if isinstance(src , (dict,h5py.Group)):
            date , secid , values = src['date'][:] , src['secid'][:] , src['values'][:]
        self.init_attr(date , secid , values)

    def __repr__(self):
        return '\n'.join([
            str(self.__class__) ,
            f'date len ({len(self.date)}): {self.date.__repr__()}' ,
            f'secid len ({len(self.secid)}): {self.secid.__repr__()}' ,
            f'values shape {self.values.shape}'
        ])
    
    def __eq__(self , other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return all([np.array_equal(getattr(self,obj),getattr(other,obj)) for obj in ['date','secid','values']])

    def init_attr(self , date , secid , values):
        assert values.shape == (len(date) , len(secid))
        self.date, self.secid, self.values = np.array(date), np.array(secid), np.array(values)

    def replace_values(self , values): 
        old_rcv = (self.date,self.secid,self.values)
        upd_rcv = (None,None,values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,replace=True))

    def alter_secid(self , new_secid , new_values): 
        old_rcv = (self.date,self.secid,self.values)
        upd_rcv = (None,new_secid,new_values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,all_within=True))

    def add_secid(self , new_secid , new_values): 
        old_rcv = (self.date,self.secid,self.values)
        upd_rcv = (None,new_secid,new_values)
        self.init_attr(*_update_rcv(old_rcv,upd_rcv,all_without=True))

class DataFDS():
    # Feature-Date-Sec
    def __init__(self , feature = None , datas = None , src = None , rand_new = False) -> None:
        if rand_new:
            feature = [f'value{i}' for i in range(np.random.randint(2,5))]
            _date  = np.random.randint(0,10) + np.arange(np.random.randint(10,15))
            _secid = np.random.randint(0,10) + np.arange(np.random.randint(15,20))
            datas = [Data1F(_date , _secid , np.random.randn(len(_date) , len(_secid))) for f in feature]
        elif isinstance(src , (dict , h5py.Group)):
            if np.isin(['__date__','__secid__','__feature__','__values__'],list(src.keys())).all():
                feature = src['__feature__']
                datas = [{
                    'date' : src['__date__'] ,
                    'secid' : src['__secid__'] ,
                    'values' : src['__values__'][:,:,i]
                } for i,f in enumerate(feature)]
            else:
                feature , datas = list(src.keys()) , src.values()
        self.init_attr(feature , datas)

    def __repr__(self):
        d0 = self.data[str(self.feature[0])]
        return '\n'.join([
            str(self.__class__) ,
            f'feature len ({len(self.feature)}): {self.feature.__repr__()}' , 
            f'date[0] len ({len(d0.date)}): {d0.date.__repr__()}' ,
            f'secid[0] len ({len(d0.secid)}): {d0.secid.__repr__()}' ,
            f'values[0] shape {d0.values.shape}'
        ])
    
    def __eq__(self , other):
        if self.__class__ != other.__class__:
            return NotImplemented
        if not np.array_equal(self.feature , other.feature):
            return False
        for k in self.feature:
            if self.data[str(k)] != other.data[str(k)]: return False
        return True

    def _data1f(self , data):
        return data if isinstance(data , Data1F) else Data1F(src=data)

    def init_attr(self , feature , datas):
        assert len(feature) == len(datas)
        self.feature = feature
        self.data = {str(k):self._data1f(d) for k,d in zip(feature , datas)}
    
    def add_feature(self , feature , data):
        assert feature not in self.feature , feature
        self.feature = np.concatenate([self.feature , [feature]])
        self.data[str(feature)] = self._data1f(data)

    def replace_date(self , feature , data):
        assert feature in self.feature , feature
        self.data[str(feature)] = self._data1f(data)

    def to_DSF(self):
        date , pos_date ,_ = index_union([d.date for d in self.data.values()])
        secid, pos_secid,_ = index_union([d.secid for d in self.data.values()])
        feat  = self.feature
        datas = self.data.values()
        all_values = np.full((len(date) , len(secid),len(feat)) , np.nan)
        for i_feat , (data , i_date , i_sec) in enumerate(zip(datas , pos_date , pos_secid)):
            tmp = all_values[i_date,:,i_feat]
            tmp[:,i_sec] = data.values[:]
            all_values[i_date,:,i_feat] = tmp
        return DataDSF(date , [Data1D(secid,feat,all_values[i]) for i in range(len(date))])

class DataTank():
    def __init__(self , filename = None , mode = 'guess' , open = True , compress = False) -> None:
        self.filename = filename
        self.mode = mode
        self.file = None
        if open: self.open(mode = mode)
        self.compress = compress
        self.str_dtype = h5py.string_dtype(encoding = 'utf-8')

    def __repr__(self):
        return f'{self.__class__} : {self.filename} -- ({"Opened" if self.isopen() else "Closed"})'

    def __enter__(self):
        if not self.isopen(): self.open()
        return self
    
    def __exit__(self , exc_type, exc_value, traceback):
        self.close()

    def _check_read_mode(self):
        return self.open() if not self.isopen() or self.file.mode not in ['r' ,'r+'] else self.file
    
    def _check_write_mode(self):
        return self.open() if not self.isopen() or self.file.mode not in ['r+'] else self.file
    
    def _full_path(self , path):
        return '/'.join([str(f) for f in path]) if isinstance(path , (list,tuple)) else path
    
    def keys(self):
        return self.file.keys() if self.file is not None else None

    def change_file(self , filename):
        self.filename = filename
        if self.isopen(): self.open(self.file.mode)

    def open(self , mode = None): 
        if self.filename is None: return NotImplemented
        if mode is None: mode = self.mode
        if mode == 'guess': 
            mode = 'r+' if os.path.exists(self.filename) else 'w'
        if not os.path.exists(self.filename):
            h5py.File(self.filename , 'w' , fs_strategy='fsm' , fs_persist = False).close()
        self.file = h5py.File(self.filename , 'r+' if mode == 'w' else mode)
        return self.file
    
    def close(self): 
        if self.isopen():
            self.file.close()
            self.file = None

    def isopen(self):
        return self.file is not None and isinstance(self.file , h5py.File)

    def reopen(self):
        self.close()
        self.open(mode = 'r+' if self.mode == 'w' else self.mode)

    def tree(self , object = None , leaf_method = None):
        if object is None: object = self._check_read_mode()
        if isinstance(object , (h5py.Group,h5py.File)):
            return {key:self.tree(val , leaf_method) for key,val in object.items()}
        else:
            if leaf_method is None:
                return -1
            else:
                x = getattr(object , leaf_method)
                return x() if inspect.ismethod(x) else x

    def print_tree(self, object = None , print_pre='' , depth = 3 , width = 4 , depth_full_width = 2 , num_tail_keys = 3 , full_tree = False):
        if object is None: object = self._check_read_mode()
        if not full_tree and len(object.items()) > width and depth_full_width <= 0:
            show_pre , show_pro = width - (width // 2) , len(object.items()) - width // 2
        else:
            show_pre , show_pro = len(object.items()) + 1 , len(object.items())
        keys = list(object.keys())
        keys = keys[:show_pre] + keys[show_pro:]
        inter = '├──'
        afpre = '│  '
        for i , key in enumerate(keys):
            if i == show_pre: print(f'{print_pre}{inter} ...... omitting {len(object)-width} memebers')
            if i == len(keys) - 1: 
                inter = '└──'
                afpre = '   '                
            obj = object.get(key)
            print(f'{print_pre}{inter} {key}' , self.repr_object(obj))
            if full_tree or (depth > 1 and (not self.end_of_leaf(obj))):
                self.print_tree(obj,f'{print_pre}{afpre} ',depth-1,width,depth_full_width-1,num_tail_keys,full_tree)

    def get_object(self , path):
        path = self._full_path(path)
        return self.file.get(path)
    
    def repr_object(self , obj , num_tail_keys = 3):
        str_list = [obj.__repr__()]
        if isinstance(obj , h5py.Group) and self.end_of_leaf(obj):
            _tailkeys   = [str(k) for k in list(obj.keys())[:num_tail_keys]]
            _n_omitkeys = len(obj.keys()) - num_tail_keys
            str_list.append('('+', '.join(_tailkeys)+('' if _n_omitkeys <= 0 else f' and {_n_omitkeys} more')+')')
        else:
            if len(obj) > 50: 
                v = sorted(list(obj.keys()))
                str_list.append('-'.join([v[0] , v[-1]]))
        return ' '.join(str_list)
    
    def del_object(self , path , ask = True):
        path = self._full_path(path)
        if self.file.get(path) is None: return
        if ask and input('press yes to confirm') != 'yes': return
        try:
            del self.file[path]
            return True
        except:
            g = self.file[path]
            for key in g.keys(): del g[key]
            return False

    def delete_all(self , allow = False):
        if not allow: return NotImplemented
        rand_str = 'yes'
        if input(f'press {rand_str} to confirm') != rand_str: return
        rand_num = np.random.randint(100,999)
        if input(f'press {rand_num} to confirm') != str(rand_num): return
        rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        if input(f'press {rand_str} to confirm') != rand_str: return
        [self.del_object(k , ask = False) for k in self.file.keys()]
    
    def create_object(self , path , data = None , dtype = None , overwrite = False , compress = None , **kwargs):
        assert self.file.mode in ['r+'] , self.file.mode
        path = self._full_path(path)
        if self.get_object(path) is not None:
            if overwrite:
                self.del_object(path , ask = False)
            else:
                raise Exception(f'{path} already exists!')
        if data is None:
            return self.file.create_group(path)
        else:
            compression = 'gzip' if compress or self.compress else None
            return self.file.create_dataset(path , data = data , dtype = dtype , compression = compression)

    def is_Data1D(self , obj):
        if not np.isin(['secid','feature','values'],list(obj.keys())).all(): return False
        return obj['values'].shape == (len(obj['secid']) , len(obj['feature']))

    def is_DataFrame(self , obj):
        return np.isin(['__columns__','__columns_dtype__'],list(obj.attrs.keys())).all()
    
    def is_compress(self , obj):
        if isinstance(obj , h5py.Group):
            if len(obj) > 0:
                return obj[list(obj.keys())[0]].compression is not None
            else:
                return False
        else:
            return obj.compression is not None

    def read_guess(self , path , feature = None):
        path = self._full_path(path)
        if self.is_DataFrame(self.get_object(path)):
            return self.read_dataframe(path , feature = feature)
        else:
            return self.read_data1D(path , feature = feature)
        
    def write_guess(self , path , data , overwrite = False , compress = None):
        if isinstance(data , pd.DataFrame):
            self.write_dataframe(path , data = data , overwrite = overwrite , compress = compress)
        elif isinstance(data , Data1D):
            self.write_data1D(path , data = data , overwrite = overwrite , compress = compress)
        else:
            raise Exception(TypeError)

    def write_dataframe(self , path , data , overwrite = False , compress = None):
        assert self.file.mode in ['r+'] , self.file.mode
        if not isinstance(data , pd.DataFrame): data = pd.DataFrame(data)
        path = self._full_path(path)
        self.del_object(path , ask = not overwrite)
        columns , columns_dtype = data.columns.values.tolist() , data.dtypes.values.astype(str).tolist()
        save_dtype = [self.str_dtype if dtype == 'object' else None for dtype in columns_dtype]
        for col , dtype in zip(columns , save_dtype):
            self.create_object([path, col], data=data.loc[:,col], dtype=dtype, compress = compress)
        self.set_group_attrs(path , __columns__ = columns)
        self.set_group_attrs(path , __columns_dtype__ = columns_dtype)
        
    def read_dataframe(self , path , feature = None):
        path = self._full_path(path)
        data = self.get_object(path)
        assert data is not None, path
        columns = self.get_group_attrs(path , '__columns__') # self.get_object([path , '__columns__'])[:].astype(str)
        col_dtype = self.get_group_attrs(path , '__columns_dtype__') # self.get_object([path , '__columns_dtype__'])[:].astype(str)
        if feature is not None: assert all(np.isin(feature , columns)) , np.setdiff1d(feature , columns)
        df = pd.DataFrame()
        for col , dtype in zip(columns , col_dtype): 
            if (feature is not None) and (col not in feature): continue
            if dtype == 'object':
                df[col] = np.char.decode(data[col][:].astype(bytes) , 'utf-8')
            else:
                df[col] = data[col][:].astype(getattr(np , dtype))  
        if feature is not None: df = df.loc[:,feature]          
        return df

    def write_data1D(self , path , data , overwrite = False , compress = None):
        assert self.file.mode in ['r+'] , self.file.mode
        if not isinstance(data , Data1D): data = Data1D(src=data)
        path = self._full_path(path)
        self.del_object(path , ask = not overwrite)
        for key in ['secid','feature']:
            dtype = self.str_dtype if isinstance(getattr(data , key)[0] , str) else None
            self.create_object([path , key] , data = list(getattr(data , key)) , dtype = dtype , compress = compress)
        self.create_object([path , 'values'] , data = data.values , compress = compress)

    def read_data1D(self , path , feature = None , none_if_incomplete = False):
        portal = self.get_object(path)
        if none_if_incomplete:
            if not all(np.isin(['secid','feature','values'] , list(portal.keys()))): return None
        if feature is None:
            return Data1D(src = portal)
        else:
            portal = self.get_object(path)
            all_feature = portal['feature'][:].astype(str)
            if feature is None: feature = all_feature
            if isinstance(feature , str): feature = [feature]
            ifeat = np.array([np.where(all_feature == f)[0][0] for f in feature])
            ifeat.sort()
            secid = portal['secid'][:]
            feature = all_feature[ifeat]
            values = portal['values'][:,ifeat]
            return Data1D(secid , feature , values)
    
    def write_dataDSF(self , path , data , overwrite = True):
        assert self.file.mode in ['r+'] , self.file.mode
        # assert isinstance(data , DataDSF) , data.__class__
        path = self._full_path(path)
        for k , v in data.data.items():
            self.write_data1D([path , str(k)] , v , overwrite = overwrite)

    def read_dataDSF(self , path , start = None, end = None):
        __start_time__ = time.time()
        if start is None: start = -1
        if end is None: end = 99999999
        portal = self.get_object(path)
        date = np.array(list(portal.keys())).astype(int)
        date = date[(date >= start) & (date <= end)]
        result = DataDSF(src={str(k):self.read_data1D([path,str(k)]) for k in date})
        print(f'loading: {time.time() - __start_time__:.2f} secs')
        return result

    def read_dataFDS(self , path , start = None , end = None , dict_only = True) -> None:
        __start_time__ = time.time()
        if start is None: start = -1
        if end is None: end = 99999999
        portal = self.get_object(path)
        date = np.array(list(portal.keys())).astype(int)
        date = date[(date >= start) & (date <= end)]
        secid, pos_secid, _ = index_union([portal[str(d)]['secid'][:] for d in date])
        feat , pos_feat , _ = index_union([portal[str(d)]['feature'][:] for d in date])
        feat = feat.astype(str)
        datas = [portal[str(d)]['values'] for d in date]
        all_values = np.full((len(date) , len(secid), len(feat)) , np.nan)
        for i_date , (data , i_sec , i_feat) in enumerate(zip(datas , pos_secid , pos_feat)):
            tmp = all_values[i_date,i_sec,:]
            tmp[:,i_feat] = data[:]
            all_values[i_date,i_sec,:] = tmp
        print(f'loading: {time.time() - __start_time__:.2f} secs')
        if dict_only : 
            return {
                '__feature__' : feat ,
                '__date__'    : date ,
                '__secid__'   : secid ,
                '__values__'  : all_values
            }
        else:
            __start_time__ = time.time()
            result = DataFDS(feat , [Data1F(date,secid,all_values[:,:,i]) for i in range(len(feat))])
            print(f'Transform: {time.time() - __start_time__:.2f} secs')
            return result
    
    def get_group_attrs(self , path , attr = ['__information__' , '__create_time__','__last_date__', '__update_time__']):
        g = self.get_object(path)
        assert g is not None, path
        return g.attrs.get(attr) if isinstance(attr , str) else {a:g.attrs.get(a) for a in attr}

    def set_group_attrs(self , path , overwrite = True , **kwargs):
        if len(kwargs) == 0: return NotImplemented
        g = self.get_object(path)
        assert g is not None, path
        attr = {k:v for k,v in kwargs.items() if v is not None}
        if not overwrite:
            exist_attrs = g.attrs.__dir__()
            attr = {k:v for k,v in attr.items() if v not in exist_attrs}
        [g.attrs.modify(k , v) for k,v in attr.items() if v is not None]
    
    def repack(self , target_path = None , compress = None):
        # repack h5 file to avoid aggregating file size to unlimited amount
        # if target_path is None , repack to self
        is_open = self.isopen()
        self.close()
        if target_path is None: target_path = self.filename
        self.filename = repack_DataTank(self.filename , target_path , compress = compress)
        if is_open: self.reopen()

    def end_of_leaf(self , object = None):
        if object is None: object = self.file
        return isinstance(object , h5py.Dataset) or all([isinstance(object[key] , h5py.Dataset) for key in object.keys()])
    
    def leaf_list(self , object = None , call_list = []):
        if object is None: object = self.file
        if isinstance(call_list , str): call_list = list(call_list)
        assert all(np.isin(call_list , ['is_DataFrame' , 'is_Data1D' , 'is_compress']))
        leafs = []
        for key in object.keys():
            obj = object[key]
            if self.end_of_leaf(obj):
                leaf_char = {'path' :obj.name , 'key'  :key}
                for _call in call_list: leaf_char[_call] = getattr(self , _call)(obj)
                leafs.append(leaf_char)
            else:
                leafs = np.concatenate([leafs , self.leaf_list(obj , call_list)])
        return leafs
    
def tree_diff(tree1 , tree2):
    keys1 = list(tree1.keys()) if isinstance(tree1 , dict) else []
    keys2 = list(tree2.keys()) if isinstance(tree2 , dict) else []
    keys_diff = np.setdiff1d(keys1 , keys2)
    keys_share = np.intersect1d(keys1 , keys2)
    result = {key:tree1[key] for key in keys_diff}
    for key in keys_share:
        result.update(tree_diff(tree1[key] , tree2[key]))
    return result

def copy_tree(source , source_path ,  target , target_path , compress = None, 
              print_process = True , print_time = True , print_pre = '' ,
              **kwargs):
    if not isinstance(source , DataTank): return NotImplemented
    assert isinstance(target , DataTank) , type(target)
    assert isinstance(source_path , str) and isinstance(target_path , str) , (source_path , target_path)

    portal = source.get_object(source_path)
    if source.end_of_leaf(portal):
        source_compress = source.is_compress(portal)
        data = source.read_guess(source_path)
        target.write_guess(target_path , data , overwrite = True , compress = source_compress)
    elif isinstance(portal , h5py.Dataset):
        print('Not here!!!')
        data , dtype , source_compress = portal[:] , portal.dtype , portal.compression is not None or compress
        target.create_object(target_path,data=data,dtype=dtype,overwrite=True,compress=source_compress)
        return
    else:
        for key in source.get_object(source_path).keys():
            key_source_path = '/'.join([source_path , key])
            key_target_path = '/'.join([target_path , key])
            copy_tree(source , key_source_path ,  target , key_target_path , compress = compress , 
                      print_process = print_process , print_time = print_time , print_pre = print_pre ,
                      **kwargs)
    attrs = {k:v for k,v in source.get_group_attrs(source_path).items() if v is not None}
    target.set_group_attrs(target_path , overwrite = True , **attrs) 
    if len(attrs) > 0:
        target.set_group_attrs(target_path , overwrite = True , **attrs) 
        if print_process: 
            print_msg = f'{print_pre}{os.path.relpath(target.filename)}/{target_path} copying Done!'
            print(f'{time.ctime()} : {print_msg}' if print_time else print_msg)

def repack_DataTank(source_tank_path , target_tank_path = None):
    # repack h5 file to avoid aggregating file size to unlimited amount
    # if target_tank_path is None , create a new file name to repack to
    # if target_tank_path == source_tank_path , repack and replace source_tank_path
    if target_tank_path is None:
        target_tank_path = file_shadow_name(source_tank_path , 'new')

    if source_tank_path == target_tank_path:
        new_target_tank_path = file_shadow_name(source_tank_path , 'shadow')
        repack_DataTank(source_tank_path , new_target_tank_path)
        os.unlink(source_tank_path)
        os.rename(new_target_tank_path , source_tank_path)
    else:
        assert not os.path.exists(target_tank_path) , target_tank_path
        if input(f'Repack to {target_tank_path} , press yes to confirm') != 'yes': return
        source_dtank = DataTank(source_tank_path , 'r') 
        target_dtank = DataTank(target_tank_path , 'w')
        try:
            copy_tree(source_dtank , '.' , target_dtank , '.')
        except:
            traceback.print_exc()
        finally:
            source_dtank.close()
            target_dtank.close()
    return target_tank_path

def file_shadow_name(old_name , insert = 'new'):
    arr = os.path.basename(old_name).split('.')
    arr.insert(-1,insert)
    return os.path.join(os.path.dirname(old_name) , '.'.join(arr))