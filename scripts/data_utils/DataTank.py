import h5py
import pandas as pd
import numpy as np
import random, string , os , time

def _index_union(idxs):
    new_idx = None
    for idx in idxs:
        if new_idx is None or idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    new_idx = np.sort(new_idx)
    pos = tuple([None if idx is None else np.intersect1d(new_idx , idx , return_indices=True)[1] for idx in idxs])
    return new_idx , pos

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

    new_row, pos_row = _index_union([old_row , upd_row])  
    new_col, pos_col = _index_union([old_col , upd_col]) 
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
        secid, pos_secid = _index_union([d.secid for d in self.data.values()])
        feat , pos_feat  = _index_union([d.feature for d in self.data.values()])
        datas = self.data.values()
        all_values = np.full((len(date) , len(secid),len(feat)) , np.nan)
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
        date , pos_date  = _index_union([d.date for d in self.data.values()])
        secid, pos_secid = _index_union([d.secid for d in self.data.values()])
        feat  = self.feature
        datas = self.data.values()
        all_values = np.full((len(date) , len(secid),len(feat)) , np.nan)
        for i_feat , (data , i_date , i_sec) in enumerate(zip(datas , pos_date , pos_secid)):
            tmp = all_values[i_date,:,i_feat]
            tmp[:,i_sec] = data.values[:]
            all_values[i_date,:,i_feat] = tmp
        return DataDSF(date , [Data1D(secid,feat,all_values[i]) for i in range(len(date))])

class DataTank():
    def __init__(self , filename = None , open = False , mode = 'guess' , compression = 'gzip') -> None:
        self.filename = filename
        self.file = None
        if open: self.open(mode = mode)
        self._zip = compression

    def __repr__(self):
        return f'{self.__class__} : {self.filename} -- ({"Closed" if self.file is None else "Opened"})'

    def change_file(self , filename):
        self.filename = filename
        if isinstance(self.file , h5py.File): self.open(self.file.mode)

    def open(self , mode = 'guess'): 
        if mode == 'guess': mode = 'r+' if os.path.exists(self.filename) else 'w'
        self.file = h5py.File(self.filename , mode)
        return self.file

    def _check_read_mode(self):
        return self.open() if self.file is None or self.file.mode not in ['r' ,'r+'] else self.file
    
    def _check_write_mode(self):
        return self.open() if self.file is None or self.file.mode not in ['w' ,'r+'] else self.file
    
    def close(self): self.file.close()

    def tree(self , object = None):
        if object is None: object = self._check_read_mode()
        if isinstance(object , h5py.Group):
            result = {}
            for k,v in object.items(): result.update({k:self.tree(v)})
            return {object.filename:result} if isinstance(object , h5py.File) else result
        else:
            return object.__repr__()

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
            if not full_tree and isinstance(obj , h5py.Group) and depth <= 1:
                _tailkeys , _n_omitkeys = [str(k) for k in list(obj.keys())[:num_tail_keys]] , len(obj.keys()) - num_tail_keys
                print_tail = '('+', '.join(_tailkeys)+('' if _n_omitkeys <= 0 else f' and {_n_omitkeys} more')+')'
                print(f'{print_pre}{inter} {key}' , obj , print_tail)
            else:
                print(f'{print_pre}{inter} {key}' , obj)
                if isinstance(obj , h5py.Group): 
                    self.print_tree(obj,f'{print_pre}{afpre} ',depth-1,width,depth_full_width-1,num_tail_keys,full_tree)


    def get_object(self , file):
        file = self._full_path(file)
        return self.file.get(file)
    
    def del_object(self , file , ask = True):
        file = self._full_path(file)
        if self.file.get(file) is None: return
        if ask and input('press yes to confirm') != 'yes': return
        del self.file[file]

    def delete_all(self , allow = False):
        if not allow: return NotImplemented
        rand_str = 'yes'
        if input(f'press {rand_str} to confirm') != rand_str: return
        rand_num = np.random.randint(100,999)
        if input(f'press {rand_num} to confirm') != str(rand_num): return
        rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        if input(f'press {rand_str} to confirm') != rand_str: return
        [self.del_object(k) for k in self.file.keys()]
    
    def create_object(self , file , data = None , **kwargs):
        assert self.file.mode in ['w' ,'r+'] , self.file.mode
        file = self._full_path(file)
        assert self.get_object(file) is None, file
        if data is None:
            return self.file.create_group(file)
        else:
            return self.file.create_dataset(file , data = data , **kwargs)

    def write_dataframe(self , file , data , overwrite = False):
        assert self.file.mode in ['w' ,'r+'] , self.file.mode
        if not isinstance(data , pd.DataFrame): data = pd.DataFrame(data)
        file = self._full_path(file)
        self.del_object(file , ask = not overwrite)
        str_dtype = h5py.string_dtype(encoding='utf-8')
        columns , columns_dtype = data.columns.values.tolist() , data.dtypes.values.astype(str).tolist()
        save_dtype = [str_dtype if dtype == 'object' else None for dtype in columns_dtype]
        for col , sdtype in zip(columns , save_dtype):
            self.create_object([file, col], data=data.loc[:,col], dtype=sdtype, compression=self._zip)
        self.set_group_attrs(file , __columns__ = columns)
        self.set_group_attrs(file , __columns_dtype__ = columns_dtype)
        
    def read_dataframe(self , file):
        file = self._full_path(file)
        data = self.get_object(file)
        assert data is not None, file
        columns = self.get_group_attrs(file , '__columns__') # self.get_object([file , '__columns__'])[:].astype(str)
        col_dtype = self.get_group_attrs(file , '__columns_dtype__') # self.get_object([file , '__columns_dtype__'])[:].astype(str)
        df = pd.DataFrame()
        for col , dtype in zip(columns , col_dtype): 
            if dtype == 'object':
                df[col] = np.char.decode(data[col][:].astype(bytes) , 'utf-8')
            else:
                df[col] = data[col][:].astype(getattr(np , dtype))            
        return df

    def write_data1D(self , file , data , overwrite = False):
        assert self.file.mode in ['w' ,'r+'] , self.file.mode
        if not isinstance(data , Data1D): data = Data1D(src=data)
        file = self._full_path(file)
        self.del_object(file , ask = not overwrite)
        self.create_object([file , 'secid'] , data = data.secid , compression = self._zip)
        self.create_object([file , 'feature'] , data = data.feature , compression = self._zip)
        self.create_object([file , 'values'] , data = data.values.astype(float) , compression = self._zip)

    def read_data1D(self , file):
        return Data1D(src=self.get_object(file))
    
    def write_dataDSF(self , file , data , overwrite = True):
        assert self.file.mode in ['w' ,'r+'] , self.file.mode
        # assert isinstance(data , DataDSF) , data.__class__
        file = self._full_path(file)
        for k , v in data.data.items():
            self.write_data1D([file , str(k)] , v , overwrite = overwrite)

    def read_dataDSF(self , file , start = None, end = None):
        __start_time__ = time.time()
        if start is None: start = -1
        if end is None: end = 99999999
        fileDSF = self.get_object(file)
        date = np.array(list(fileDSF.keys())).astype(int)
        date = date[(date >= start) & (date <= end)]
        result = DataDSF(src={str(k):self.read_data1D([file,str(k)]) for k in date})
        print(f'loading: {time.time() - __start_time__:.2f} secs')
        return result

    def read_dataFDS(self , file , start = None , end = None , dict_only = True) -> None:
        __start_time__ = time.time()
        if start is None: start = -1
        if end is None: end = 99999999
        portal = self.get_object(file)
        date = np.array(list(portal.keys())).astype(int)
        date = date[(date >= start) & (date <= end)]
        secid, pos_secid = _index_union([portal[str(d)]['secid'][:] for d in date])
        feat , pos_feat  = _index_union([portal[str(d)]['feature'][:] for d in date])
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
    
    def get_group_attrs(self , file , attr = ['__information__' , '__create_time__','__last_date__', '__update_time__']):
        g = self.get_object(file)
        assert g is not None, file
        return g.attrs.get(attr) if isinstance(attr , str) else {a:g.attrs.get(a) for a in attr}

    def set_group_attrs(self , file , **kwargs):
        g = self.get_object(file)
        assert g is not None, file
        [g.attrs.modify(k , v) for k,v in kwargs.items() if v is not None] 

    def _full_path(self , file):
        return '/'.join([str(f) for f in file]) if isinstance(file , (list,tuple)) else file