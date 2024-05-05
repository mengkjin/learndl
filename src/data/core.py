import gc , os
import numpy as np
import pandas as pd
import xarray as xr  

import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Any , ClassVar , Optional

from .fetcher import get_target_dates , load_target_file
from ..classes import DataProcessCfg , NdData
from ..environ import DIR
from ..func import match_values , index_union , index_intersect , forward_fillna
from ..func.time import date_offset , Timer
from ..func.primas import neutralize_2d , process_factor

@dataclass
class DataBlock:
    values  : Any = None 
    secid   : Any = None 
    date    : Any = None 
    feature : Any = None
    save_option : ClassVar[str] = 'pt'

    def __post_init__(self) -> None:
        if self.values is not None: 
            if isinstance(self.feature , str): 
                self.feature = np.array([self.feature])
            elif isinstance(self.feature , list):
                self.feature = np.array(self.feature)
            if self.ndim == 3: self.values = self.values[:,:,None]

    def uninitiate(self):
        self.values  = None
        self.secid   = None
        self.date    = None
        self.feature = None

    def asserted(self):
        if self.shape:
            assert self.ndim == 4
            assert isinstance(self.values , (np.ndarray , Tensor))
            assert self.shape[0] == len(self.secid) 
            assert self.shape[1] == len(self.date)
            assert self.shape[2] == len(self.feature)
        return self
    
    def __repr__(self):
        if self.initiate:
            return '\n'.join(['initiated ' + str(self.__class__) , f'values shape {self.shape}'])
        else:
            return 'uninitiate ' + str(self.__class__) 
    @property
    def initiate(self): return self.values is not None
    @property
    def shape(self): return [] if self.values is None else self.values.shape 
    @property
    def dtype(self): return None if self.values is None else self.values.dtype
    @property
    def ndim(self): return None if self.values is None else self.values.ndim

    def update(self , **kwargs):
        [setattr(self,k,v) for k,v in kwargs.items() if k in ['values','secid','date','feature']]
        return self.asserted()
    
    @classmethod
    def merge(cls , block_list):
        blocks = [blk for blk in block_list if isinstance(blk , cls) and blk.initiate]
        if len(blocks) == 0: return cls()
        elif len(blocks) == 1: return blocks[0]
            
        values = [blk.values for blk in blocks]
        secid  = index_union([blk.secid for blk in blocks])[0]
        date   = index_union([blk.date  for blk in blocks])[0]
        l1 = len(np.unique(np.concatenate([blk.feature for blk in blocks])))
        l2 = sum([len(blk.feature) for blk in blocks])
        distinct_feature = (l1 == l2)

        for blk in blocks: blk.align_secid_date(secid , date)

        if distinct_feature:
            feature = np.concatenate([blk.feature for blk in blocks])
            newdata = np.concatenate([blk.values  for blk in blocks] , axis = -1)
        else:
            feature, p0f , p1f = index_union([blk.feature for blk in blocks])
            newdata = np.full((*blocks[0].shape[:-1],len(feature)) , np.nan , dtype = float)
            for i , data in enumerate(values): newdata[...,p0f[i]] = data[...,p1f[i]]

        return cls(newdata , secid , date , feature)

    def merge_others(self , others : list):
        return self.merge([self , *[others]])
        
    def save(self , key : str , predict=False , start_dt = None , end_dt = None):
        path = self.block_path(key , predict) 
        os.makedirs(os.path.dirname(path),exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        self.save_dict(data , path)
    
    def as_tensor(self , asTensor = True):
        if asTensor and isinstance(self.values , np.ndarray): self.values = torch.tensor(self.values)
        return self
    
    def as_type(self , dtype = None):
        if dtype and isinstance(self.values , np.ndarray): self.values = self.values.astype(dtype)
        if dtype and isinstance(self.values , Tensor): self.values = self.values.to(dtype)
        return self
    
    @staticmethod
    def save_dict(data : dict , file_path : str):
        if file_path is None: return NotImplemented
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        if file_path.endswith(('.npz' , '.npy' , '.np')):
            np.savez_compressed(file_path , **data)
        elif file_path.endswith(('.pt' , '.pth')):
            torch.save(data , file_path , pickle_protocol = 4)
        else:
            raise Exception(file_path)
        
    @staticmethod
    def load_dict(file_path : str , keys = None) -> dict[str,Any]:
        if file_path.endswith(('.npz' , '.npy' , '.np')):
            file = np.load(file_path)
        elif file_path.endswith(('.pt' , '.pth')):
            file = torch.load(file_path)
        else:
            raise Exception(file_path)
        keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
        data = {k:file[k] for k in keys}
        return data
    
    def align(self , secid = None , date = None , feature = None):
        self = self.align_secid_date(secid , date)
        self = self.align_feature(feature)
        return self    

    def align_secid(self , secid):
        if secid is None or len(secid) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((len(secid) , *self.shape[1:]) , np.nan)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        values[p0s] = self.values[p1s]
        self.values = values
        self.secid  = secid
        return self.as_tensor(asTensor).as_type(dtype)
    
    def align_date(self , date):
        if date is None or len(date) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((self.shape[0] , len(date) , *self.shape[2:]) , np.nan)
        _ , p0d , p1d = np.intersect1d(date , self.date , return_indices=True)
        values[:,p0d] = self.values[:,p1d]
        self.values  = values
        self.date    = date
        return self.as_tensor(asTensor).as_type(dtype)
    
    def align_secid_date(self , secid = None , date = None):
        if (secid is None or len(secid) == 0) and (date is None or len(date) == 0): 
            return self
        elif secid is None or len(secid) == 0:
            return self.align_date(date = date)
        elif date is None or len(date) == 0:
            return self.align_secid(secid = secid)
        else:
            asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
            values = np.full((len(secid),len(date),*self.shape[2:]) , np.nan)
            _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
            _ , p0d , p1d = np.intersect1d(date  , self.date  , return_indices=True)
            values[np.ix_(p0s,p0d)] = self.values[np.ix_(p1s,p1d)] 
            self.values  = torch.tensor(values).to(self.values) if isinstance(self.values , Tensor) else values
            self.secid   = secid
            self.date    = date
            return self.as_tensor(asTensor).as_type(dtype)
    
    def align_feature(self , feature):
        if feature is None or len(feature) == 0: return self
        asTensor , dtype = isinstance(self.values , Tensor) , self.dtype
        values = np.full((*self.shape[:-1],len(feature)) , np.nan)
        _ , p0f , p1f = np.intersect1d(feature , self.feature , return_indices=True)
        values[...,p0f] = self.values[...,p1f]
        self.values  = torch.tensor(values).to(self.values) if isinstance(self.values , Tensor) else values
        self.feature = feature
        return self.as_tensor(asTensor).as_type(dtype)
    
    def add_feature(self , new_feature , new_value : np.ndarray | Tensor):
        assert new_value.shape == self.shape[:-1]
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = np.concatenate([self.values,new_value],axis=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        return self
    
    def rename_feature(self , rename_dict : dict):
        if len(rename_dict) == 0: return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): feature[feature == k] = v
        self.feature = feature.astype(str)
        return self
    
    def loc(self , **kwargs) -> np.ndarray | Tensor:
        values : np.ndarray | Tensor = self.values
        for k,v in kwargs.items():  
            if isinstance(v , (str,int,float)): kwargs[k] = [v]
        if 'feature' in kwargs.keys(): 
            index  = match_values(self.feature , kwargs['feature'])
            values = values[:,:,:,index]
        if 'inday'   in kwargs.keys(): 
            index  = match_values(range(values.shape[2]) , kwargs['inday'])
            values = values[:,:,index]
        if 'date'    in kwargs.keys(): 
            index  = match_values(self.date    , kwargs['date'])
            values = values[:,index]
        if 'secid'   in kwargs.keys(): 
            index  = match_values(self.secid   , kwargs['secid'])
            values = values[index]
        return values
    
    @classmethod
    def load_path(cls , path : str): return cls(**cls.load_dict(path))
    
    @classmethod
    def load_paths(cls , paths : str | list[str], fillna = 'guess' , intersect_secid = True ,
                   start_dt = None , end_dt = None , dtype = torch.float):
        if isinstance(paths , str): paths = list(paths)
        _guess = lambda ls,excl:[os.path.basename(x).lower().startswith(excl) == 0 for x in ls]
        if fillna == 'guess':
            exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m','week')
            fillna = np.array(_guess(paths , exclude_list))
        elif fillna is None or isinstance(fillna , bool):
            fillna = np.repeat(fillna , len(paths))
        else:
            assert len(paths) == len(fillna) , (len(paths) , len(fillna))
        
        with Timer(f'Load  {len(paths)} DataBlocks'):
            blocks = [cls.load_path(path) for path in paths]

        with Timer(f'Align {len(paths)} DataBlocks'):
            # sligtly faster than .align(secid = secid , date = date)
            newsecid = None
            if intersect_secid:  newsecid = index_intersect([blk.secid for blk in blocks])[0]
            newdate  = index_union([blk.date for blk in blocks] , start_dt , end_dt)[0]
            
            for i , blk in enumerate(blocks):
                blk.align_secid_date(newsecid , newdate)
                if fillna[i]: blk.values = forward_fillna(blk.values , axis = 1)
                blk.as_tensor().as_type(dtype)

        return blocks
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        return cls.load_path(cls.block_path(key , predict , alias_search))

    @classmethod
    def load_keys(cls , keys : list[str] , predict = False , alias_search = True , **kwargs):
        paths = [cls.block_path(key , predict , alias_search) for key in keys]
        return cls.load_paths(paths , **kwargs)

    @classmethod
    def block_path(cls , key : str , predict=False, alias_search = True):
        train_mark = '.00' if predict else ''
        if key.lower() in ['y' , 'labels']: 
            return f'{DIR.block}/Y{train_mark}.{cls.save_option}'
        else:
            path = (f'{DIR.block}/X_'+'{}'+f'{train_mark}.{cls.save_option}')
            return cls.data_type_alias(path , key) if alias_search else path.format(key)
    
    @classmethod
    def load_DB(cls , data_process_param : dict[str,DataProcessCfg] , start_dt = None , end_dt = None , **kwargs):
        blocks : dict[str,'DataBlock'] = {}
        secid_align , date_align = None , None
        for i , (src_key , param) in enumerate(data_process_param.items()):
            sub_blocks = []
            for db_key in param.db_key:
                with Timer(f'{param.db_src} blocks reading {db_key} DataBase\'s'):
                    blk = cls.load_db(param.db_src , db_key , start_dt , end_dt , param.feature , **kwargs)
                    sub_blocks.append(blk)
            with Timer(f'{src_key} blocks merging'):
                blocks[src_key] = cls.merge(sub_blocks).align(secid = secid_align , date = date_align)
                secid_align , date_align = blocks[src_key].secid , blocks[src_key].date

        return blocks
    
    @classmethod
    def load_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , feature = None , **kwargs):
        target_dates = get_target_dates(db_src , db_key)
        if start_dt is not None: target_dates = target_dates[target_dates >= start_dt]
        if end_dt   is not None: target_dates = target_dates[target_dates <= end_dt]
        if len(target_dates) == 0: 
            return cls()
        if feature is not None:
            assert isinstance(feature , list) , feature
            feature = [f for f in feature if f not in ['secid' , 'date' , 'minute']]
        dfs = [cls.df_preprocess(load_target_file(db_src,db_key,date),date,feature) for date in target_dates]
        dfs = pd.concat([df for df in dfs if isinstance(df,pd.DataFrame)] , axis = 0)
        dfs = dfs.set_index(['secid' , 'date'] + ['minute'] * ('minute' in dfs.columns.values))
        xarr = NdData.from_xarray(xr.Dataset.from_dataframe(dfs))
        return cls(xarr.values , xarr.index[0] , xarr.index[1] , xarr.index[-1])
    
    @classmethod
    def df_preprocess(cls , df : Optional[pd.DataFrame] , date : int , feature = None):
        if isinstance(df , pd.DataFrame): 
            if 'date' not in df.columns.values: df['date'] = date
            if feature is not None: 
                remain_cols = ['secid','date']+['minute']*('minute' in df.columns.values)+feature
                df = df.loc[:,remain_cols]
        else:
            df = None
        return df

    @staticmethod
    def data_type_abbr(key : str):
        key = key.lower()
        if (key.startswith('trade_') and len(key)>6):
            return key[6:]
        elif key.startswith(('rtn_lag','res_lag','std_lag')):
            return '{:s}{:d}'.format(key[:3] , sum([int(s) for s in key[7:].split('_')]))
        elif key in ['y' , 'labels']:
            return 'y'
        else:
            return key
        
    @staticmethod
    def data_type_alias(path : str , key : str):
        alias_list = [key , f'trade_{key}' , key.replace('trade_','')]
        for alias in alias_list:
            if os.path.exists(path.format(alias)): 
                return path.format(alias)
        return path.format(key)
    
    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
    
        adjfactor = adjfactor and ('adjfactor' in self.feature)
        if multiply is None and divide is None and (not adjfactor): return self  

        if isinstance(price_feat , (str,)): price_feat = [price_feat]
        i_price = np.where(np.isin(self.feature , price_feat))[0].astype(int)
        if len(i_price) == 0: return self
        v_price = self.values[...,i_price]

        if adjfactor : v_price *= self.loc(feature=['adjfactor'])
        if multiply  is not None: v_price *= multiply
        if divide    is not None: v_price /= divide

        self.values[...,i_price] = v_price 
        return self
    
    def adjust_volume(self , multiply = None , divide = None , 
                      vol_feat = ['volume' , 'amount', 'turn_tt', 'turn_fl', 'turn_fr']):
        if multiply is None and divide is None: return self

        if isinstance(vol_feat , (str,)): vol_feat = [vol_feat]
        i_vol = np.where(np.isin(self.feature , vol_feat))[0]
        if len(i_vol) == 0: return self
        v_vol = self.values[...,i_vol]
        if multiply is not None: v_vol *= multiply
        if divide   is not None: v_vol /= divide
        self.values[...,i_vol] = v_vol
        return self
    
    def mask_values(self , mask = {'list_dt':91} , **kwargs):
        if not mask : return self

        if mask.get('list_dt'):
            desc = load_target_file('information' , 'description')
            assert desc is not None
            desc = desc[desc['secid'] > 0].loc[:,['secid','list_dt','delist_dt']]
            if len(np.setdiff1d(self.secid , desc['secid'])) > 0:
                add_df = pd.DataFrame({
                        'secid':np.setdiff1d(self.secid , desc['secid']) ,
                        'list_dt':21991231 , 'delist_dt':21991231})
                desc = pd.concat([desc,add_df],axis=0).reset_index(drop=True)

            desc = desc.sort_values('list_dt',ascending=False).drop_duplicates(subset=['secid'],keep='first').set_index('secid') 
            secid , date = self.secid , self.date
            
            list_dt = np.array(desc.loc[secid , 'list_dt'])
            list_dt[list_dt < 0] = 21991231
            list_dt = date_offset(list_dt , mask.get('list_dt') , astype = int)

            delist_dt = np.array(desc.loc[secid , 'delist_dt'])
            delist_dt[delist_dt < 0] = 21991231

            mask = np.stack([(date <= l) + (date >= d) for l,d in zip(list_dt , delist_dt)],axis = 0) 
            self.values[mask] = np.nan

        return self
    
    def hist_norm(self , key : str , predict = False ,
                  start_dt : Optional[int] = None , end_dt : Optional[int]  = 20161231 , 
                  step_day = 5 , **kwargs):
        if predict: return None
        if not key.startswith(('x_trade','trade','day','15m','min','30m','60m','week')): 
            return None
        key = self.data_type_abbr(key)
        maxday = {
            'day'   : 60 ,
            'other' : 1 ,
        }
        maxday = maxday[key] if key in maxday.keys() else maxday['other']

        date_slice = np.repeat(True , len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False

        secid = self.secid
        date  = self.date
        feat  = self.feature
        inday = self.shape[2]

        len_step = len(date[date_slice]) // step_day
        len_bars = maxday * inday

        x = torch.tensor(self.values[:,date_slice])
        pad_array = (0,0,0,0,maxday,0,0,0)
        x = torch.nn.functional.pad(x , pad_array , value = torch.nan)
        
        avg_x = torch.zeros(len_bars , len(feat))
        std_x = torch.zeros(len_bars , len(feat))

        x_endpoint = x.shape[1]-1 + step_day * np.arange(-len_step + 1 , 1)
        x_div = torch.ones(len(secid) , len_step , 1 , len(feat)).to(x)
        re_shape = (*x_div.shape[:2] , -1)
        if key in ['day']:
            # day : divide by endpoint
            x_div.copy_(x[:,x_endpoint,-1:])
        else:
            # will not do anything, just sample mean and std
            '''
            # Xmin day : price divide by preclose , other divide by day sum
            x_div.copy_(x[:,x_endpoint].sum(dim=2 , keepdim=True))
            price_feat = [f for f in ['preclose' , 'close', 'high', 'low', 'open', 'vwap'] if f in feat]
            if len(price_feat) > 0: x_div[...,np.isin(feat , price_feat)] = x[:,x_endpoint-1,-1:][...,feat == price_feat[0]]
            '''
            
        nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
        nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
        for i in range(maxday):
            nan_sample += x[:,x_endpoint-i].reshape(*re_shape).isnan().any(dim=-1)

        for i in range(maxday):
            vijs = ((x[:,x_endpoint - maxday+1 + i]) / (x_div + 1e-6))[nan_sample == 0]
            avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
            std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
        
        data = DataBlockNorm(avg_x , std_x)
        data.save(key)
        return data
    
    @classmethod
    def blocks_process(cls , blocks : dict[str,'DataBlock'] , key):
        np.seterr(invalid='ignore' , divide = 'ignore')
        key_abbr = cls.data_type_abbr(key)
        if key_abbr == 'y':
            final_feat = None
            data_block = blocks['labels']
            model_exp  = blocks['models']
            indus_size = model_exp.values[...,:model_exp.feature.tolist().index('size')+1]
            x = Tensor(indus_size).permute(1,0,2,3).squeeze(2)
            for i_feat,lb_name in enumerate(data_block.feature):
                if lb_name[:3] == 'rtn':
                    y_raw = Tensor(data_block.values[...,i_feat]).permute(1,0,2).squeeze(2)
                    y_std = Tensor(neutralize_2d(y_raw , x)).permute(1,0).unsqueeze(2).numpy()
                    data_block.add_feature('std'+lb_name[3:],y_std)

            y_ts = Tensor(data_block.values)[:,:,0]
            for i_feat,lb_name in enumerate(data_block.feature):
                y_pro = process_factor(y_ts[...,i_feat], dim = 0)
                if not isinstance(y_pro , Tensor): continue
                y_pro = y_pro.unsqueeze(-1).numpy()
                data_block.values[...,i_feat] = y_pro

        elif key_abbr == 'day':
            final_feat = ['open','close','high','low','vwap','turn_fl']
            data_block = blocks[key]
            data_block = data_block.adjust_price()

        elif key_abbr in ['15m','30m','60m']:
            final_feat = ['open','close','high','low','vwap','turn_fl']
            data_block = blocks[key]
            db_day     = blocks['trade_day'].align(secid = data_block.secid , date = data_block.date)
            
            gc.collect()
            
            data_block = data_block.adjust_price(divide=db_day.loc(feature='preclose'))
            data_block = data_block.adjust_volume(divide=db_day.loc(feature='volume')/db_day.loc(feature='turn_fl'),vol_feat='volume')
            
            data_block.rename_feature({'volume':'turn_fl'})
 
        elif key_abbr in ['week']:
            final_feat = ['open','close','high','low','vwap','turn_fl']
            num_days   = 5
            data_block = blocks['trade_day'].adjust_price()

            new_values = np.full(np.multiply(data_block.shape,(1,1,num_days,1)),np.nan)
            for i in range(num_days): new_values[:,num_days-1-i:,i] = data_block.values[:,:len(data_block.date)-num_days+1+i,0]
            data_block.update(values = new_values)
            data_block = data_block.adjust_price(adjfactor = False , divide=data_block.loc(inday=0,feature='preclose'))
        else:
            raise Exception(key)
        
        data_block.align_feature(final_feat)
        np.seterr(invalid='warn' , divide = 'warn')
        return data_block

@dataclass(slots=True)
class DataBlockNorm:
    avg : Tensor
    std : Tensor
    dtype : Any = None
    save_option : ClassVar[str] = 'pt'

    def __post_init__(self):
        self.avg = self.avg.to(self.dtype)
        self.std = self.std.to(self.dtype)

    def save(self , key):
        DataBlock.save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_path(cls , path : str , dtype = None):
        if not os.path.exists(path): return None
        data = DataBlock.load_dict(path)
        return cls(data['avg'] , data['std'] , dtype)

    @classmethod
    def load_paths(cls , paths : str | list[str] , dtype = None):
        if isinstance(paths , str): paths = [paths]
        norms = [cls.load_path(path , dtype) for path in paths]
        return norms
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        return cls.load_path(cls.norm_path(key , predict , alias_search))

    @classmethod
    def load_keys(cls , keys : str | list[str] , predict = False , alias_search = True , dtype = None):
        if isinstance(keys , str): keys = [keys]
        return [cls.load_key(key , predict , alias_search , dtype) for key in keys]
    
    @classmethod
    def norm_path(cls , key : str , predict = False, alias_search = True):
        if key.lower() == 'y': return f'{DIR.hist_norm}/Y.{cls.save_option}'
        path = (f'{DIR.hist_norm}/X_'+'{}'+f'.{cls.save_option}')
        return DataBlock.data_type_alias(path , key) if alias_search else path.format(key)

@dataclass(slots=True)
class ModuleData:
    '''load datas / norms / index'''
    x : dict[str,DataBlock]
    y : DataBlock
    norms : dict[str,DataBlockNorm]
    secid : np.ndarray
    date  : np.ndarray

    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def load(cls , data_type_list : list[str] , y_labels = None , predict=False , dtype = torch.float , save_upon_loading = True):
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
            x : dict[str,DataBlock] = {cls.abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
            norms = {cls.abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
            secid , date = blocks[0].secid , blocks[0].date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x' : x , 'y' : y , 'norms' : norms , 'secid' : secid , 'date' : date}
            if not predict and save_upon_loading: 
                os.makedirs(os.path.dirname(torch_pack) , exist_ok=True)
                torch.save(data , torch_pack , pickle_protocol = 4)
            data = cls(**data)

        if y_labels is not None:  data.y.align_feature(y_labels)
        return data
    
    @staticmethod
    def abbr(data_type : str): return DataBlock.data_type_abbr(data_type)