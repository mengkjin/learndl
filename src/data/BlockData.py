import torch

import os , gc
import numpy as np
import pandas as pd

from copy import deepcopy
from .DataTank import DataTank
from .DataUpdater import get_db_path , get_db_file
from ..util.basic import Timer
from ..func.basic import (
    match_values ,
    index_union , index_intersect , forward_fillna ,)
from ..func.date import date_offset
from ..func.primas import neutralize_2d , process_factor

from numpy import savez_compressed as save_npz
from numpy import load as load_npz
from torch import save as save_pt
from torch import load as load_pt

from ..environ import DIR_data

DIR_block      = f'{DIR_data}/block_data'
DIR_hist_norm  = f'{DIR_data}/hist_norm'
save_block_method = 'pt'
_div_tol = 1e-6

class DataBlock():
    def __init__(self , values = None , secid = None , date = None , feature = None) -> None:
        self.initiate = False
        if values is not None:
            self.init_attr(values , secid , date , feature)

    def init_attr(self , values , secid , date , feature):
        if values is None: 
            self._clear_attr()
            return NotImplemented
        self.initiate = True
        if isinstance(feature , str): feature = np.array([feature])
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
        assert self.values.shape[:2] == (len(self.secid) , len(self.date))
        assert self.values.shape[-1] == len(self.feature)
        return self
    
    @classmethod
    def _from_dtank(cls , dtank , inner_path , 
                    start_dt = None , end_dt = None , 
                    feature = None , **kwargs):
        portal = dtank.get_object(inner_path)
        if portal is None: return

        date = np.array(list(portal.keys())).astype(int)
        if start_dt is not None: date = date[date >= start_dt]
        if end_dt   is not None: date = date[date <= end_dt]
        if len(date) == 0: return

        datas = {str(d):dtank.read_data1D([inner_path , str(d)],feature).to_kline() for d in date}
        date  = np.array(list(datas.keys())).astype(int)
        secid , p_s0 , p_s1 = index_union([data.secid for data in datas.values()])
        feature , _ , p_f1 = index_intersect([data.feature for data in datas.values()])
        new_shape = [len(secid),len(date),len(feature)]
        if datas[str(date[0])].values.ndim == 3:
            new_shape.insert(2 , datas[str(date[0])].values.shape[1])
        newdata = np.full(tuple(new_shape) , np.nan , dtype = float)
        for i,(k,v) in enumerate(datas.items()):
            newdata[p_s0[i],i,:] = v.values[p_s1[i]][...,p_f1[i]]
        return newdata , secid , date , feature
    
    @classmethod
    def from_db(cls , db_key , inner_path , start_dt = None , end_dt = None , feature = None , **kwargs):
        db_path = get_db_path(db_key)

        datas = []
        for fn in os.listdir(db_path):
            with DataTank(os.path.join(db_path,fn),'r') as dtank:
                data = cls._from_dtank(dtank , inner_path , start_dt , end_dt , feature)
                if data is not None: datas.append(data)

        obj = cls()
        if len(datas) == 1:
            obj.init_attr(*datas[0])
        elif len(datas) > 1:
            secid , p_s0 , p_s1 = index_union([data[1] for data in datas])
            date  , p_d0 , p_d1 = index_union([data[2] for data in datas])
            feature , _  , p_f1 = index_intersect([data[3] for data in datas])
            new_shape = [len(secid),len(date),datas[0][0].shape[2],len(feature)]
            newdata = np.full(tuple(new_shape) , np.nan , dtype = float)
            for i , data in enumerate(datas):
                newdata[np.ix_(p_s0[i],p_d0[i])] = data[0][np.ix_(p_s1[i],p_d1[i])][...,p_f1[i]]# type: ignore
            obj.init_attr(newdata , secid , date , feature)
        return obj
    
    @classmethod
    def merge(cls , block_list):
        blocks = [blk for blk in block_list if blk.initiate]
        if len(blocks) == 0:
            return cls()
        elif len(blocks) == 1:
            return blocks[0]
            
        values = [blk.values for blk in blocks]
        secid  , p_s0 , p_s1 = index_union([blk.secid for blk in blocks])
        date   , p_d0 , p_d1 = index_union([blk.date  for blk in blocks])
        l1 = len(np.unique(np.concatenate([blk.feature for blk in blocks])))
        l2 = sum([len(blk.feature) for blk in blocks])
        distinct_feature = (l1 == l2)

        for i , data in enumerate(values):
            newdata = np.full((len(secid),len(date),*data.shape[2:]) , np.nan)
            newdata[np.ix_(p_s0[i],p_d0[i])] = data[np.ix_(p_s1[i],p_d1[i])] # type: ignore
            values[i] = newdata

        if distinct_feature:
            feature = np.concatenate([blk.feature for blk in blocks])
            newdata = np.concatenate(values , axis = -1)
        else:
            feature, p_f0 , p_f1 = index_union([blk.feature for blk in blocks])
            newdata = np.full((*newdata[0].shape[:-1],len(feature)) , np.nan , dtype = float)
            for i , data in enumerate(values):
                newdata[...,p_f0[i]] = data[...,p_f1[i]]
        obj = cls()
        obj.init_attr(newdata , secid , date , feature)
        return obj

    def merge_others(self , others : list):
        return self.merge([self , *[others]])
        
    def save(self , key , if_train=True , start_dt = None , end_dt = None):
        path = self.block_path(key , if_train) 
        os.makedirs(os.path.dirname(path),exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        save_dict(data , path)
    
    def to(self , asTensor = None, dtype = None):
        if asTensor and isinstance(self.values , np.ndarray): 
            self.values = torch.Tensor(self.values)
        if dtype: 
            if isinstance(self.values , np.ndarray):
                self.values = self.values.astype(dtype)
            else:
                self.values = self.values.to(dtype)
        return self
    
    def align(self , secid = None , date = None , feature = None):
        self = self.align_secid(secid)
        self = self.align_date(date)
        self = self.align_feature(feature)
        return self    

    def align_secid(self , secid):
        if secid is None or len(secid) == 0: return self
        values = np.full((len(secid) , *self.shape[1:]) , np.nan)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        values[p0s] = self.values[p1s]
        self.values  = values
        self.secid   = secid
        self.shape   = self.values.shape
        return self
    
    def align_date(self , date):
        if date is None or len(date) == 0: return self
        values = np.full((self.shape[0] , len(date) , *self.shape[2:]) , np.nan)
        _ , p0d , p1d = np.intersect1d(date , self.date , return_indices=True)
        values[:,p0d] = self.values[:,p1d]
        self.values  = values
        self.date    = date
        self.shape   = self.values.shape
        return self
    
    def align_feature(self , feature):
        if feature is None or len(feature) == 0: return self
        values = np.full((*self.shape[:-1],len(feature)) , np.nan)
        _ , p0f , p1f = np.intersect1d(feature , self.feature , return_indices=True)
        values[...,p0f] = self.values[...,p1f]
        self.values  = values
        self.feature = feature
        self.shape   = self.values.shape
        return self
    
    def add_feature(self , new_feature , new_value):
        assert new_value.shape == self.shape[:-1]
        new_value = new_value.reshape(*new_value.shape , 1)
        self.values  = np.concatenate([self.values,new_value],axis=-1)
        self.feature = np.concatenate([self.feature,[new_feature]],axis=0)
        self.shape   = self.values.shape
        return self
    
    def rename_feature(self , rename_dict):
        if len(rename_dict) == 0: return self
        feature = self.feature.astype(object)
        for k,v in rename_dict.items(): feature[feature == k] = v
        self.feature = feature.astype(str)
        return self
    
    def get(self , **kwargs) -> np.ndarray | torch.Tensor:
        values = self.values
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
    def load_path(cls , path):
        obj = cls()
        obj.init_attr(**load_dict(path))
        return obj
    
    @classmethod
    def load_paths(cls ,
                   paths , 
                   fillna = 'guess' , 
                   intersect_secid = True ,
                   union_date = True , 
                   start_dt = None , end_dt = None ,
                   dtype = torch.float):
        if isinstance(paths , str): paths = list(paths)
        _guess = lambda ls,excl:[os.path.basename(x).lower().startswith(excl) == 0 for x in ls]
        if fillna == 'guess':
            exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m','week')
            fillna = np.array(_guess(paths , exclude_list))
        elif fillna is None or isinstance(fillna , bool):
            fillna = np.repeat(fillna , len(paths))
        else:
            assert len(paths) == len(fillna) , (len(paths) , len(fillna))
        
        with Timer(f'Load  {len(paths)} DataBlocks') as t:
            blocks = [DataBlock.load_path(path) for path in paths]

        with Timer(f'Align {len(paths)} DataBlocks') as t:
            # sligtly faster than .align(secid = secid , date = date)
            newsecid = newdate = None
            if intersect_secid: 
                newsecid,p_s0,p_s1 = index_intersect([blk.secid for blk in blocks])
            if union_date: 
                newdate ,p_d0,p_d1 = index_union([blk.date for blk in blocks] , start_dt , end_dt)
            
            for i , blk in enumerate(blocks):
                secid = newsecid if newsecid is not None else blk.secid
                date  = newdate  if newdate  is not None else blk.date
                if blk.shape[:2] == (len(secid),len(date)): 
                    values = blk.values
                else: # secid/date alter
                    values = np.full((len(secid),len(date),*blk.shape[2:]) , np.nan)
                    if newsecid is None:
                        values[:,p_d0[i]] = blk.values[:,p_d1[i]]
                    elif newdate is None:
                        values[p_s0[i]] = blk.values[p_s1[i]] #type:ignore
                    else:
                        values[np.ix_(p_s0[i],p_d0[i])] = blk.values[np.ix_(p_s1[i],p_d1[i])] #type:ignore

                date_slice = np.repeat(True , len(date))
                if start_dt is not None: date_slice[date < start_dt] = False
                if end_dt   is not None: date_slice[date > end_dt]   = False
                values , date = values[:,date_slice] , date[date_slice]

                if fillna[i]: values = forward_fillna(values , axis = 1)
                blk.update(values = values , secid = secid , date = date)
                blk.to(asTensor = True , dtype = dtype)
        return blocks
    
    @classmethod
    def load_key(cls , key , if_train = True , alias_search = True , dtype = None):
        return cls.load_path(cls.block_path(key , if_train , alias_search))

    @classmethod
    def load_keys(cls , keys , if_train = True , alias_search = True , **kwargs):
        paths = [cls.block_path(key , if_train , alias_search) for key in keys]
        return cls.load_paths(paths , **kwargs)

    @classmethod
    def block_path(cls , key , if_train=True, alias_search = True , method = save_block_method):
        train_mark = '' if if_train else '.00'
        if key.lower() == 'y': return f'{DIR_block}/Y{train_mark}.{method}'
        path = (f'{DIR_block}/X_'+'{}'+f'{train_mark}.{method}')
        return data_type_alias(path , key) if alias_search else path.format(key)
    
    @classmethod
    def load_DB_source(cls , DB_source , start_dt = None , end_dt = None , **kwargs):
        BlockDict = {}
        for i , src_key in enumerate(DB_source.keys()):
            blocks = []
            db_key = src_key if DB_source[src_key].get('db') is None else DB_source[src_key].get('db')
            inner_path= DB_source[src_key]['inner_path']
            inner_path = [inner_path] if isinstance(inner_path,str) else inner_path
            feature = DB_source[src_key].get('feature')
            for path in inner_path:
                with Timer(f'{db_key} blocks reading {path} Data1D\'s') as t:
                    blocks.append(cls.from_db(db_key , path , feature = feature , 
                                                    start_dt = start_dt , end_dt = end_dt , **kwargs))
            with Timer(f'{src_key} blocks merging') as t:
                BlockDict[src_key] = cls.merge(blocks)

            if i == 0:
                secid_align , date_align = BlockDict[src_key].secid , BlockDict[src_key].date
            else:
                BlockDict[src_key].align(secid = secid_align , date = date_align)

        return BlockDict
    
    def adjust_price(self , adjfactor = True , multiply = 1 , divide = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
    
        adjfactor = adjfactor and ('adjfactor' in self.feature)
        if multiply is None and divide is None and (not adjfactor): return self  

        if isinstance(price_feat , (str,)): price_feat = [price_feat]
        i_price = np.where(np.isin(self.feature , price_feat))[0].astype(int)
        if len(i_price) == 0: return self
        v_price = self.values[...,i_price]

        if adjfactor : v_price *= self.get(feature=['adjfactor'])
        if multiply  is not None: v_price *= multiply
        if divide    is not None: v_price /= divide

        self.values[...,i_price] = v_price #type:ignore
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
        self.values[...,i_vol] = v_vol #type:ignore
        return self
    
    def mask_values(self , mask = {} , after_ipo = 91 , **kwargs):
        if not mask : return self

        if mask.get('list_dt'):
            with DataTank(get_db_file(get_db_path('information')) , 'r') as info:
                desc = info.read_dataframe('stock/description')
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
            list_dt = date_offset(list_dt , after_ipo , astype = int)

            delist_dt = np.array(desc.loc[secid , 'delist_dt'])
            delist_dt[delist_dt < 0] = 21991231

            mask = np.stack([(date <= l) + (date >= d) for l,d in zip(list_dt,delist_dt)],axis = 0) 
            self.values[mask] = np.nan

        return self
    
    def hist_norm(self , key , if_train=True,
                  start_dt : int | None = None , end_dt : int | None  = 20161231 , 
                  step_day = 5 , **kwargs):
        if not if_train: return None
        if not key.startswith(('x_trade','trade','day','15m','min','30m','60m','week')): 
            return None
        key = data_type_abbr(key)
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
        inday = self.values.shape[2]

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
            pass
            """
            # Xmin day : price divide by preclose , other divide by day sum
            x_div.copy_(x[:,x_endpoint].sum(dim=2 , keepdim=True))
            price_feat = [f for f in ['preclose' , 'close', 'high', 'low', 'open', 'vwap'] if f in feat]
            if len(price_feat) > 0: x_div[...,np.isin(feat , price_feat)] = x[:,x_endpoint-1,-1:][...,feat == price_feat[0]]
            """
            
        nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
        nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
        for i in range(maxday):
            nan_sample += x[:,x_endpoint-i].reshape(*re_shape).isnan().any(dim=-1)

        for i in range(maxday):
            vijs = ((x[:,x_endpoint - maxday+1 + i]) / (x_div + _div_tol))[nan_sample == 0]
            avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
            std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
        
        data = DataBlockNorm(avg_x , std_x)
        data.save(key)
        return data
    
def blocks_process(BlockDict , key) -> DataBlock:
    np.seterr(invalid='ignore' , divide = 'ignore')
    assert isinstance(BlockDict , dict) , type(BlockDict)
    assert all([isinstance(block , DataBlock) for block in BlockDict.values()]) , DataBlock
    
    key_abbr = data_type_abbr(key)

    if key_abbr == 'y':
        final_feat = None
        data_block = BlockDict['labels']

        indus_size = BlockDict['models'].values[...,:BlockDict['models'].feature.tolist().index('size')+1]
        x = torch.FloatTensor(indus_size).permute(1,0,2,3).squeeze(2)
        for i_feat,lb_name in enumerate(BlockDict['labels'].feature):
            if lb_name[:3] == 'rtn':
                y_raw = torch.FloatTensor(BlockDict['labels'].values[...,i_feat]).permute(1,0,2).squeeze(2)
                y_std = neutralize_2d(y_raw , x).permute(1,0).unsqueeze(2).numpy()
                BlockDict['labels'].add_feature('std'+lb_name[3:],y_std)
        del BlockDict['models']

        y_ts = torch.FloatTensor(BlockDict['labels'].values)[:,:,0]
        for i_feat,lb_name in enumerate(BlockDict['labels'].feature):
            y_pro = process_factor(y_ts[...,i_feat], dim = 0)
            if not isinstance(y_pro , torch.Tensor): continue
            y_pro = y_pro.unsqueeze(-1).numpy()
            BlockDict['labels'].values[...,i_feat] = y_pro
        
    elif key_abbr == 'day':
        final_feat = ['open','close','high','low','vwap','turn_fl']
        data_block = BlockDict[key].adjust_price()

    elif key_abbr in ['15m','30m','60m']:
        final_feat = ['open','close','high','low','vwap','turn_fl']
        data_block = BlockDict[key]
        db_day = BlockDict['trade_day'].align(secid = data_block.secid , date = data_block.date)
        del BlockDict['trade_day']
        gc.collect()
        
        data_block = data_block.adjust_price(divide=db_day.get(feature='preclose'))
        data_block = data_block.adjust_volume(divide=db_day.get(feature='volume')/db_day.get(feature='turn_fl'),vol_feat='volume')
        
        data_block.rename_feature({'volume':'turn_fl'})
    elif key_abbr in ['week']:
        
        final_feat = ['open','close','high','low','vwap','turn_fl']
        num_days = 5
        data_block = BlockDict['trade_day']

        data_block = data_block.adjust_price()
        new_values = np.full(np.multiply(data_block.shape,(1,1,num_days,1)),np.nan)
        for i in range(num_days): new_values[:,num_days-1-i:,i] = data_block.values[:,:len(data_block.date)-num_days+1+i,0]
        data_block.update(values = new_values)
        data_block = data_block.adjust_price(adjfactor = False , divide=data_block.get(inday=0,feature='preclose'))
    else:
        raise Exception(key)
    
    data_block.align_feature(final_feat)
    np.seterr(invalid='warn' , divide = 'warn')

    return data_block

"""
def block_load_DB(DB_source , start_dt = None , end_dt = None , **kwargs):
    BlockDict = {}
    for src_key in DB_source.keys():
        blocks = []
        db_key = src_key if DB_source[src_key].get('db') is None else DB_source[src_key].get('db')
        inner_path= DB_source[src_key]['inner_path']
        inner_path = [inner_path] if isinstance(inner_path,str) else inner_path
        feature = DB_source[src_key].get('feature')
        for path in inner_path:
            with Timer(f'{db_key} blocks reading {path} Data1D\'s') as t:
                blocks.append(DataBlock.from_db(db_key , path , feature = feature , 
                                                  start_dt = start_dt , end_dt = end_dt , **kwargs))
        with Timer(f'{src_key} blocks merging') as t:
            BlockDict[src_key] = DataBlock.merge(blocks)

    if len(BlockDict) > 1:
        with Timer(f'{len(BlockDict)} blocks aligning') as t:
            for i , src_key in enumerate(DB_source.keys()):
                if i == 0:
                    secid , date = BlockDict[src_key].secid , BlockDict[src_key].date
                else:
                    BlockDict[src_key] = BlockDict[src_key].align(secid = secid , date = date)
    return BlockDict

def block_adjust_price(data_block , adjfactor = True , multiply = None , divide = None , 
                       price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
    
    adjfactor = adjfactor and ('adjfactor' in data_block.feature)
    if multiply is None and divide is None and (not adjfactor): return data_block  

    if isinstance(price_feat , (str,)): price_feat = [price_feat]
    i_price = np.where(np.isin(data_block.feature , price_feat))[0]
    if len(i_price) == 0: return data_block
    v_price = data_block.values[...,i_price]

    if adjfactor : v_price = np.multiply(v_price , data_block.get(feature=['adjfactor']))
    if multiply  is not None: v_price = np.multiply(v_price , multiply)
    if divide    is not None: v_price = np.divide(v_price , divide)

    data_block.values[...,i_price] = v_price
    return data_block

def block_adjust_volume(data_block , multiply = None , divide = None , 
                     vol_feat = ['volume' , 'amount', 'turn_tt', 'turn_fl', 'turn_fr']):
    if multiply is None and divide is None: return data_block

    if isinstance(vol_feat , (str,)): vol_feat = [vol_feat]
    i_vol = np.where(np.isin(data_block.feature , vol_feat))[0]
    if len(i_vol) == 0: return data_block
    v_vol = data_block.values[...,i_vol]
    if multiply is not None: v_vol = np.multiply(v_vol , multiply)
    if divide   is not None: v_vol = np.divide(v_vol , divide)
    data_block.values[...,i_vol] = v_vol
    return data_block

def block_mask(data_block , mask = {} , after_ipo = 91 , **kwargs):
    if not mask : return data_block
    assert isinstance(data_block , DataBlock) , type(data_block)

    if mask.get('list_dt'):
        with DataTank(get_db_file(get_db_path('information')) , 'r') as info:
            desc = info.read_dataframe('stock/description')
        desc = desc[desc['secid'] > 0].loc[:,['secid','list_dt','delist_dt']]
        if len(np.setdiff1d(data_block.secid , desc['secid'])) > 0:
            add_df = pd.DataFrame({
                    'secid':np.setdiff1d(data_block.secid , desc['secid']) ,
                    'list_dt':21991231 , 'delist_dt':21991231})
            desc = pd.concat([desc,add_df],axis=0).reset_index(drop=True)

        desc = desc.sort_values('list_dt',ascending=False).drop_duplicates(subset=['secid'],keep='first').set_index('secid') 
        secid , date = data_block.secid , data_block.date
        
        list_dt = np.array(desc.loc[secid , 'list_dt'])
        list_dt[list_dt < 0] = 21991231
        list_dt = date_offset(list_dt , after_ipo , astype = int)

        delist_dt = np.array(desc.loc[secid , 'delist_dt'])
        delist_dt[delist_dt < 0] = 21991231

        mask = np.stack([(date <= l) + (date >= d) for l,d in zip(list_dt,delist_dt)],axis = 0) 
        data_block.values[mask] = np.nan

    return data_block
"""

class DataBlockNorm:
    def __init__(self , avg : torch.Tensor, std : torch.Tensor , dtype = None) -> None:
        self.avg = avg.to(dtype)
        self.std = std.to(dtype)

    def save(self , key):
        save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_path(cls , path , dtype = None):
        if not os.path.exists(path): return None
        data = load_dict(path)
        return cls(data['avg'] , data['std'] , dtype)

    @classmethod
    def load_paths(cls , paths , dtype = None):
        if isinstance(paths , str): paths = [paths]
        norms = [cls.load_path(path , dtype) for path in paths]
        return norms
    
    @classmethod
    def load_key(cls , key , if_train = True , alias_search = True , dtype = None):
        return cls.load_path(cls.norm_path(key , if_train , alias_search))

    @classmethod
    def load_keys(cls , keys , if_train = True , alias_search = True , dtype = None):
        if isinstance(keys , str): keys = [keys]
        return [cls.load_key(key , if_train , alias_search , dtype) for key in keys]
    
    @classmethod
    def norm_path(cls , key , if_train=True, alias_search = True , method = save_block_method):
        if key.lower() == 'y': return f'{DIR_hist_norm}/Y.{method}'
        path = (f'{DIR_hist_norm}/X_'+'{}'+f'.{method}')
        return data_type_alias(path , key) if alias_search else path.format(key)

def data_type_abbr(key):
    if (key.startswith('trade_') and len(key)>6):
        return key[6:]
    elif key.startswith(('rtn_lag','res_lag')):
        return f'{key[:3]}{sum([int(s) for s in key[7:].split("_")])}'
    else:
        return key
    
def data_type_alias(path , key):
    alias_list = [key , f'trade_{key}' , key.replace('trade_','')]
    for alias in alias_list:
        if os.path.exists(path.format(alias)): 
            return path.format(alias)
    raise Exception(path.format(key)) 
    
def save_dict(data , file_path):
    if file_path is None: return NotImplemented
    os.makedirs(os.path.dirname(file_path) , exist_ok=True)
    if file_path.endswith(('.npz' , '.np')):
        save_npz(file_path , **data)
    elif file_path.endswith(('.pt' , '.pth')):
        save_pt(data , file_path , pickle_protocol = 4)
    else:
        raise Exception(file_path)
    
def load_dict(file_path , keys = None):
    if file_path.endswith(('.npz' , '.np')):
        file = load_npz(file_path)
    elif file_path.endswith(('.pt' , '.pth')):
        file = load_pt(file_path)
    else:
        raise Exception(file_path)
    keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
    data = {k:file[k] for k in keys}
    return data