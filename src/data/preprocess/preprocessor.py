import torch
import numpy as np

from abc import ABC , abstractmethod
from typing import Any

from src.proj import Proj
from src.func.tensor import neutralize_2d , process_factor
from src.data.util import DataBlock
from src.data.loader import BlockLoader , FactorCategory1Loader

class BaseTypePreProcessor(ABC):
    TRADE_FEAT : list[str] = ['open','close','high','low','vwap','turn_fl']

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @abstractmethod
    def block_loaders(self) -> dict[str,BlockLoader]: ... 
    @abstractmethod
    def final_feat(self) -> list | None: ... 
    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: ...
        
    def load_blocks(self , start = None , end = None , secid_align = None , date_align = None , indent = 0 , vb_level : Any = 1 , **kwargs):
        blocks : dict[str,DataBlock] = {}
        vb_level = Proj.vb(vb_level)
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start , end , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align(secid_align , date_align , inplace = True)
            secid_align = blocks[src_key].secid
            date_align  = blocks[src_key].date
        return blocks
    
    def process_blocks(self, blocks : dict[str,DataBlock]):
        np.seterr(invalid = 'ignore' , divide = 'ignore')
        data_block = self.process(blocks)
        data_block = data_block.align_feature(self.final_feat() , inplace = True)
        np.seterr(invalid = 'warn' , divide = 'warn')
        return data_block

class _ClassProperty:
    def __init__(self , method : str):
        assert method in dir(self) , f'{method} is not in {dir(self)}'
        self.method = method
        self.cache_values = {}

    def __get__(self,instance,owner) -> str:
        if owner not in self.cache_values:
            self.cache_values[owner] = getattr(self , self.method)(owner)
        return self.cache_values[owner]

    def __set__(self,instance,value):
        raise AttributeError(f'{instance.__class__.__name__}.{self.method} is read-only attributes')

    def category0(self , owner) -> str:
        return Proj.Conf.Factor.STOCK.cat1_to_cat0(owner.category1)

    def category1(self , owner) -> str:
        return str(owner.__qualname__).removeprefix('PrePro_').lower()
    
class BaseFactorPreProcessor(BaseTypePreProcessor):
    category0 = _ClassProperty('category0')
    category1 = _ClassProperty('category1')    

    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'factor' : FactorCategory1Loader(self.category1 , normalize = True , fill_method = 'drop' , preprocess = True)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['factor']

class PrePro_y(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'y' : BlockLoader('labels_ts', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'tushare_cne5_exp', [*Proj.Conf.Factor.RISK.indus, 'size'])}
    def final_feat(self): return None
    def process(self , blocks : dict[str,DataBlock]): 
        data_block , model_exp = blocks['y'] , blocks['risk']
        indus_size = model_exp.values[...,:]
        x = torch.Tensor(indus_size).squeeze(2)
        for i_feat,lb_name in enumerate(data_block.feature):
            if lb_name.startswith('rtn'):
                y_raw = data_block.values[...,i_feat].squeeze(2)
                y_std = neutralize_2d(y_raw , x , dim = 0)
                assert y_std is not None , 'y_std is None'
                y_std = y_std.unsqueeze(2)
                data_block.add_feature('std'+lb_name[3:],y_std)

        y_ts = data_block.values[:,:,0]
        for i_feat,lb_name in enumerate(data_block.feature):
            y_pro = process_factor(y_ts[...,i_feat], dim = 0)
            if y_pro is None: 
                continue
            data_block.values[...,i_feat] = y_pro.unsqueeze(-1)

        return data_block

class PrePro_day(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day' : BlockLoader('trade_ts', 'day', ['adjfactor', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): return blocks['day'].adjust_price()
    
class PrePro_15m(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): 
        data_block = blocks['15m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_30m(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_60m(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'60m' : BlockLoader('trade_ts', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): 
        data_block = blocks['60m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_week(BaseTypePreProcessor):
    WEEKDAYS = 5
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day':BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def load_blocks(self , start = None , end = None , secid_align = None , date_align = None , indent = 0 , vb_level : Any = 1 , **kwargs):
        vb_level = Proj.vb(vb_level)
        if start is not None and start < 0: 
            start = 2 * start
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start , end , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align(secid_align , date_align , inplace = True)
            secid_align = blocks[src_key].secid
            date_align  = blocks[src_key].date
        return blocks
    
    def process(self , blocks): 
        data_block = blocks['day'].adjust_price()

        new_values = np.full(np.multiply(data_block.shape,(1, 1, self.WEEKDAYS, 1)),np.nan)
        for i in range(self.WEEKDAYS): 
            new_values[:,self.WEEKDAYS-1-i:,i] = data_block.values[:,:len(data_block.date)-self.WEEKDAYS+1+i,0]
        data_block.update(values = new_values)
        data_block = data_block.adjust_price(adjfactor = False , divide=data_block.loc(inday = 0,feature = 'preclose'))
        return data_block
    
class PrePro_style(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'style' : BlockLoader('models', 'tushare_cne5_exp', Proj.Conf.Factor.RISK.style)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['style']

class PrePro_indus(BaseTypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'indus' : BlockLoader('models', 'tushare_cne5_exp', Proj.Conf.Factor.RISK.indus)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['indus']

class PrePro_quality(BaseFactorPreProcessor): ...

class PrePro_growth(BaseFactorPreProcessor): ...

class PrePro_value(BaseFactorPreProcessor): ...

class PrePro_earning(BaseFactorPreProcessor): ...

class PrePro_surprise(BaseFactorPreProcessor): ...
    
class PrePro_coverage(BaseFactorPreProcessor): ...

class PrePro_forecast(BaseFactorPreProcessor): ...

class PrePro_adjustment(BaseFactorPreProcessor): ...

class PrePro_hf_momentum(BaseFactorPreProcessor): ...
    
class PrePro_hf_volatility(BaseFactorPreProcessor): ...

class PrePro_hf_correlation(BaseFactorPreProcessor): ...

class PrePro_hf_liquidity(BaseFactorPreProcessor): ...

class PrePro_momentum(BaseFactorPreProcessor): ...

class PrePro_volatility(BaseFactorPreProcessor): ...

class PrePro_correlation(BaseFactorPreProcessor): ...

class PrePro_liquidity(BaseFactorPreProcessor): ...

class PrePro_holding(BaseFactorPreProcessor): ...

class PrePro_trading(BaseFactorPreProcessor): ...
