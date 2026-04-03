from __future__ import annotations

import torch
import numpy as np
import polars as pl

from typing import Any , Literal

from src.proj import Proj , DB , CALENDAR
from src.func.tensor import neutralize_2d , process_factor
from src.data.util import DataBlock
from src.data.loader import BlockLoader

from .core import PreProcessor , FactorPreProcessor , TradePreProcessor , MicellaneousPreProcessor

class PrePros:
    @classmethod
    def keys(cls) -> list[str]:
        return [name for name in PreProcessor.registry.keys()]

    @classmethod
    def start_date(cls , type : Literal['fit' , 'predict'] = 'predict') -> int:
        return PreProcessor.start_date(type)

    @classmethod
    def get_processor(cls , key : str , type : Literal['fit' , 'predict'] , **kwargs) -> PreProcessor:
        return PreProcessor.registry[key](type , **kwargs)

    @classmethod
    def iter_processors(cls , type : Literal['fit' , 'predict'] , **kwargs):
        for key in PreProcessor.registry.keys():
            yield cls.get_processor(key , type)

class PrePro_y(TradePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'y' : BlockLoader('labels_ts', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'tushare_cne5_exp', [*Proj.Conf.Factor.RISK.indus, 'size'])}
    def final_feat(self): return None
    def process(self , blocks : dict[str,DataBlock]): 
        if any([block.empty for block in blocks.values()]):
            return DataBlock()
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

class PrePro_day(TradePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day' : BlockLoader('trade_ts', 'day', ['adjfactor', *self.final_feat()])}
    def process(self , blocks): return blocks['day'].adjust_price()
    
class PrePro_15m(TradePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def process(self , blocks): 
        data_block = blocks['15m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_30m(TradePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_60m(TradePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'60m' : BlockLoader('trade_ts', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def process(self , blocks): 
        data_block = blocks['60m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_week(TradePreProcessor):
    WEEKDAYS = 5
    
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day':BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *self.final_feat()])}
    
    def load_blocks(self , start = None , end = None , secid = None , indent = 0 , vb_level : Any = 1 , **kwargs):
        vb_level = Proj.vb(vb_level)
        if start is not None and start < 0: 
            start = 2 * start
        elif start is not None and start > 0:
            start = CALENDAR.td(start , -self.WEEKDAYS + 1).td
        blocks : dict[str,DataBlock] = {}
        date = CALENDAR.range(start , end)
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start , end , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align_secid_date(secid , date , inplace = True)
            secid = blocks[src_key].secid
        return blocks
    
    def process(self , blocks): 
        data_block = blocks['day'].adjust_price()

        new_values = np.full(np.multiply(data_block.shape,(1, 1, self.WEEKDAYS, 1)),np.nan)
        for i in range(self.WEEKDAYS): 
            new_values[:,self.WEEKDAYS-1-i:,i] = data_block.values[:,:len(data_block.date)-self.WEEKDAYS+1+i,0]
        data_block.update(values = new_values)
        data_block = data_block.adjust_price(adjfactor = False , divide=data_block.loc(inday = 0,feature = 'preclose'))
        return data_block
    
class PrePro_style(PreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'style' : BlockLoader('models', 'tushare_cne5_exp', Proj.Conf.Factor.RISK.style)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['style']

class PrePro_indus(PreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'indus' : BlockLoader('models', 'tushare_cne5_exp', Proj.Conf.Factor.RISK.indus)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['indus']

class PrePro_quality(FactorPreProcessor): ...

class PrePro_growth(FactorPreProcessor): ...

class PrePro_value(FactorPreProcessor): ...

class PrePro_earning(FactorPreProcessor): ...

class PrePro_surprise(FactorPreProcessor): ...
    
class PrePro_coverage(FactorPreProcessor): ...

class PrePro_forecast(FactorPreProcessor): ...

class PrePro_adjustment(FactorPreProcessor): ...

class PrePro_hf_momentum(FactorPreProcessor): ...
    
class PrePro_hf_volatility(FactorPreProcessor): ...

class PrePro_hf_correlation(FactorPreProcessor): ...

class PrePro_hf_liquidity(FactorPreProcessor): ...

class PrePro_momentum(FactorPreProcessor): ...

class PrePro_volatility(FactorPreProcessor): ...

class PrePro_correlation(FactorPreProcessor): ...

class PrePro_liquidity(FactorPreProcessor): ...

class PrePro_holding(FactorPreProcessor): ...

class PrePro_trading(FactorPreProcessor): ...

class PrePro_dfl2(MicellaneousPreProcessor):
    CALCULATION_WINDOW = 250
    MIN_SAMPLES = 90
    FEATURE_CHUNK_SIZE = 20
    def pre_process(self , start : int | None = None , end : int | None = None , * , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 'max' , **kwargs) -> DataBlock:
        # 1. load data into pl.DataFrame
        start = start or self.load_start
        df = DB.loads_pl('sellside', 'dongfang.l2_chars', start = CALENDAR.td(start , -self.CALCULATION_WINDOW + 1).td , end = end , key_column = None)
        if secid is not None:
            df = df.filter(pl.col('secid').is_in(secid))
        # 2. Identify the columns as features (exclude index columns)
        feature = [c for c in df.columns if c not in ['secid', 'date']]

        # 3. Apply rolling z-score partitioned over secid
        df = df.sort(['secid', 'date'])
        blocks = []
        for i in range(0, len(feature), self.FEATURE_CHUNK_SIZE):
            sub_feature = feature[i:i + self.FEATURE_CHUNK_SIZE]
            sub_df = df.select(['secid', 'date'] + sub_feature).lazy().with_columns([
                ((pl.col(feat) - pl.col(feat).rolling_mean(window_size=self.CALCULATION_WINDOW, min_samples=self.MIN_SAMPLES).over("secid")) / 
                pl.col(feat).rolling_std(window_size=self.CALCULATION_WINDOW, min_samples=self.MIN_SAMPLES).over("secid")).alias(feat)
                for feat in sub_feature
            ]).collect()
            blocks.append(DataBlock.from_polars(sub_df).slice_date(start , end))
        del df
        return DataBlock.merge(blocks , inplace = True)
