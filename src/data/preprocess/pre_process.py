import argparse , gc , inspect , torch
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from datetime import datetime
from typing import Any , Iterator

from src.proj import Logger , Timer , Duration
from src.basic import CONF , CALENDAR
from src.func.primas import neutralize_2d , process_factor
from src.data.util import DataBlock
from src.data.loader import BlockLoader , FactorCategory1Loader

__all__ = ['DataPreProcessor']

TRAIN_DATASET = ['y' , 'day' , '30m' , 'style' , 'indus' , 'week' , 
                 'quality' , 'growth' , 'value' , 'earning' , 
                 'surprise' , 'coverage' , 'forecast' , 'adjustment' , 
                 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                 'momentum' , 'volatility' , 'correlation' , 'liquidity' , 
                 'holding' , 'trading']
PREDICT_DATASET = TRAIN_DATASET

@dataclass(slots=True)
class DataPreProcessor:
    predict         : bool 
    blocks          : list[str] = field(default_factory=list)
    mask            : dict[str,Any] = field(default_factory=dict)
    load_start_dt   : int | None = None
    load_end_dt     : int | None = None
    save_start_dt   : int | None = None
    save_end_dt     : int | None = None
    hist_start_dt   : int | None = None
    hist_end_dt     : int | None = None    

    def __post_init__(self):
        self.blocks = [DataBlock.data_type_abbr(blk) for blk in self.blocks]
        if self.predict:
            self.load_start_dt = -366
        else:
            self.load_start_dt = 20070101
            self.save_start_dt = 20070101
            self.hist_end_dt   = 20161231
        if not self.mask: 
            self.mask = {'list_dt': 91}

    def processors(self) -> Iterator[tuple[str , 'TypePreProcessor']]:
        for blk in self.blocks:
            yield blk , select_processor(blk)
    
    @classmethod
    def proceed(cls , predict = True):
        return cls.main(predict = True , vb_level = 10 if predict else 1)

    @classmethod
    def main(cls , predict = False, confirm = 0 , parser = None , data_types : list[str] | None = None , indent : int = 0 , vb_level : int = 1):
        if parser is None:
            parser = argparse.ArgumentParser(description = 'manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()
        if not predict and not args.confirm and \
            not input('Confirm update data? type "yes" to confirm!').lower()[0] == 'y' : 
            return
        
        if data_types is None:
            blocks = PREDICT_DATASET if predict else TRAIN_DATASET
        else:
            blocks = data_types
        processor = cls(predict , blocks)
        Logger.remark(f'Data PreProcessing start with {len(processor.blocks)} datas and predict = {predict}!' , indent = indent , vb_level = vb_level)
        Logger.stdout(f'Will process {str(list(processor.blocks))} at {CALENDAR.dates_str([processor.load_start_dt,processor.load_end_dt])}' , 
                        indent = indent + 1 , vb_level = vb_level + 1)
        # return processor
        for key , proc in processor.processors():
            modified_time = DataBlock.last_modified_time(key , predict)
            if CALENDAR.is_updated_today(modified_time):
                time_str = datetime.strptime(str(modified_time) , '%Y%m%d%H%M%S').strftime("%Y-%m-%d %H:%M:%S")
                Logger.skipping(f'[{key.upper()}] already preprocessing at {time_str}!' , indent = indent + 1 , vb_level = vb_level)
                continue

            tt1 = datetime.now()
            Logger.stdout(f'Preprocess [{key.upper()}] with predict={predict} start...' , indent = indent + 1 , vb_level = vb_level + 2)

            with Timer(f'[{key}] blocks loading' , indent = indent + 2 , vb_level = vb_level + 2 , enter_vb_level = vb_level + 3):
                block_dict = proc.load_blocks(processor.load_start_dt, processor.load_end_dt, indent = indent + 2 , vb_level = vb_level + 3)
            with Timer(f'[{key}] blocks process' , indent = indent + 2 , vb_level = vb_level + 2):
                data_block = proc.process_blocks(block_dict)
            with Timer(f'[{key}] blocks masking' , indent = indent + 2 , vb_level = vb_level + 2):   
                data_block = data_block.mask_values(mask = processor.mask)
            with Timer(f'[{key}] blocks saving ' , indent = indent + 2 , vb_level = vb_level + 2):
                data_block.save(key , predict , processor.save_start_dt , processor.save_end_dt)
            with Timer(f'[{key}] blocks norming' , indent = indent + 2 , vb_level = vb_level + 2):
                data_block.hist_norm(key , predict , processor.hist_start_dt , processor.hist_end_dt)
            del data_block
            gc.collect()
            Logger.success(f'Preprocess [{key.upper()}] with predict={predict} finished! Cost {Duration(since = tt1)}' , 
                           indent = indent + 1 , vb_level = vb_level + 1)
            Logger.divider(vb_level = vb_level + 1)

class TypePreProcessor(ABC):
    TRADE_FEAT : list[str] = ['open','close','high','low','vwap','turn_fl']

    @abstractmethod
    def block_loaders(self) -> dict[str,BlockLoader]: ... 
    @abstractmethod
    def final_feat(self) -> list | None: ... 
    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: ...
        
    def load_blocks(self , start_dt = None , end_dt = None , secid_align = None , date_align = None , indent = 0 , vb_level = 1 , **kwargs):
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start_dt , end_dt , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align(secid_align , date_align , inplace = True)
            secid_align = blocks[src_key].secid
            date_align  = blocks[src_key].date
        return blocks
    
    def process_blocks(self, blocks : dict[str,DataBlock]):
        np.seterr(invalid = 'ignore' , divide = 'ignore')
        data_block = self.process(blocks)
        data_block = data_block.align_feature(self.final_feat() , inplace = True)
        np.seterr(invalid = 'warn' , divide = 'warn')
        return data_block
    
def select_processor(key : str) -> TypePreProcessor:
    return getattr(inspect.getmodule(select_processor) , f'pp_{key.lower()}')()

class pp_y(TypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'y' : BlockLoader('labels_ts', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'tushare_cne5_exp', [*CONF.Factor.RISK.indus, 'size'])}
    def final_feat(self): return None
    def process(self , blocks : dict[str,DataBlock]): 
        data_block , model_exp = blocks['y'] , blocks['risk']
        indus_size = model_exp.values[...,:]
        x = torch.Tensor(indus_size).permute(1,0,2,3).squeeze(2)
        for i_feat,lb_name in enumerate(data_block.feature):
            if lb_name[:3] == 'rtn':
                y_raw = torch.Tensor(data_block.values[...,i_feat]).permute(1,0,2).squeeze(2)
                y_std = torch.Tensor(neutralize_2d(y_raw , x)).permute(1,0).unsqueeze(2).numpy()
                data_block.add_feature('std'+lb_name[3:],y_std)

        y_ts = torch.Tensor(data_block.values)[:,:,0]
        for i_feat,lb_name in enumerate(data_block.feature):
            y_pro = process_factor(y_ts[...,i_feat], dim = 0)
            if not isinstance(y_pro , torch.Tensor): 
                continue
            y_pro = y_pro.unsqueeze(-1).numpy()
            data_block.values[...,i_feat] = y_pro

        return data_block
class pp_day(TypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day' : BlockLoader('trade_ts', 'day', ['adjfactor', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): return blocks['day'].adjust_price()
    
class pp_15m(TypePreProcessor):
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
    
class pp_30m(TypePreProcessor):
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
    
class pp_60m(TypePreProcessor):
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
    
class pp_week(TypePreProcessor):
    WEEKDAYS = 5
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'day':BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def load_blocks(self , start_dt = None , end_dt = None , secid_align = None , date_align = None , indent = 0 , vb_level = 1 , **kwargs):
        if start_dt is not None and start_dt < 0: 
            start_dt = 2 * start_dt
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start_dt , end_dt , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align(secid_align , date_align , inplace = True)
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
    
class pp_style(TypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'style' : BlockLoader('models', 'tushare_cne5_exp', CONF.Factor.RISK.style)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['style']

class pp_indus(TypePreProcessor):
    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'indus' : BlockLoader('models', 'tushare_cne5_exp', CONF.Factor.RISK.indus)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['indus']

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
        return CONF.Factor.STOCK.cat1_to_cat0(owner.category1)

    def category1(self , owner) -> str:
        return str(owner.__qualname__).removeprefix('pp_').lower()
    
class FactorPreProcessor(TypePreProcessor):
    category0 = _ClassProperty('category0')
    category1 = _ClassProperty('category1')    

    def block_loaders(self) -> dict[str,BlockLoader]: 
        return {'factor' : FactorCategory1Loader(self.category1 , normalize = True , fill_method = 'drop' , preprocess = True)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['factor']

class pp_quality(FactorPreProcessor): ...

class pp_growth(FactorPreProcessor): ...

class pp_value(FactorPreProcessor): ...

class pp_earning(FactorPreProcessor): ...

class pp_surprise(FactorPreProcessor): ...
    
class pp_coverage(FactorPreProcessor): ...

class pp_forecast(FactorPreProcessor): ...

class pp_adjustment(FactorPreProcessor): ...

class pp_hf_momentum(FactorPreProcessor): ...
    
class pp_hf_volatility(FactorPreProcessor): ...

class pp_hf_correlation(FactorPreProcessor): ...

class pp_hf_liquidity(FactorPreProcessor): ...

class pp_momentum(FactorPreProcessor): ...

class pp_volatility(FactorPreProcessor): ...

class pp_correlation(FactorPreProcessor): ...

class pp_liquidity(FactorPreProcessor): ...

class pp_holding(FactorPreProcessor): ...

class pp_trading(FactorPreProcessor): ...
