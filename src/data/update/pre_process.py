import argparse , gc , inspect, time , torch
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from typing import Any , Iterator , Literal , Optional

from src.basic import CONF , Timer , CALENDAR , Logger
from src.func.primas import neutralize_2d , process_factor
from src.func.classproperty import classproperty_str
from src.data.util import DataBlock
from src.data.loader import BlockLoader , FactorLoader

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
    load_start_dt   : Optional[int] = None
    load_end_dt     : Optional[int] = None
    save_start_dt   : Optional[int] = None
    save_end_dt     : Optional[int] = None
    hist_start_dt   : Optional[int] = None
    hist_end_dt     : Optional[int] = None    

    def __post_init__(self):
        self.blocks = [DataBlock.data_type_abbr(blk) for blk in self.blocks]
        if self.predict:
            self.load_start_dt = -366
        else:
            self.load_start_dt = 20070101
            self.save_start_dt = 20070101
            self.hist_end_dt   = 20161231
        if not self.mask: self.mask = {'list_dt': 91}

    def processors(self) -> Iterator[tuple[str , 'TypePreProcessor']]:
        for blk in self.blocks:
            yield blk , select_processor(blk)
    
    @classmethod
    def proceed(cls):
        return cls.main(predict = True)

    @classmethod
    def main(cls , predict = False, confirm = 0 , parser = None , data_types : Optional[list[str]] = None):
        if parser is None:
            parser = argparse.ArgumentParser(description = 'manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()
        if not predict and not args.confirm and \
            not input('Confirm update data? type "yes" to confirm!').lower()[0] == 'y' : return
        t1 = time.time()
        Logger.info(f'predict is {predict} , Data Processing start!')
        
        if data_types is None:
            blocks = PREDICT_DATASET if predict else TRAIN_DATASET
        else:
            blocks = data_types
        processor = cls(predict , blocks = blocks)
        Logger.info(f'{len(processor.blocks)} datas : {str(list(processor.blocks))} , from {processor.load_start_dt} to {processor.load_end_dt}')
        # return processor
        for key , proc in processor.processors():
            modified_time = DataBlock.last_modified_time(key , predict)
            if CALENDAR.is_updated_today(modified_time):
                Logger.info(f'{key} is up to {modified_time} already!')
                continue
            tt1 = time.time()

            with Timer(f'{key} blocks loading' , newline=True):
                block_dict = proc.load_blocks(processor.load_start_dt, processor.load_end_dt)
            with Timer(f'{key} blocks process'):
                data_block = proc.process_blocks(block_dict)
            with Timer(f'{key} blocks masking'):   
                data_block = data_block.mask_values(mask = processor.mask)
            with Timer(f'{key} blocks saving '):
                data_block.save(key , predict , processor.save_start_dt , processor.save_end_dt)
            with Timer(f'{key} blocks norming'):
                data_block.hist_norm(key , predict , processor.hist_start_dt , processor.hist_end_dt)
            del data_block
            gc.collect()
            Logger.info(f'{key} finished! Cost {time.time() - tt1:.2f} Seconds')
            Logger.separator()

        Logger.info(f'Data Processing Finished! Cost {time.time() - t1:.2f} Seconds')

class TypePreProcessor(ABC):
    TRADE_FEAT : list[str] = ['open','close','high','low','vwap','turn_fl']

    @abstractmethod
    def block_loaders(self) -> dict[str,BlockLoader]: ... 
    @abstractmethod
    def final_feat(self) -> Optional[list]: ... 
    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: ...
        
    def load_blocks(self , start_dt = None , end_dt = None , secid_align = None , date_align = None , **kwargs):
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load_block(start_dt , end_dt , **kwargs).align(secid = secid_align , date = date_align)
            secid_align = blocks[src_key].secid
            date_align  = blocks[src_key].date
        return blocks
    
    def process_blocks(self, blocks : dict[str,DataBlock]):
        np.seterr(invalid = 'ignore' , divide = 'ignore')
        data_block = self.process(blocks)
        data_block = data_block.align_feature(self.final_feat())
        np.seterr(invalid = 'warn' , divide = 'warn')
        return data_block
    
def select_processor(key : str) -> TypePreProcessor:
    return getattr(inspect.getmodule(select_processor) , f'pp_{key.lower()}')()

class pp_y(TypePreProcessor):
    def block_loaders(self):
        return {'y' : BlockLoader('labels_ts', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'tushare_cne5_exp', [*CONF.RISK['indus'], 'size'])}
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
            if not isinstance(y_pro , torch.Tensor): continue
            y_pro = y_pro.unsqueeze(-1).numpy()
            data_block.values[...,i_feat] = y_pro

        return data_block
class pp_day(TypePreProcessor):
    def block_loaders(self): 
        return {'day' : BlockLoader('trade_ts', 'day', ['adjfactor', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): return blocks['day'].adjust_price()
    
class pp_15m(TypePreProcessor):
    def block_loaders(self): 
        return {'15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): 
        data_block = blocks['15m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class pp_30m(TypePreProcessor):
    def block_loaders(self): 
        return {'30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class pp_60m(TypePreProcessor):
    def block_loaders(self): 
        return {'60m' : BlockLoader('trade_ts', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): 
        data_block = blocks['60m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class pp_week(TypePreProcessor):
    WEEKDAYS = 5
    def block_loaders(self): 
        return {'day':BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def load_blocks(self , start_dt = None , end_dt = None , secid_align = None , date_align = None , **kwargs):
        if start_dt is not None and start_dt < 0: start_dt = 2 * start_dt
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load_block(start_dt , end_dt , **kwargs).align(secid = secid_align , date = date_align)
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
    def block_loaders(self): 
        return {'style' : BlockLoader('models', 'tushare_cne5_exp', CONF.RISK['style'])}
    def final_feat(self): return None
    def process(self , blocks): return blocks['style']

class pp_indus(TypePreProcessor):
    def block_loaders(self): 
        return {'indus' : BlockLoader('models', 'tushare_cne5_exp', CONF.RISK['indus'])}
    def final_feat(self): return None
    def process(self , blocks): return blocks['indus']

class FactorPreProcessor(TypePreProcessor):
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
    @classproperty_str
    def category1(cls): return cls.__qualname__.removeprefix('pp_').lower()
    def block_loaders(self): 
        return {'factor' : FactorLoader(self.category0 , self.category1)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['factor']

class pp_quality(FactorPreProcessor):
    category0 = 'fundamental'

class pp_growth(FactorPreProcessor):
    category0 = 'fundamental'

class pp_value(FactorPreProcessor):
    category0 = 'fundamental'

class pp_earning(FactorPreProcessor):
    category0 = 'fundamental'

class pp_surprise(FactorPreProcessor):
    category0 = 'analyst'

class pp_coverage(FactorPreProcessor):
    category0 = 'analyst'

class pp_forecast(FactorPreProcessor):
    category0 = 'analyst'

class pp_adjustment(FactorPreProcessor):
    category0 = 'analyst'

class pp_hf_momentum(FactorPreProcessor):
    category0 = 'high_frequency'

class pp_hf_volatility(FactorPreProcessor):
    category0 = 'high_frequency'

class pp_hf_correlation(FactorPreProcessor):
    category0 = 'high_frequency'

class pp_hf_liquidity(FactorPreProcessor):
    category0 = 'high_frequency'

class pp_momentum(FactorPreProcessor):
    category0 = 'behavior'

class pp_volatility(FactorPreProcessor):
    category0 = 'behavior'

class pp_correlation(FactorPreProcessor):
    category0 = 'behavior'

class pp_liquidity(FactorPreProcessor):
    category0 = 'behavior'

class pp_holding(FactorPreProcessor):
    category0 = 'money_flow'

class pp_trading(FactorPreProcessor):
    category0 = 'money_flow'
