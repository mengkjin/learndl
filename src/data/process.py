import argparse , gc , inspect, time
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from torch import Tensor
from typing import Any , Iterator , Optional

from .core import DataBlock , data_type_abbr
from .loader import BlockLoader
from ..basic import CONF
from ..basic.util import Timer
from ..func.primas import neutralize_2d , process_factor

TRAIN_DATASET = ['y' , 'day' , '30m' , 'style' , 'indus' , 'week']
PREDICT_DATASET = ['y' , 'day' , '30m' , 'style' , 'indus' , 'week']

@dataclass(slots=True)
class DataProcessor:
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
        self.blocks = [data_type_abbr(blk) for blk in self.blocks]
        if self.predict:
            self.load_start_dt = -366
        else:
            self.save_start_dt = 20070101
            self.hist_end_dt   = 20161231
        if not self.mask: self.mask = {'list_dt': 91}

    def processors(self) -> Iterator[tuple[str , '_TypeProcessor']]:
        for blk in self.blocks:
            yield blk , select_processor(blk)
    
    @classmethod
    def main(cls , predict = False, confirm = 0 , parser = None , data_types : Optional[list[str]] = None):
        if parser is None:
            parser = argparse.ArgumentParser(description = 'manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()
        if not predict and not args.confirm and \
            not input('Confirm update data? print "yes" to confirm!').lower()[0] == 'y' : return
        t1 = time.time()
        print(f'predict is {predict} , Data Processing start!')
        
        if data_types is None:
            blocks = PREDICT_DATASET if predict else TRAIN_DATASET
        else:
            blocks = data_types
        processor = cls(predict , blocks = blocks)
        print(f'{len(processor.blocks)} datas :' + str(list(processor.blocks)))
        # return processor
        for key , proc in processor.processors():
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
            print(f'{key} finished! Cost {time.time() - tt1:.2f} Seconds')
            print('-' * 80)

        print(f'Data Processing Finished! Cost {time.time() - t1:.2f} Seconds')

class _TypeProcessor(ABC):
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
    
def select_processor(key : str) -> _TypeProcessor:
    return getattr(inspect.getmodule(select_processor) , f'proc{key.capitalize()}')()

class procY(_TypeProcessor):
    def block_loaders(self):
        return {'y' : BlockLoader('labels', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'risk_exp', [*CONF.RISK_INDUS, 'size'])}
    def final_feat(self): return None
    def process(self , blocks : dict[str,DataBlock]): 
        data_block , model_exp = blocks['y'] , blocks['risk']
        indus_size = model_exp.values[...,:]
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

        return data_block
class procDay(_TypeProcessor):
    def block_loaders(self): 
        return {'day' : BlockLoader('trade', 'day', ['adjfactor', *self.TRADE_FEAT])}
    def final_feat(self): return self.TRADE_FEAT
    def process(self , blocks): return blocks['day'].adjust_price()
    
class proc15m(_TypeProcessor):
    def block_loaders(self): 
        return {'15m' : BlockLoader('trade', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade', 'day', ['volume', 'turn_fl', 'preclose'])}
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
    
class proc30m(_TypeProcessor):
    def block_loaders(self): 
        return {'30m' : BlockLoader('trade', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade', 'day', ['volume', 'turn_fl', 'preclose'])}
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
    
class proc60m(_TypeProcessor):
    def block_loaders(self): 
        return {'60m' : BlockLoader('trade', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,            
                'day' : BlockLoader('trade', 'day', ['volume', 'turn_fl', 'preclose'])}
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
    
class procWeek(_TypeProcessor):
    WEEKDAYS = 5
    def block_loaders(self): 
        return {'day':BlockLoader('trade', 'day', ['adjfactor', 'preclose', *self.TRADE_FEAT])}
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
    
class procStyle(_TypeProcessor):
    def block_loaders(self): 
        return {'style' : BlockLoader('models', 'risk_exp', CONF.RISK_STYLE)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['style']

class procIndus(_TypeProcessor):
    def block_loaders(self): 
        return {'indus' : BlockLoader('models', 'risk_exp', CONF.RISK_INDUS)}
    def final_feat(self): return None
    def process(self , blocks): return blocks['indus']