import argparse , gc , time
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from torch import Tensor
from typing import Any , ClassVar , Iterator , Optional

from .core import DataBlock , data_type_abbr
from ..func.time import Timer , today
from ..func.primas import neutralize_2d , process_factor

@dataclass(slots=True)
class DataProcessor:
    predict         : bool 
    blocks          : list[str] = field(default_factory=list)
    load_start_dt   : Optional[int] = None
    load_end_dt     : Optional[int] = None
    save_start_dt   : Optional[int] = 20070101
    save_end_dt     : Optional[int] = None
    hist_start_dt   : Optional[int] = None
    hist_end_dt     : Optional[int] = 20161231
    mask            : dict[str,Any] = field(default_factory=dict)

    default_mask : ClassVar[dict[str,Any]] = {'list_dt': 91}

    def __post_init__(self):
        self.blocks = [data_type_abbr(blk) for blk in self.blocks]
        if self.predict:
            self.load_start_dt = today(-366)
            self.load_end_dt   = None
            self.save_start_dt = None
            self.save_end_dt   = None
            self.hist_start_dt = None
            self.hist_end_dt   = None
        if not self.mask: self.mask = self.default_mask

    def processors(self) -> Iterator[tuple[str , '_TypeProcessor']]:
        for blk in self.blocks:
            yield blk , self.default_type_processor(blk)

    @staticmethod
    def default_type_processor(blk : str) -> '_TypeProcessor':
        return {
            'y' : procLabel ,
            'day' : procDay ,
            '30m' : proc30min ,
            '15m' : proc15min ,
            'week' : procWeek ,
        }[blk]()
    
    @classmethod
    def main(cls , predict = False, confirm = 0 , parser = None , data_key = None):
        if parser is None:
            parser = argparse.ArgumentParser(description='manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()

        if not predict and not args.confirm and \
            not input('Confirm update data? print "yes" to confirm!').lower()[0] == 'y' : return

        t1 = time.time()
        print(f'predict is {predict} , Data Processing start!')

        processor = cls(predict , blocks = ['y' , 'trade_day' , 'trade_30m'])
        print(f'{len(processor.blocks)} datas :' + str(list(processor.blocks)))

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
    @abstractmethod
    def block_loaders(self) -> dict[str,'BlockLoader']: ... 
    @abstractmethod
    def final_feat(self) -> Optional[list]: ... 
    @abstractmethod
    def process(self, blocks : dict[str,DataBlock]) -> DataBlock: ...

    @dataclass(slots=True)
    class BlockLoader:
        db_src  : str
        db_key  : str | list
        feature : Optional[list] = None
        def load_block(self , start_dt = None , end_dt = None , **kwargs):
            sub_blocks = []
            db_keys = self.db_key if isinstance(self.db_key , list) else [self.db_key]
            for db_key in db_keys:
                with Timer(f' --> {self.db_src} blocks reading [{db_key}] DataBase\'s'):
                    blk = DataBlock.load_db(self.db_src , db_key , start_dt , end_dt , self.feature , **kwargs)
                    sub_blocks.append(blk)
            with Timer(f' --> {self.db_src} blocks merging'):
                sub_blocks = DataBlock.merge(sub_blocks)
            return sub_blocks
        
    def load_blocks(self , start_dt = None , end_dt = None , secid_align = None , date_align = None , **kwargs):
        blocks : dict[str,DataBlock] = {}
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load_block(start_dt , end_dt , **kwargs).align(secid = secid_align , date = date_align)
            secid_align = blocks[src_key].secid
            date_align  = blocks[src_key].date
        return blocks
    def process_blocks(self, blocks : dict[str,DataBlock]):
        np.seterr(invalid='ignore' , divide = 'ignore')
        data_block = self.process(blocks)
        data_block = data_block.align_feature(self.final_feat())
        np.seterr(invalid='warn' , divide = 'warn')
        return data_block

class procLabel(_TypeProcessor):
    def block_loaders(self):
        return {'labels': self.BlockLoader('labels',['ret10_lag','ret20_lag']) ,
                'models': self.BlockLoader('models','risk_exp')}
    def final_feat(self): return None
    def process(self , blocks : dict[str,DataBlock]): 
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

        return data_block
    
class procDay(_TypeProcessor):
    def block_loaders(self): 
        return {'day' : self.BlockLoader('trade','day',['adjfactor','close','high','low','open','vwap','turn_fl'])}
    def final_feat(self): return ['open','close','high','low','vwap','turn_fl']
    def process(self , blocks : dict[str,DataBlock]): 
        data_block = blocks['day']
        data_block = data_block.adjust_price()
        return data_block
    
class proc15min(_TypeProcessor):
    def block_loaders(self): 
        return {'15m' : self.BlockLoader('trade','15min',['close','high','low','open','volume','vwap']) ,
                'day' : self.BlockLoader('trade','day',['volume','turn_fl','preclose'])}
    def final_feat(self): return ['open','close','high','low','vwap','turn_fl']
    def process(self , blocks : dict[str,DataBlock]): 
        data_block = blocks['15m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide=db_day.loc(feature='preclose'))
        data_block = data_block.adjust_volume(divide=db_day.loc(feature='volume')/db_day.loc(feature='turn_fl'),vol_feat='volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class proc30min(_TypeProcessor):
    def block_loaders(self): 
        return {'30m' : self.BlockLoader('trade','30min',['close','high','low','open','volume','vwap']) ,            
                'day' : self.BlockLoader('trade','day',['volume','turn_fl','preclose'])}
    def final_feat(self): return ['open','close','high','low','vwap','turn_fl']

    def process(self , blocks : dict[str,DataBlock]): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide=db_day.loc(feature='preclose'))
        data_block = data_block.adjust_volume(divide=db_day.loc(feature='volume')/db_day.loc(feature='turn_fl'),vol_feat='volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class proc60min(_TypeProcessor):
    def block_loaders(self): 
        return {'60m' : self.BlockLoader('trade','60min',['close','high','low','open','volume','vwap']) ,            
                'day' : self.BlockLoader('trade','day',['volume','turn_fl','preclose'])}
    def final_feat(self): return ['open','close','high','low','vwap','turn_fl']
    def process(self , blocks : dict[str,DataBlock]): 
        data_block = blocks['60m']
        db_day     = blocks['day'].align(secid = data_block.secid , date = data_block.date)
        
        data_block = data_block.adjust_price(divide=db_day.loc(feature='preclose'))
        data_block = data_block.adjust_volume(divide=db_day.loc(feature='volume')/db_day.loc(feature='turn_fl'),vol_feat='volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class procWeek(_TypeProcessor):
    def block_loaders(self): 
        return {'day':self.BlockLoader('trade','day',['adjfactor','preclose','close','high','low','open','vwap','turn_fl'])}
    def final_feat(self): return ['open','close','high','low','vwap','turn_fl']
    def process(self , blocks : dict[str,DataBlock]): 
        num_days   = 5
        data_block = blocks['day'].adjust_price()

        new_values = np.full(np.multiply(data_block.shape,(1,1,num_days,1)),np.nan)
        for i in range(num_days): new_values[:,num_days-1-i:,i] = data_block.values[:,:len(data_block.date)-num_days+1+i,0]
        data_block.update(values = new_values)
        data_block = data_block.adjust_price(adjfactor = False , divide=data_block.loc(inday=0,feature='preclose'))
        return data_block
    