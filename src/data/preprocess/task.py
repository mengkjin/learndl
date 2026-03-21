import argparse

from datetime import datetime
from typing import Any , Iterator

from src.proj import Logger , Duration , CALENDAR , Dates
from src.data.util import DataBlock

from . import preprocessor as PrePro

__all__ = ['PreProcessorTask']

DATASET_FIT = [
    'y' , 'day' , '30m' , 'style' , 'indus' , 'week' , 
    'quality' , 'growth' , 'value' , 'earning' , 
    'surprise' , 'coverage' , 'forecast' , 'adjustment' , 
    'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
    'momentum' , 'volatility' , 'correlation' , 'liquidity' , 
    'holding' , 'trading'
]
DATASET_PREDICT = DATASET_FIT
LOAD_OVERLAP_DAYS = 30

class PreProcessorTask:
    def __init__(self , predict : bool , keys : list[str] , mask : dict[str,Any] | None = None , **kwargs):
        self.predict = predict
        self.keys = [DataBlock.data_type_abbr(blk) for blk in keys]
        self.mask = mask or {'list_dt': 91}
        self.load_start = CALENDAR.td(CALENDAR.updated() , -366).td if predict else 20070101
        self.load_end   = None
        self.hist_start = None
        self.hist_end   = None if predict else 20161231

    def processors(self) -> Iterator[tuple[str , 'PrePro.BaseTypePreProcessor']]:
        for blk in self.keys:
            yield blk , getattr(PrePro , f'PrePro_{blk.lower()}')()
    
    @classmethod
    def main(cls , predict = False, confirm = 0 , * , parser = None , data_types : list[str] | None = None , indent : int = 0 , vb_level : int = 1 , 
             force_update : bool = False):
        if parser is None:
            parser = argparse.ArgumentParser(description = 'manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()
        if not predict and not args.confirm and \
            not input('Confirm update data? type "yes" to confirm!').lower()[0] == 'y' : 
            return
        
        if data_types is None:
            keys = DATASET_PREDICT if predict else DATASET_FIT
        else:
            keys = data_types
        processor = cls(predict , keys)
        Logger.note(f'Data PreProcessing start with {len(processor.keys)} datas and predict = {predict}!' , indent = indent , vb_level = vb_level)
        Logger.stdout(f'Will process {str(list(processor.keys))} at {Dates(processor.load_start,processor.load_end)}' , 
                        indent = indent + 1 , vb_level = vb_level + 1)

        for key , proc in processor.processors():
            modified_time = DataBlock.last_preprocess_time(key , predict)
            if not force_update and CALENDAR.is_updated_today(modified_time):
                time_str = datetime.strptime(str(modified_time) , '%Y%m%d%H%M%S').strftime("%Y-%m-%d %H:%M:%S")
                Logger.skipping(f'[{key.upper()}] already preprocessing at {time_str}!' , indent = indent + 1 , vb_level = vb_level + 1)
                continue

            tt1 = datetime.now()
            Logger.stdout(f'Preprocess [{key.upper()}] with predict={predict} start...' , indent = indent + 1 , vb_level = vb_level + 3)
            
            with Logger.Timer(f'[{key}] dumped loading' , indent = indent + 2 , vb_level = vb_level + 3):
                dumped_blocks = DataBlock.load_preprocess(key , predict)
                dumped_last_date = CALENDAR.td(dumped_blocks.date[-1] , -LOAD_OVERLAP_DAYS).td if not dumped_blocks.empty else -1

            with Logger.Timer(f'[{key}] blocks loading' , indent = indent + 2 , vb_level = vb_level + 3 , enter_vb_level = vb_level + 5):
                load_start = max(processor.load_start , dumped_last_date)
                block_dict = proc.load_blocks(load_start, processor.load_end, indent = indent + 2 , vb_level = vb_level + 5)

            with Logger.Timer(f'[{key}] blocks process' , indent = indent + 2 , vb_level = vb_level + 3):
                new_block = proc.process_blocks(block_dict).set_flags(category = 'preprocess' , predict = predict , preprocess_key = key)
            if new_block.empty:
                Logger.alert1(f'[{key}] blocks process is empty! Skip saving...' , indent = indent + 2 , vb_level = vb_level + 3)
                continue

            with Logger.Timer(f'[{key}] blocks masking' , indent = indent + 2 , vb_level = vb_level + 3):   
                new_block = new_block.mask_values(mask = processor.mask)

            with Logger.Timer(f'[{key}] blocks merging' , indent = indent + 2 , vb_level = vb_level + 3):
                data_block = dumped_blocks.merge_others(new_block , inplace = True)

            with Logger.Timer(f'[{key}] blocks dumping' , indent = indent + 2 , vb_level = vb_level + 3):
                data_block = data_block.align_date(data_block.date_within(processor.load_start , processor.load_end) , inplace = True)
                data_block.save_dump()

            with Logger.Timer(f'[{key}] blocks norming' , indent = indent + 2 , vb_level = vb_level + 3):
                data_block.hist_norm(key , predict , processor.hist_start , processor.hist_end)
            
            # gc.collect()
            Logger.success(f'Preprocess [{key.upper()}] (predict={predict},{Dates(data_block.date)}) finished! Cost {Duration(since = tt1)}' , 
                           indent = indent + 1 , vb_level = vb_level + 1)

            del data_block
            Logger.divider(vb_level = vb_level + 3)