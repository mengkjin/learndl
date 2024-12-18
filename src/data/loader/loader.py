import numpy as np

from dataclasses import dataclass
from typing import Optional

from src.basic import CALENDAR , PATH , Timer
from src.data.util import DataBlock


@dataclass(slots=True)
class BlockLoader:
    db_src  : str
    db_key  : str | list
    feature : Optional[list] = None

    def __post_init__(self):
        assert f'DB_{self.db_src}' in [p.name for p in PATH.database.iterdir()] , f'DB_{self.db_src} not in {PATH.database}'
        src_path = PATH.database.joinpath(f'DB_{self.db_src}')
        assert np.isin(self.db_key , [p.name for p in src_path.iterdir()]).all() , f'{self.db_key} not all in {src_path}'

    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        return self.load_block(start_dt , end_dt)
    
    def load_block(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        if end_dt is not None   and end_dt < 0:   end_dt   = CALENDAR.today(end_dt)
        if start_dt is not None and start_dt < 0: start_dt = CALENDAR.today(start_dt)

        sub_blocks = []
        db_keys = self.db_key if isinstance(self.db_key , list) else [self.db_key]
        for db_key in db_keys:
            with Timer(f' --> {self.db_src} blocks reading [{db_key}] DataBase'):
                blk = DataBlock.load_db(self.db_src , db_key , start_dt , end_dt , feature = self.feature)
                sub_blocks.append(blk)
        if len(sub_blocks) <= 1:  
            block = sub_blocks[0]
        else:
            with Timer(f' --> {self.db_src} blocks merging ({len(sub_blocks)})'): 
                block = DataBlock.merge(sub_blocks)
        return block

@dataclass(slots=True)
class FrameLoader:
    db_src  : str
    db_key  : str
    reserved_src : Optional[list[str]] = None

    def __post_init__(self):
        assert PATH.database.joinpath(f'DB_{self.db_src}' , self.db_key).exists() , \
            f'{PATH.database}/{self.db_src}/{self.db_key} not exists'
    
    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        return self.load_frame(start_dt , end_dt)

    def load_frame(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        df = PATH.db_load_multi(self.db_src , self.db_key , start_dt=start_dt , end_dt=end_dt , date_colname = 'date')
        return df