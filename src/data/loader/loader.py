import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional

from src.proj import PATH
from src.basic import Timer , DB
from src.data.util import DataBlock

@dataclass(slots=True)
class BlockLoader:
    db_src  : str
    db_key  : str | list
    feature : Optional[list] = None
    use_alt : bool = True

    def __post_init__(self):
        src_path = PATH.database.joinpath(f'DB_{self.db_src}')
        assert src_path.exists() , f'{src_path} not exists'
        assert np.isin(self.db_key , [p.name for p in src_path.iterdir()]).all() , f'{self.db_key} not all in {src_path}'

    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        return self.load_block(start_dt , end_dt)
    
    def load_block(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        sub_blocks = []
        db_keys = self.db_key if isinstance(self.db_key , list) else [self.db_key]
        for db_key in db_keys:
            with Timer(f' --> {self.db_src} blocks reading [{db_key}] DataBase'):
                blk = DataBlock.load_db(self.db_src , db_key , start_dt , end_dt , 
                                        feature = self.feature , use_alt = self.use_alt)
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
    use_alt : bool = True

    def __post_init__(self):
        assert PATH.database.joinpath(f'DB_{self.db_src}' , self.db_key).exists() , \
            f'{PATH.database}/{self.db_src}/{self.db_key} not exists'
    
    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        return self.load_frame(start_dt , end_dt)

    def load_frame(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        df = DB.db_load_multi(self.db_src , self.db_key , start_dt=start_dt , end_dt=end_dt , use_alt = self.use_alt)
        return df
    
@dataclass
class FactorLoader:
    category0  : str
    category1  : str

    def __post_init__(self):
        from src.res.factor.calculator import StockFactorHierarchy
        self.hier = StockFactorHierarchy()
        self.hier.validate_category(self.category0 , self.category1)
    
    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        return self.load_block(start_dt , end_dt)

    def load_block(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        factors : list[pd.DataFrame] = []
        with Timer(f' --> factor blocks reading [{self.category0} , {self.category1}]'):
            for calc in self.hier.iter_instance(category0 = self.category0 , category1 = self.category1):
                df = calc.Loads(start_dt , end_dt)
                df = df.rename(columns = {calc.factor_name:'value'}).assign(feature = calc.factor_name)
                factors.append(df)
        with Timer(f' --> factor blocks merging ({len(factors)} factors)'): 
            df = pd.concat(factors).pivot_table('value' , ['secid','date'] , 'feature')
            block = DataBlock.from_dataframe(df)
        return block