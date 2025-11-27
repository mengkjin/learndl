import pandas as pd

from dataclasses import dataclass
from typing import Literal

from src.proj import PATH , Timer
from src.basic import DB , CONF , CALENDAR
from src.data.util import DataBlock

@dataclass
class BlockLoader:
    """
    Loader for block data of db_src , db_key
    example:
        loader = BlockLoader(db_src = 'trade_ts' , db_key = 'day')
        block = loader.load(start_dt = 20250101 , end_dt = 20250331)
    """
    db_src  : str
    db_key  : str | list | None = None
    feature : list | None = None
    use_alt : bool = True

    def __post_init__(self):
        assert self.src_path.exists() , f'{self.src_path} not exists'
        for key in self.iter_keys():
            assert self.src_path.joinpath(key).exists() , f'{key} not exists in {self.src_path}'

    @property
    def src_path(self):
        return DB.src_path(self.db_src)

    def load(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> DataBlock:
        """Load block data , alias for load_block"""
        return self.load_block(start_dt , end_dt , silent = silent)
    
    def load_block(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> DataBlock:
        """Load block data , alias for load"""
        sub_blocks = []
        for db_key in self.iter_keys():
            with Timer(f' --> {self.db_src} blocks reading [{db_key}] DataBase' , silent = silent):
                blk = DataBlock.load_db(self.db_src , db_key , start_dt , end_dt , 
                                        feature = self.feature , use_alt = self.use_alt)
                sub_blocks.append(blk)
        if len(sub_blocks) <= 1:  
            block = sub_blocks[0]
        else:
            with Timer(f' --> {self.db_src} blocks merging ({len(sub_blocks)})' , silent = silent): 
                block = DataBlock.merge(sub_blocks)
        return block

    def iter_keys(self) -> list[str]:
        if self.db_key is None:
            return []
        elif isinstance(self.db_key , list):
            return self.db_key
        else:
            return [self.db_key]
@dataclass
class FrameLoader:
    """
    Loader for frame data of db_src , db_key
    example:
        loader = FrameLoader(db_src = 'trade_ts' , db_key = 'day')
        df = loader.load(start_dt = 20250101 , end_dt = 20250331)
    """
    db_src  : str
    db_key  : str
    reserved_src : list[str] | None = None
    use_alt : bool = True

    def __post_init__(self):
        assert PATH.database.joinpath(f'DB_{self.db_src}' , self.db_key).exists() , \
            f'{PATH.database}/{self.db_src}/{self.db_key} not exists'
    
    def load(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> pd.DataFrame:
        """Load frame data , alias for load_frame"""
        return self.load_frame(start_dt , end_dt , silent = silent)

    def load_frame(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> pd.DataFrame:
        """Load frame data , alias for load"""
        df = DB.load_multi(self.db_src , self.db_key , start_dt=start_dt , end_dt=end_dt , use_alt = self.use_alt)
        return df
    
class FactorLoader(BlockLoader):
    """
    Loader for factor data given factor names
    example:
        loader = FactorLoader(name = ['df_scores_v0'])
        df = loader.load(start_dt = 20250101 , end_dt = 20250331)
    """
    def __init__(
        self , 
        names : str | list[str] , 
        normalize = False , 
        fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
        **kwargs
    ):
        super().__init__('factor')
        self.names = names if isinstance(names , list) else [names]
        self.normalize = normalize
        self.fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = fill_method
        self.kwargs = kwargs
        
    def load_block(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> DataBlock:
        """Load factor data , alias for load"""
        factors : list[pd.DataFrame] = []
        from src.res.factor.calculator import FactorCalculator
        with Timer(f' --> factor blocks reading [{len(self.names)} factors]' , silent = silent):
            dates = CALENDAR.td_within(start_dt , end_dt)
            for calc in FactorCalculator.iter(selected_factors = self.names , **self.kwargs):
                df = calc.Loads(dates , normalize = self.normalize , fill_method = self.fill_method)
                df = df.rename(columns = {calc.factor_name:'value'}).assign(feature = calc.factor_name)
                factors.append(df)
        with Timer(f' --> factor blocks merging ({len(factors)} factors)' , silent = silent): 
            assert len([fac for fac in factors if not fac.empty]) > 0 , f'no factors found for {self.names}'
            df = pd.concat([fac for fac in factors if not fac.empty]).pivot_table('value' , ['secid','date'] , 'feature')
            block = DataBlock.from_dataframe(df)
        return block

class FactorCategory1Loader(BlockLoader):
    """
    Loader for factor data given category1
    example:
        loader = FactorCategory1Loader(category1 = 'quality')
        df = loader.load(start_dt = 20250101 , end_dt = 20250331)
    """
    def __init__(
        self , 
        category1 : str , 
        normalize = False , 
        fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
        **kwargs
    ):
        super().__init__('factor')
        self.category1 = category1
        self.normalize = normalize
        self.fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = fill_method
        self.kwargs = kwargs
        
    def load_block(self , start_dt : int | None = None , end_dt : int | None = None , silent = False) -> DataBlock:
        """Load factor data , alias for load"""
        factors : list[pd.DataFrame] = []
        from src.res.factor.calculator import FactorCalculator
        with Timer(f' --> factor blocks reading [{self.category0} , {self.category1}]' , silent = silent):
            dates = CALENDAR.td_within(start_dt , end_dt)
            for calc in FactorCalculator.iter(category0 = self.category0 , category1 = self.category1 , **self.kwargs):
                df = calc.Loads(dates , normalize = self.normalize , fill_method = self.fill_method)
                df = df.rename(columns = {calc.factor_name:'value'}).assign(feature = calc.factor_name)
                factors.append(df)
        with Timer(f' --> factor blocks merging ({len(factors)} factors)' , silent = silent): 
            assert len([fac for fac in factors if not fac.empty]) > 0 , f'no factors found for {self.category0} , {self.category1}'
            df = pd.concat([fac for fac in factors if not fac.empty]).pivot_table('value' , ['secid','date'] , 'feature')
            block = DataBlock.from_dataframe(df)
        return block

    @property
    def category0(self) -> str:
        """Get the category0 of the factor"""
        return CONF.Factor.STOCK.cat1_to_cat0(self.category1)