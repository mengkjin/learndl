import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.func.transform import winsorize , whiten

from src.res.factor.calculator import StockFactorCalculator
from src.res.facdef.level0.fundamental.valuation_static import (btop , btop_rank3y , etop , etop_rank3y , cfev , cfev_rank3y)
from src.res.facdef.level0.fundamental.valuation_dynamic import btop_stability , etop_stability , cfev_stability

def btop_augment_range(date : int):
    bp_raw = btop.Eval(date)

    fixed_asset_ratio  = DATAVENDOR.get_fin_latest('bs@fix_assets@qtr / ta@qtr' , date)
    capital_intensity  = DATAVENDOR.get_fin_latest('ta@qtr / sales@qtr' , date)
    liability_ratio    = DATAVENDOR.get_fin_latest('liab@qtr / ta@qtr' , date)
    inventory_turnover = -DATAVENDOR.get_fin_latest('sales@qtr / bs@inventories@qtr' , date)

    # score = lambda x : x.rank(pct = True)
    heavy_asset = pd.concat([fixed_asset_ratio , capital_intensity , liability_ratio , inventory_turnover] , axis = 1)
    heavy_asset = heavy_asset.rank(pct = True).mean(axis = 1)
    assert isinstance(heavy_asset , pd.Series) , f'heavy_asset must be a pandas series, but got {type(heavy_asset)}'
    heavy_asset = heavy_asset.rename('heavy_asset').rank(pct = True).reindex(bp_raw.index)

    bp_range = (bp_raw < 1 / 0.8) & (heavy_asset >= 0.5)
    return bp_range

def etop_augment_range(date : int):
    dedt = (DATAVENDOR.get_fin_hist('dedt@ttm' , date , 9) >= 0).groupby('secid').all()
    npro = (DATAVENDOR.get_fin_hist('npro@ttm' , date , 9) >= 0).groupby('secid').all()
    total_np = (DATAVENDOR.get_fin_hist('total_np@ttm' , date , 9) >= 0).groupby('secid').all()
    sales = (DATAVENDOR.get_fin_hist('sales@ttm' , date , 9) >= 3e8).groupby('secid').all()

    ep_range = pd.concat([dedt , npro , total_np , sales] , axis = 1).all(axis = 1)

    return ep_range

def cfev_augment_range(date : int):
    val = cfev.Eval(date)
    cf_range = (val > 0)
    return cf_range

class btop_augment(StockFactorCalculator):
    init_date = 20110101
    category1 = 'value'
    description = 'btop增强因子'
    
    def calc_factor(self, date: int):
        x_range = btop_augment_range(date)
        v = pd.concat([whiten(winsorize(calc.Eval(date).where(x_range , np.nan))) 
                       for calc in [btop , btop_rank3y , btop_stability]] , axis = 1).mean(axis = 1)
        return v
    
class etop_augment(StockFactorCalculator):
    init_date = 20110101
    category1 = 'value'
    description = 'etop增强因子'
    
    def calc_factor(self, date: int):
        x_range = etop_augment_range(date)
        v = pd.concat([whiten(winsorize(calc.Eval(date).where(x_range , np.nan))) 
                       for calc in [etop , etop_rank3y , etop_stability]] , axis = 1).mean(axis = 1)
        return v
    
class cfev_augment(StockFactorCalculator):
    init_date = 20110101
    category1 = 'value'
    description = 'cfev增强因子'
    
    def calc_factor(self, date: int):
        x_range = cfev_augment_range(date)
        v = pd.concat([whiten(winsorize(calc.Eval(date).where(x_range , np.nan))) 
                       for calc in [cfev , cfev_rank3y , cfev_stability]] , axis = 1).mean(axis = 1)
        return v
    
class valuation_augment(StockFactorCalculator):
    init_date = 20110101
    category1 = 'value'
    description = '估值增强因子'
    
    def calc_factor(self, date: int):
        bp  = btop_augment.Eval(date) 
        ep  = etop_augment.Eval(date)
        cfp = cfev_augment.Eval(date)
        if any(f.empty or f.isna().all() for f in [bp , ep , cfp]): 
            return pd.Series()
        v = pd.concat([whiten(winsorize(calc)) for calc in [bp , ep , cfp]] , axis = 1).mean(axis = 1)
        return v
    