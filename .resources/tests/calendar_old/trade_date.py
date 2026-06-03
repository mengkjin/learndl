import torch
import numpy as np
import pandas as pd

from typing import Any
from .basic import BC

class TradeDate:
    def __new__(cls , date : int | Any , *args , **kwargs):
        if isinstance(date , TradeDate):
            return date
        return super().__new__(cls)
    def __init__(self , date : int | Any , force_trade_date = False):
        if not isinstance(date , TradeDate):
            self.cd = int(date)
            if force_trade_date or self.cd < BC.min_date or self.cd > BC.max_date:
                self.td : int = self.cd 
            else:
                self.td = BC.full['td'].loc[self.cd]

    def __repr__(self):
        return str(self.td)

    def __int__(self): 
        return int(self.td)
    
    def __str__(self):
        return str(self.td)
    
    def __add__(self , n : int):
        return self.offset(n)
    
    def __sub__(self , n : int):
        return self.offset(-n)
    
    def __lt__(self , other):
        return int(self) < int(other)
    
    def __le__(self , other):
        return int(self) <= int(other)
    
    def __gt__(self , other):
        return int(self) > int(other)
    
    def __ge__(self , other):
        return int(self) >= int(other)
    
    def __eq__(self , other):
        return int(self) == int(other)

    def as_int(self):
        return int(self)
    
    def offset(self , n : int):
        return self._cls_offset(self , n)

    @classmethod
    def _cls_offset(cls , td0 , n : int):
        td0 = cls(td0)
        if n == 0: 
            return td0
        elif td0 < BC.min_date or td0 > BC.max_date:
            return td0
        assert isinstance(n , (int , np.integer)) , f'n must be a integer, got {type(n)}'
        d_index = BC.full['td_index'].loc[td0.td] + n
        d_index = np.maximum(np.minimum(d_index , BC.max_td_index) , 0)
        new_date = BC.trd.loc[d_index,'td']
        return cls(new_date) 

    @staticmethod
    def as_numpy(td):
        if isinstance(td , int): 
            td = np.array([td])
        elif isinstance(td , pd.Series): 
            td = td.to_numpy()
        elif isinstance(td , list): 
            td = np.array(td)
        elif isinstance(td , torch.Tensor): 
            td = td.cpu().numpy()
        return td.astype(int)
