import torch
import numpy as np
import pandas as pd

from ...environ import PATH , CONF
from ..core import GetData

from src.data import DataBlock
from typing import Literal
from copy import deepcopy
import torch
import torch.nn.functional as F

class DataVendor:
    def __init__(self):
        pass

    def get_returns(self , start_dt : int , end_dt : int):
        self.day_ret = GetData.daily_returns(start_dt , end_dt)

    def nday_fut_ret(self , nday : int = 10 , lag : Literal[1,2] = 2 , step : int = 5 , retain_shape = False ,
                    auto_rename = True):
        assert lag > 0 , f'lag must be positive : {lag}'
        block = deepcopy(self.day_ret).as_tensor()
        block.values = F.pad(block.values[:,lag:] , (0,0,0,0,0,lag),value=torch.nan)

        new_value = block.values.unfold(1 , nday , step).exp().prod(dim = -1) - 1
        new_date  = block.date[::step][:new_value.shape[1]]
        secid     = block.secid
        feature   = block.feature

        if auto_rename: feature = ['ret']
        new_block = DataBlock(new_value , secid , new_date , feature)
        if retain_shape: new_block.align_date(block.date)
        return new_block