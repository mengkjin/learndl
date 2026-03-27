from typing import Any , Literal
from src.data import DATAVENDOR

def random(start = 20240101 , end = 20240531 , step = 5 , nfactor = 2):
    return DATAVENDOR.random_factor(start , end , step , nfactor).to_dataframe()

def real(names : str | list[str] | Any ,
         factor_type : Literal['factor' , 'pred'] = 'factor' , 
         start = 20240101 , end = 20240531 , step = 5):
    return DATAVENDOR.real_factor(factor_type , names , start , end , step).to_dataframe()
