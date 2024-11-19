from typing import Any , Literal
from src.data import DATAVENDOR

def random(start_dt = 20240101 , end_dt = 20240531 , step = 5 , nfactor = 2):
    return DATAVENDOR.random_factor(start_dt , end_dt , step , nfactor).to_dataframe()

def real(names : str | list[str] | Any ,
         factor_type : Literal['factor' , 'pred'] = 'factor' , 
         start_dt = 20240101 , end_dt = 20240531 , step = 5):
    return DATAVENDOR.real_factor(factor_type , names , start_dt , end_dt , step).to_dataframe()
