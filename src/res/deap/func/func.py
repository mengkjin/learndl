import pandas as pd
import torch
from pathlib import Path
from typing import Callable

from . import math_func as MF
from ..env import gpDefaults

def gp_labels_raw(CP = None , neutral_factor = None , neutral_group = None , nday = 10 , delay = 1 , 
                  slice_date = None, df_columns = None , device = None) -> torch.Tensor:
    '''
    ------------------------ gp labels raw ------------------------
    
    生成原始预测标签,中性化后的10日收益
    input:
        CP:             calculated close price
        neutral_factor: size , for instance
        neutral_group:  indus , for instance
    output:
        labels_raw:     
    '''
    if CP is None:
        CP = df2ts(read_gp_data(gp_filename_converter()('CP'),slice_date,df_columns) , 'CP' , device)    
    labels = MF.ts_delay(MF.pctchg(CP, nday) , -nday-delay)  # t+1至t+11的收益率
    neutral_x = MF.neutralize_xdata_2d(neutral_factor, neutral_group)
    labels_raw = MF.neutralize_2d(labels, neutral_x , inplace=True)  # 市值行业中性化
    return labels_raw

def gp_filename_converter() -> Callable[[str], Path]:
    '''
    ------------------------ gp input data filenames ------------------------
    原始因子名与文件名映射
    input:
    output:
        wrapper: function to convert gp_key into parquet filename 
    '''
    filename_dict = {'op':'open','hp':'high','lp':'low','vp':'vwap','cp':'close_adj',
                     'vol':'volume','bp':'bp_lf','ep':'ep_ttm','ocfp':'ocfp_ttm',
                     'dp':'dividendyield2','rtn':'return1','indus':'cs_indus_code'}
    def wrapper(gp_key : str) -> Path:
        assert gp_key.isupper() or gp_key.islower() , gp_key

        rawkey = gp_key.lower()
        if rawkey in filename_dict.keys(): 
            rawkey = filename_dict[rawkey]

        zscore = gp_key.islower() and rawkey not in ['cs_indus_code' , 'size']

        return gpDefaults.dir_data.joinpath(f'{rawkey}' + '_zscore' * zscore + '_day.parquet')
    return wrapper

def read_gp_data(filename : Path | str , slice_date=None,df_columns=None,df_index=None,input_freq='D') -> pd.DataFrame:
    '''
    ------------------------ read gp data and convert to torch.tensor ------------------------
    
    读取单个原始因子文件并转化成tensor,额外返回df表格的行列字典
    input:
        filename:    filename gp data
        slice_date:  [insample_start, insample_end, outsample_start, outsample_end]
        df_columns:  if not None, filter columns
        df_index     if not None, filter rows, cannot be used if slice_date given
        input_freq:  freq faster than day should code later
    output:
        df:          result data
    '''
    df = pd.read_parquet(filename, engine='fastparquet')
    assert isinstance(df , pd.DataFrame) , f'{type(df)} is not a DataFrame'
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: 
        df = df[(slice_date[0] <= df.index.values) & (df.index.values <= slice_date[-1])] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    if df_columns is not None: 
        df = df.loc[:,df_columns]# 选取指定股票
    if slice_date is None and df_index is not None: 
        df = df.loc[df_index]
    return df if isinstance(df , pd.DataFrame) else df.to_frame()

def df2ts(x : pd.DataFrame | torch.Tensor , gp_key = '' , device = None , share_memory = True) -> torch.Tensor:
    # additional treatment based by gp_key
    if isinstance(x , pd.DataFrame): 
        x = torch.FloatTensor(x.values)
    if gp_key == 'DP': # raw dividend factor , nan means 0
        x.nan_to_num_()
    if isinstance(x , torch.Tensor):
        if device is not None: 
            x = x.to(device)
        if share_memory: 
            x.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return x
