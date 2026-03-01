import copy
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from typing import Any , Callable

from src.proj import Logger
from src.proj.func import torch_load
from src.res.deap.param import gpDefaults , gpParameters
from src.res.deap.func import math_func as MF , factor_func as FF
from .logger import gpLogger
from .status import gpStatus

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
    labels = MF.ts_delay(MF.pctchg(CP, nday) , -nday-delay , no_alert = True)  # t+1至t+11的收益率
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
    if df_columns is not None and len(df_columns) > 0: 
        df = df.loc[:,df_columns]# 选取指定股票
    if slice_date is None and df_index is not None and len(df_index) > 0: 
        df = df.loc[df_index]
    return df if isinstance(df , pd.DataFrame) else df.to_frame()

def df2ts(x : pd.DataFrame | torch.Tensor , gp_key = '' , device : torch.device | None = None , share_memory = True) -> torch.Tensor:
    # additional treatment based by gp_key
    if isinstance(x , pd.DataFrame): 
        x = torch.FloatTensor(x.values)
    if gp_key == 'DP': # raw dividend factor , nan means 0
        x.nan_to_num_()
    if isinstance(x , torch.Tensor):
        if device is not None: 
            x = x.to(device)
        if share_memory and (device is None or device.type == 'cpu'): 
            x.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return x

class gpInput:
    """遗传规划输入,包括参数、输入、输出、文件管理、内存管理、计时器、评价器、数据列"""
    def __init__(self , param : gpParameters | None = None , status : gpStatus | None = None , **kwargs):
        self.param     = param if param is not None else gpParameters(test_code = True)
        self.status    = status if status is not None else gpStatus(0 , 0)
        self.inputs    = []
        self.tensors :   dict[str , torch.Tensor] = {}
        self.records :   dict[str , Any] = {}
        
        self.logger   = gpLogger()

    def __repr__(self):
        return f'{self.__class__.__name__}(inputs={len(self.inputs)}, tensors={len(self.tensors)}, records={len(self.records)})'

    @property
    def i_iter(self) -> int:
        return self.status.i_iter

    @property
    def device(self):
        return self.param.device
    @property
    def gp_argnames(self):
        return self.param.gp_argnames
    @property
    def n_args(self):
        return self.param.n_args
    @property
    def df_index(self) -> np.ndarray:
        return self.records.get('df_index' , np.array([]))
    @property
    def df_columns(self) -> np.ndarray:
        return self.records.get('df_columns' , np.array([]))
    @property
    def size(self) -> torch.Tensor:
        return self.tensors['size']
    @property
    def indus(self) -> torch.Tensor:
        return self.tensors['indus']
    @property
    def labels_raw(self) -> torch.Tensor:
        return self.tensors['labels_raw']
    @property
    def labels_res(self) -> torch.Tensor:
        return self.tensors['labels_res']
    @property
    def universe(self) -> torch.Tensor:
        return self.tensors['universe']

    def load_data(self):
        package_path = gpDefaults.dir_pack.joinpath(f'gp_data_package' + '_test' * self.param.test_code + '.pt')
        package_require = ['gp_argnames' , 'inputs' , 'size' , 'indus' , 'labels_raw' , 'df_index' , 'df_columns' , 'universe']

        load_finished = False
        package_data = torch_load(package_path , map_location = self.device) if package_path.exists() else {}

        if not np.isin(package_require , list(package_data.keys())).all() or not np.isin(self.gp_argnames , package_data['gp_argnames']).all():
            if self.param.show_progress: 
                Logger.stdout(f'Exists "{package_path}" but Lack Required Data!' , indent = 1)
                Logger.stdout(f'Required: {package_require}' , indent = 1)
                Logger.stdout(f'Available: {list(package_data.keys())}' , indent = 1)
        else:
            assert np.isin(package_require , list(package_data.keys())).all() , np.setdiff1d(package_require , list(package_data.keys()))
            assert np.isin(self.gp_argnames , package_data['gp_argnames']).all() , np.setdiff1d(self.gp_argnames , package_data['gp_argnames'])
            assert package_data['df_index'] is not None

            if self.param.show_progress: 
                Logger.stdout(f'Directly load "{package_path}"' , indent = 1)
            for gp_key in self.gp_argnames:
                gp_val = package_data['inputs'][package_data['gp_argnames'].index(gp_key)]
                gp_val = df2ts(gp_val , gp_key , self.device)
                self.inputs.append(gp_val)

            for gp_key in ['size' , 'indus' , 'labels_raw' , 'universe']: 
                gp_val = package_data[gp_key]
                gp_val = df2ts(gp_val , gp_key , self.device)
                self.tensors[gp_key] = gp_val

            for gp_key in ['df_index' , 'df_columns']: 
                gp_val = package_data[gp_key]
                self.records[gp_key] = gp_val

            load_finished = True

        if not load_finished:
            if self.param.show_progress: 
                Logger.stdout(f'Load from Parquet Files:' , indent = 1)
            gp_filename = gp_filename_converter()
            nrowchar = 0
            for i , gp_key in enumerate(self.gp_argnames):
                if self.param.show_progress and nrowchar == 0: 
                    Logger.stdout('' , end='', indent = 1)
                gp_val = read_gp_data(gp_filename(gp_key),self.param.slice_date,self.df_columns)
                if i == 0: 
                    self.records['df_columns'] = gp_val.columns.values
                    self.records['df_index'] = gp_val.index.values
                gp_val = df2ts(gp_val , gp_key , self.device)
                self.inputs.append(gp_val)
                
                if self.param.show_progress:
                    Logger.stdout(gp_key , end=',')
                    nrowchar += len(gp_key) + 1
                    if nrowchar >= 100 or i == len(self.gp_argnames):
                        Logger.stdout()
                        nrowchar = 0

            for gp_key in ['size' , 'indus']: 
                gp_val = read_gp_data(gp_filename(gp_key),self.param.slice_date,self.df_columns)
                gp_val = df2ts(gp_val , gp_key , self.device)
                self.tensors[gp_key] = gp_val

            if 'CP' in self.gp_argnames:
                CP = self.inputs[self.gp_argnames.index('CP')]      
            else:
                CP = df2ts(read_gp_data(gp_filename('CP'),self.param.slice_date,self.df_columns) , 'CP' , self.device)    
            self.tensors['universe']   = ~CP.isnan() 
            self.tensors['labels_raw'] = gp_labels_raw(CP , self.tensors['size'] , self.tensors['indus'])
            gpDefaults.dir_pack.mkdir(parents=True, exist_ok=True)
            saved_data = {
                'gp_argnames' : self.gp_argnames ,
                'inputs' : self.inputs ,
                'size' : self.tensors['size'] ,
                'indus' : self.tensors['indus'] ,
                'labels_raw' : self.tensors['labels_raw'] ,
                'universe' : self.tensors['universe'] ,
                'df_index' : self.df_index ,
                'df_columns' : self.df_columns ,
            }
            torch.save(saved_data , package_path)
            Logger.stdout(f'Package data saved to "{package_path}"' , indent = 1)

        self.insample  = torch.Tensor((self.df_index >= self.param.slice_date[0]) * 
                                      (self.df_index <= self.param.slice_date[1])).bool()
        self.outsample = torch.Tensor((self.df_index >= self.param.slice_date[2]) * 
                                      (self.df_index <= self.param.slice_date[3])).bool()
        if self.param.factor_neut_type == 1:
            self.insample_2d = self.insample.reshape(-1,1).expand(self.tensors['labels_raw'].shape)

        self.logger.save_state(self.param.params , 'params', i_iter = 0) # useful to assert same index as package data
        self.logger.save_state({'df_index' : self.df_index , 'df_columns' : self.df_columns},'df_axis' , i_iter = 0) # useful to assert same index as package data

    def update_residual(self , **kwargs):
        """计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化"""
        assert self.param.labels_neut_type in ['svd' , 'all'] , self.param.labels_neut_type #  'all'
        assert self.param.svd_mat_method in ['coef_ts' , 'total'] , self.param.svd_mat_method

        self.neutra = None
        if self.i_iter == 0:
            labels_res = copy.deepcopy(self.tensors['labels_raw'])
            elites     = None
        else:
            labels_res = self.logger.load_state('res' , self.i_iter - 1 , device = self.device)
            elites     = self.logger.load_state('elt' , self.i_iter - 1 , device = self.device)
        neutra = elites

        if isinstance(elites , torch.Tensor) and self.param.labels_neut_type == 'svd': 
            assert isinstance(labels_res , torch.Tensor) , type(labels_res)
            if self.param.svd_mat_method == 'total':
                elites_mat = FF.factor_coef_total(elites[self.insample],dim=-1)
            else:
                elites_mat = FF.factor_coef_with_y(elites[self.insample], labels_res[self.insample].unsqueeze(-1), corr_dim=1, dim=-1)
            neutra = FF.top_svd_factors(elites_mat, elites, top_n = self.param.svd_top_n ,top_ratio=self.param.svd_top_ratio, dim=-1 , inplace = True) # use svd factors instead
            Logger.stdout(f'  -> Elites({elites.shape[-1]}) Shrink to SvdElites({neutra.shape[-1]})')

        if isinstance(neutra , torch.Tensor) and neutra.numel(): 
            self.neutra = neutra.cpu()
            Logger.stdout(f'  -> Neutra has {neutra.shape[-1]} Elements')

        assert isinstance(labels_res , torch.Tensor) , type(labels_res)
        labels_res = MF.neutralize_2d(labels_res, neutra , inplace = True) 
        self.logger.save_state(labels_res, 'res', self.i_iter) 

        if self.param.factor_neut_type > 0 and self.param.labels_neut_type == 'svd':
            lastneutra = None if self.i_iter == 0 else self.logger.load_state('neu' , self.i_iter - 1 , device = self.device)
            if isinstance(lastneutra , torch.Tensor): 
                lastneutra = lastneutra.cpu()
            if isinstance(neutra , torch.Tensor): 
                lastneutra = torch.cat([lastneutra , neutra.cpu()] , dim=-1) if isinstance(lastneutra , torch.Tensor) else neutra.cpu()
            self.logger.save_state(lastneutra , 'neu', self.i_iter) 
            del lastneutra
        
        self.tensors['labels_res'] = labels_res