import copy
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Any

from src.proj import Logger , Proj , CALENDAR
from src.proj.util import torch_load
from src.func import tensor as T
from src.data import DATAVENDOR
from src.data.util import DataBlock
from src.data.loader import BlockLoader
from src.res.gp.param import gpDefaults , gpParameters
from src.res.gp.func import factor_func as FF
from .logger import gpLogger
from .status import gpStatus

# db_src , db_key , feature , scale

@dataclass
class InputElement:
    name: str
    db_src: str
    db_key: str
    feature: str
    scale: float = 1.
    inverse: bool = False
    fill_nan: Any = None

    def load_tuple(self) -> tuple[str, str]:
        return (self.db_src, self.db_key)

    @property
    def is_raw(self) -> bool:
        return self.name.islower()

    @property
    def is_fac(self) -> bool:
        return self.name.isupper()

    def fac_input(self):
        return InputElement(self.name.upper() , self.db_src , self.db_key , self.feature , self.scale , self.inverse)

    def raw_input(self):
        return InputElement(self.name.lower() , self.db_src , self.db_key , self.feature , self.scale , self.inverse)

    def get_datablock(self , start : int = 20100101 , end : int = 20241231) -> DataBlock:
        block = BlockLoader(self.db_src , self.db_key , feature = [self.feature]).load(start , end , vb_level = 'never')
        return self.adjust_datablock(block)

    def adjust_datablock(self , block : DataBlock , cp_block : DataBlock | None = None) -> DataBlock:
        feature_list = [f for f in block.feature]
        if self.feature not in feature_list:
            return block
        i = feature_list.index(self.feature)
        block.feature[i] = self.name
        values : torch.Tensor = block.values[...,i]
        if self.inverse:
            values = 1. / values
        if self.is_fac:
            values = T.zscore(values , dim = 0)
        elif self.scale != 1.:
            values = values * self.scale
        if self.fill_nan is not None and cp_block is not None:
            cp_notnan = ~cp_block.values[...,0].isnan()
            values = values.nan_to_num(self.fill_nan).where(cp_notnan , torch.nan)
        block.values[...,i] = values
        return block

INPUT_MAPPING = {
    'cp': InputElement('cp' , 'trade_ts' , 'day' , 'close'), 
    'turn': InputElement('turn' , 'trade_ts' , 'day' , 'turn_fl'), 
    'vol': InputElement('vol' , 'trade_ts' , 'day' , 'volume' , 1e-6), 
    'amt': InputElement('amt' , 'trade_ts' , 'day' , 'amount' , 1e-6), 
    'op': InputElement('op' , 'trade_ts' , 'day' , 'open'), 
    'hp': InputElement('hp' , 'trade_ts' , 'day' , 'high'), 
    'lp': InputElement('lp' , 'trade_ts' , 'day' , 'low'), 
    'vp': InputElement('vp' , 'trade_ts' , 'day' , 'vwap'), 
    'bp': InputElement('bp' , 'trade_ts' , 'day_val' , 'pb' , inverse = True), 
    'ep': InputElement('ep' , 'trade_ts' , 'day_val' , 'pe' , inverse = True), 
    'dp': InputElement('dp' , 'trade_ts' , 'day_val' , 'dv_ratio'), 
    'rtn': InputElement('rtn' , 'trade_ts' , 'day' , 'pctchange' , 0.01),
}
INPUT_MAPPING.update({k.upper():v.fac_input() for k,v in INPUT_MAPPING.items()})

def get_features_block(src : str , key : str , features : list[str] | None , start : int = 20100101 , end : int = 20241231) -> DataBlock:
    '''
    获取特征
    input:
        features:    features
        start:       start date
        end:      end date
    output:
        block:       DataBlock        
    '''
    secid = DATAVENDOR.secid(end)
    dates = CALENDAR.range(start , end , 'td')
    block = BlockLoader(src , key , feature = features).load(start , end , vb_level = 'never').\
        align_secid_date(secid , dates , inplace = True)
    return block

def load_inputs(inputs : list[str] , start : int = 20100101 , end : int = 20241231 , cp_block : DataBlock | None = None) -> DataBlock:
    '''
    读取输入因子并转化成DataBlock,额外返回secid和date的ndarray
    input:
        filename:    filename gp data
        start:       start date
        end:      end date
        input_freq:  1 , 5 , ...
    output:
        block:       DataBlock        
    '''

    input_elements = [INPUT_MAPPING[input] for input in inputs]
    load_tuples = list(set([element.load_tuple() for element in input_elements]))
    assert load_tuples , f'load_tuples is empty'
    
    if cp_block is None:
        cp_block = get_cp_block(start , end)
    blocks : list[DataBlock] = []
    
    for tup in load_tuples:
        sub_elements = [element for element in input_elements if element.load_tuple() == tup]
        load_features = list(set([element.feature for element in sub_elements]))
        block = get_features_block(tup[0] , tup[1] , load_features , start , end)
        raw_features = {element.feature : element for element in sub_elements if element.is_raw}
        if raw_features:
            raw_block = block.align_feature(list(raw_features.keys()) , inplace = False)
            [element.adjust_datablock(raw_block , cp_block) for element in raw_features.values()]
            blocks.append(raw_block)
        
        fac_features = {element.feature : element for element in sub_elements if element.is_fac}
        if fac_features:
            fac_block = block.align_feature(list(fac_features.keys()) , inplace = False)
            [element.adjust_datablock(fac_block , cp_block) for element in fac_features.values()]
            blocks.append(fac_block)
        assert blocks , f'blocks is empty'
        block = DataBlock.merge(blocks , inplace = True).align_feature(inputs , inplace = True)
    return block

def get_cp_block(start : int = 20100101 , end : int = 20241231) -> DataBlock:
    '''
    获取收盘价
    input:
        start:       start date
        end:      end date
    output:
        cp_block:          close price tensor
    '''
    return get_features_block('trade_ts' , 'day' , ['close'] , start , end)

def get_return_block(start : int = 20100101 , end : int = 20241231 , nday : int = 10 , delay : int = 1) -> DataBlock:
    '''
    获取收益率
    input:
        start:    start date
        end:      end date
        extend:      extend
    output:
        return_block:      return tensor
    '''
    element = InputElement('rtn' , 'trade_ts' , 'day' , 'pctchange' , 0.01)
    secid = DATAVENDOR.secid(end)
    dates = CALENDAR.range(start , end , 'td')
    new_end_dt = CALENDAR.td(end , 20).td
    block = element.get_datablock(start , new_end_dt)
    block.values = T.ts_delay(T.ts_product(block.values + 1 , nday) - 1 , -nday-delay , no_alert = True)
    block = block.align_secid_date(secid , dates , inplace = True)
    return block

def init_neutral_exp(start : int = 20100101 , end : int = 20241231 , * , device : torch.device | str | None = None) -> torch.Tensor:
    '''
    获取中性化使用的行业因子和市值因子
    input:
        start:        start date
        end:       end date
    output:
        neutral_exp:  neutral_exp tensor
    '''
    secid = DATAVENDOR.secid(end)
    dates = CALENDAR.range(start , end , 'td')
    block = BlockLoader('models' , 'tushare_cne5_exp').load(start , end , vb_level = 'never').\
        align_secid_date(secid , dates , inplace = True)
    values = block.loc(feature = Proj.Conf.Factor.RISK.indus + ['size']).squeeze().to(device)
    return values

def init_labels_raw(start : int = 20100101 , end : int = 20241231 , * , neutral_exp : torch.Tensor | None = None , nday = 10 , delay = 1 , 
                    device : torch.device | str | None = None) -> torch.Tensor:
    '''
    生成原始预测标签,中性化后的10日收益
    input:
        neutral_exp:    neutral_exp , industry and size factors
        nday:           nday
        delay:          delay
        start:          start date
        end:         end date
    output:
        labels_raw:     
    '''
    
    rtn = get_return_block(start , end , nday , delay).values.squeeze().to(device)
    if neutral_exp is not None:
        neutral_exp = neutral_exp.to(rtn)
    labels_raw = T.neutralize_2d(rtn , neutral_exp , method = 'torch' , inplace = True)  # 市值行业中性化
    
    assert labels_raw is not None , 'labels_raw is None'
    return labels_raw

class gpInput:
    """遗传规划输入,包括参数、输入、输出、文件管理、内存管理、计时器、评价器、数据列"""
    def __init__(self , param : gpParameters , status : gpStatus , logger : gpLogger , 
                 vb_level : Any = 2):
        self.param     = param
        self.status    = status
        self.logger    = logger
        self.inputs :    list[torch.Tensor] = []
        self.tensors :   dict[str , torch.Tensor] = {}
        self.records :   dict[str , Any] = {}
        self.vb_level = Proj.vb(vb_level)

    def __repr__(self):
        return f'{self.__class__.__name__}(inputs={len(self.inputs)}, tensors={len(self.tensors)}, records={len(self.records)})'

    @property
    def start(self) -> int:
        return self.param.insample_dates[0]
    @property
    def end(self) -> int:
        return self.param.outsample_dates[1]

    @property
    def i_iter(self) -> int:
        return self.status.i_iter
    @property
    def device(self):
        return self.param.device
    @property
    def share_memory(self):
        return self.param.worker_num > 1 and (self.device is None or str(self.device).startswith('cpu'))
    @property
    def argnames(self):
        return self.param.argnames
    @property
    def n_args(self):
        return self.param.n_args
    @property
    def secid(self) -> np.ndarray:
        return self.records.get('secid' , np.array([]))
    @property
    def date(self) -> np.ndarray:
        return self.records.get('date' , np.array([]))
    @property
    def neutral_exp(self) -> torch.Tensor:
        return self.tensors['neutral_exp']
    @property
    def labels_raw(self) -> torch.Tensor:
        return self.tensors['labels_raw']
    @property
    def labels_res(self) -> torch.Tensor:
        return self.tensors['labels_res']
    @property
    def universe(self) -> torch.Tensor:
        return self.tensors['universe']
    @property
    def insample(self) -> torch.Tensor:
        return self.tensors['insample']
    @property
    def outsample(self) -> torch.Tensor:
        return self.tensors['outsample']

    @property
    def package_path(self):
        return gpDefaults.dir_data_package.joinpath('test_package.pt' if self.param.test_code else 'fit_package.pt')

    def load_data(self , force_reload : bool = False):
        loaded = False
        if not force_reload:
            loaded = self.load_package()
        if not loaded:
            self.load_source()
              
        self.tensors['insample']  = torch.Tensor((self.records['date'] >= self.param.insample_dates[0]) * 
                                      (self.records['date'] <= self.param.insample_dates[1])).bool().to(self.device)
        self.tensors['outsample'] = torch.Tensor((self.records['date'] >= self.param.outsample_dates[0]) * 
                                      (self.records['date'] <= self.param.outsample_dates[1])).bool().to(self.device)

        if self.share_memory:
            for input in self.inputs:
                input.share_memory_()
            for tensor in self.tensors.values():
                tensor.share_memory_()
            self.insample.share_memory_()
            self.outsample.share_memory_()

        self.logger.save_states({'params' : self.param.params, 'secid' : self.records['secid'] , 'date' : self.records['date']}) # useful to assert same index as package data
        return self

    def load_package(self):
        if not self.package_path.exists():
            return False

        try:
            package_data = torch_load(self.package_path , map_location = self.device)
        except Exception as e:
            Logger.error(f'Error loading package: {e}' , indent = 1 , vb_level = self.vb_level)
            self.package_path.unlink()
            return False

        package_require = ['argnames' , 'inputs' , 'neutral_exp' , 'labels_raw' , 'secid' , 'date' , 'universe']
        
        if not np.isin(package_require , list(package_data.keys())).all():
            Logger.alert1(f'Exists "{self.package_path}" but Lack Required Data!' , indent = 1 , vb_level = self.vb_level)
            Logger.alert1(f'Required: {package_require}' , indent = 1 , vb_level = self.vb_level)
            Logger.alert1(f'Available: {list(package_data.keys())}' , indent = 1 , vb_level = self.vb_level)
            return False

        if not np.isin(self.argnames , package_data['argnames']).all():
            Logger.alert1(f'Exists package_data["argnames"] but Lack Required Data!' , indent = 1 , vb_level = self.vb_level)
            Logger.alert1(f'Required: {self.argnames}' , indent = 1 , vb_level = self.vb_level)
            Logger.alert1(f'Available: {package_data['argnames']}' , indent = 1 , vb_level = self.vb_level)
            return False

        Logger.stdout(f'Directly load "{self.package_path}"' , indent = 1 , vb_level = self.vb_level)

        for gp_key in self.argnames:
            gp_val = package_data['inputs'][package_data['argnames'].index(gp_key)]
            gp_val = gp_val.to(self.device)
            self.inputs.append(gp_val)

        for gp_key in ['neutral_exp' , 'labels_raw' , 'universe']: 
            gp_val = package_data[gp_key].to(self.device)
            self.tensors[gp_key] = gp_val

        for gp_key in ['secid' , 'date']: 
            gp_val = package_data[gp_key]
            self.records[gp_key] = gp_val

        return True

    
    def load_source(self):
        Logger.stdout(f'Load from DB' , indent = 1 , vb_level = self.vb_level)
        nrowchar = 0

        cp_block = get_cp_block(self.start , self.end)

        self.tensors['neutral_exp'] = init_neutral_exp(self.start , self.end).to(self.device)
        self.tensors['universe']   = ~cp_block.values.squeeze().isnan().to(self.device)
        self.tensors['labels_raw'] = init_labels_raw(self.start , self.end  , neutral_exp = self.neutral_exp).to(self.device)
        
        input_block = load_inputs(self.argnames , self.start , self.end , cp_block)
        self.records['date'] = input_block.date
        self.records['secid'] = input_block.secid

        for i , gp_key in enumerate(self.argnames):
            if nrowchar == 0: 
                Logger.stdout('' , end='', indent = 1 , vb_level = self.vb_level)
            gp_val = input_block.values[...,i:i+1].squeeze().to(self.device)
            self.inputs.append(gp_val)
            
            Logger.stdout(gp_key , end=',', vb_level = self.vb_level)
            nrowchar += len(gp_key) + 1
            if nrowchar >= 100 or i == len(self.argnames):
                Logger.stdout(vb_level = self.vb_level)
                nrowchar = 0

        gpDefaults.dir_data_package.mkdir(parents=True, exist_ok=True)
        saved_data = {
            'argnames' : self.argnames ,
            'inputs' : self.inputs ,
            'neutral_exp' : self.tensors['neutral_exp'] ,
            'labels_raw' : self.tensors['labels_raw'] ,
            'universe' : self.tensors['universe'] ,
            'secid' : self.records['secid'] ,
            'date' : self.records['date'] ,
        }
        torch.save(saved_data , self.package_path)
        Logger.success(f'Package data saved to "{self.package_path}"' , indent = 1 , vb_level = self.vb_level)

    def update_residual(self , **kwargs):
        """计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化"""
        assert self.param.labels_neut_type in ['svd' , 'all'] , self.param.labels_neut_type #  'all'
        assert self.param.svd_mat_method in ['coef_ts' , 'total'] , self.param.svd_mat_method

        self.neutra = None
        if self.i_iter == 0:
            labels_res = copy.deepcopy(self.labels_raw)
            elites     = None
        else:
            labels_res = self.logger.load_state('res' , self.i_iter - 1 , device = self.device)
            elites     = self.logger.load_state('elt' , self.i_iter - 1 , device = self.device)
        neutra = elites

        if isinstance(elites , torch.Tensor) and self.param.labels_neut_type == 'svd': 
            assert isinstance(labels_res , torch.Tensor) , type(labels_res)
            if self.param.svd_mat_method == 'total':
                elites_mat = FF.factor_corr_total(elites[self.insample],dim=-1)
            else:
                elites_mat = FF.factor_corr_with_y(elites[self.insample], labels_res[self.insample].unsqueeze(-1), corr_dim=1, dim=-1)
            neutra = FF.top_svd_factors(elites_mat, elites, top_n = self.param.svd_top_n ,top_ratio=self.param.svd_top_ratio, dim=-1 , inplace = True) # use svd factors instead
            assert neutra is not None , 'neutra is None'
            Logger.stdout(f'Elites({elites.shape[-1]}) Shrink to SvdElites({neutra.shape[-1]})' , indent = 1 , vb_level = self.vb_level)

        if isinstance(neutra , torch.Tensor) and neutra.numel(): 
            self.neutra = neutra.cpu()
            Logger.stdout(f'Neutra has {neutra.shape} Elements' , indent = 1 , vb_level = 'max')

        assert isinstance(labels_res , torch.Tensor) , type(labels_res)
        if isinstance(neutra , pd.DataFrame):
            neutra = torch.Tensor(neutra.values)
        labels_res = T.neutralize_2d(labels_res, neutra , inplace = True) 
        assert labels_res is not None , 'labels_res is None'
        self.logger.save_state('res', labels_res, self.i_iter) 

        if self.param.factor_neut_type > 0 and self.param.labels_neut_type == 'svd':
            lastneutra = None if self.i_iter == 0 else self.logger.load_state('neu' , self.i_iter - 1 , device = self.device)
            if isinstance(lastneutra , torch.Tensor): 
                lastneutra = lastneutra.cpu()
            if isinstance(neutra , torch.Tensor): 
                lastneutra = torch.cat([lastneutra , neutra.cpu()] , dim=-1) if isinstance(lastneutra , torch.Tensor) else neutra.cpu()
            self.logger.save_state('neu', lastneutra, self.i_iter) 
            del lastneutra
        
        self.tensors['labels_res'] = labels_res.to(self.device)
        if self.share_memory:
            self.labels_res.share_memory_()
        return self