from dataclasses import dataclass , field

import torch
import pandas as pd

from src.proj import Logger
from . import gp_math_func as MF

@dataclass
class FactorValue:
    name    : str
    process : str
    value   : torch.Tensor | pd.DataFrame | None
    infos   : dict = field(default_factory=dict)

    def __repr__(self):
        attr_repr = []
        for k , v in self.__dict__.items():
            attr_repr.append(f'{k}=torch.Tensor(shape{tuple(v.shape)},{v.device})' if isinstance(v , torch.Tensor) else f'{k}={v}')
        return f'{self.__class__.__name__}({", ".join(attr_repr)})'
    
    def isnull(self):
        return self.value is None
    
    def to_dataframe(self , index = None , columns = None):
        if self.value is None: 
            return None
        elif isinstance(self.value , pd.DataFrame):
            return self.value
        else:
            return pd.DataFrame(data = self.value.cpu().numpy() , index = index , columns = columns)

def process_factor(value , stream = 'inf_winsor_norm' , dim = 1 , trim_ratio = 7. , **kwargs):
    '''
    ------------------------ process factor value ------------------------
    处理因子值 , 'inf_trim_winsor_norm_neutral_nan'
    input:
        value:         factor value to be processed
        process_key:   can be any of 'inf_trim/winsor_norm_neutral_nan'
        dim:           default to 1
        trim_ratio:    what extend can be identified as outlier? range is determined as med ± trim_ratio * brandwidth
        norm_tol:      if norm required, the tolerance to eliminate factor if standard deviation is too trivial
    output:
        value:         processed factor value
    '''
    if value is None or MF.allna(value , inf_as_na = True): 
        return None

    # assert 'inf' in stream or 'trim' in stream or 'winsor' in stream , stream
    if 'trim' in stream or 'winsor' in stream:
        med       = value.nanmedian(dim , keepdim=True).values
        bandwidth = (value.nanquantile(0.75 , dim , keepdim=True) - value.nanquantile(0.25 , dim , keepdim=True)) / 2
        lbound , ubound = med - trim_ratio * bandwidth , med + trim_ratio * bandwidth
    for _str in stream.split('_'):
        if _str == 'mean':
            value -= torch.nanmean(value , dim, keepdim=True)
        elif _str == 'inf':
            value.nan_to_num_(torch.nan,torch.nan,torch.nan)
        elif _str == 'trim':
            value[(value > ubound) + (value < lbound)] = torch.nan
        elif _str == 'winsor':
            value = torch.where(value > ubound , ubound , value)
            value = torch.where(value < lbound , lbound , value)
        elif _str == 'norm': 
            value -= torch.nanmean(value , dim, keepdim=True)
            value /= value.square().nansum(dim , keepdim = True).sqrt() + 1e-6 
        elif _str == 'nan': 
            value = value.nan_to_num_()
    return value

def decay_weight(method , max_len , exp_halflife = -1):
    if method == 'constant':
        w = torch.ones(max_len)
    elif method == 'linear':
        w = torch.arange(max_len)
    elif method == 'exp':
        if exp_halflife <= 0:
            w = torch.ones(max_len)
        else:
            w = torch.arange(max_len).div(exp_halflife).pow(2)
    else:
        raise KeyError(method)
    return w

def factor_coef_mean(x , mean_dim = 0, dim = -1 , weight = None):
    if x is None:
        return x
    assert x.shape[dim] > 0 , (x.shape , dim)
    if x.shape[dim] == 1: 
        return torch.ones((1,1)).to(x)
    assert mean_dim >= 0 , f'mean_dim must be non-negative : {mean_dim}'
    assert dim >= -1 , f'dim must >= -1 : {dim}'
    if dim == -1: 
        dim = x.dim() - 1
    assert mean_dim != dim , f'mean_dim and dim should be different'
    if weight is not None:
        assert len(weight) == x.shape[mean_dim] , f'length of weight and shape[mean_dim] should be the same'
    else:
        weight = torch.ones(x.shape[mean_dim])
    # ij = torch.arange(x.shape[dim])
    new_dim = dim if dim < mean_dim else dim - 1
    coefs = []
    for i in range(x.shape[mean_dim]):
        x_ = x.select(mean_dim , i).transpose(-1,new_dim)
        x_ = torch.corrcoef(x_[~x_.isnan().any(-1)].T)
        coefs.append(x_ * weight[i])
    x = torch.stack(coefs).sum(0) / sum(weight)
    # x[ij,ij] = 1
    return x

def factor_coef_total(x , dim = -1):
    if x is None: 
        return x
    assert x.shape[dim] > 0 , (x.shape , dim)
    if x.shape[dim] == 1: 
        return torch.ones((1,1)).to(x)
    ij = torch.arange(x.shape[dim])
    x = x.transpose(-1 , dim).reshape(-1 , len(ij))
    x = torch.corrcoef(x[~x.isnan().any(-1)].T)
    x[ij,ij] = 1
    return x

def factor_coef_with_y(x , y , corr_dim = 1, dim = -1):
    if x is None: 
        return x
    assert corr_dim >= 0 , f'corr_dim must be non-negative : {corr_dim}'
    assert dim >= -1 , f'dim must >= -1 : {dim}'
    if dim == -1:
        dim = x.dim() - 1
    assert corr_dim != dim , f'corr_dim and dim should be different'
    assert x.shape[dim] > 0 , (x.shape , dim)
    if x.shape[dim] == 1: 
        return torch.ones((1,1)).to(x)
    ij = torch.arange(x.shape[dim])
    new_dim = dim if dim < corr_dim else dim - 1
    try:
        # raise torch.cuda.OutOfMemoryError
        tscorr = MF.corrwith(x , y , dim = corr_dim)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        tscorr = []
        for i in range(x.shape[dim]):
            # Logger.stdout(x.select(dim , i).shape , y.select(dim , 0).shape)
            tscorr.append(MF.corrwith(x.select(dim , i) , y.select(dim , 0) , dim = corr_dim))
        tscorr = torch.stack(tscorr , dim = new_dim)
    x = tscorr.transpose(-1 , new_dim)
    x = torch.corrcoef(x[~x.isnan().any(-1)].T)
    assert len(x) == len(ij) , (x , ij)
    x[ij,ij] = 1
    return x

def svd_factors(mat , raw_factor , top_n = -1 , top_ratio = 0. , dim = -1 , inplace = True):
    if mat is None or raw_factor is None: 
        return raw_factor
    assert mat.dim() == 2 , mat.shape
    assert mat.shape[0] == mat.shape[1] , mat.shape
    assert mat.shape[0] == raw_factor.shape[dim] , (mat.shape , raw_factor.shape , dim)
    svd = torch.linalg.svd(mat)
    right  = 'jkl'
    left   = 'i' + right[dim] 
    target = right.replace(right[dim] , '') + 'i' 
    einsum_formula = f'{left},{right}->{target}'

    finite_ij  = raw_factor.select(dim,0).isfinite()
    if inplace:
        raw_factor.nan_to_num_(0,0,0)
    else:
        raw_factor = raw_factor.nan_to_num(0,0,0)
    for k in range(1, raw_factor.shape[dim]): 
        finite_ij += raw_factor.select(dim,k).isfinite()
    vector = torch.einsum(einsum_formula , svd.U, raw_factor)
    vector[~finite_ij] = torch.nan
    return vector , svd

def top_svd_factors(mat , raw_factor , top_n = 1 , top_ratio = 0. , dim = -1 , inplace = True):
    '''
    mat1 = factor_coef_mean(raw_factor , dim = -1 , weight = None)
    mat2 = factor_coef_total(raw_factor, dim = -1)
    mat3 = factor_coef_with_y(raw_factor , y , corr_dim=1 , dim = -1)
    '''
    if mat is None or raw_factor is None: 
        return raw_factor
    vector , svd = svd_factors(mat , raw_factor , dim = dim , inplace = inplace)
    where  = svd.S.cumsum(0) / svd.S.sum() <= top_ratio
    where += torch.arange(vector.shape[-1] , device=where.device) < max(top_n , where.sum() + 1)
    Logger.stdout(svd.S.cumsum(0) / svd.S.sum())
    return vector[...,where]

@dataclass
class MultiFactorValue:
    value  : torch.Tensor
    weight : torch.Tensor
    inputs : torch.Tensor

    def __repr__(self) -> str:
        attr_repr = []
        for k , v in self.__dict__.items():
            attr_repr.append(f'{k}=torch.Tensor(shape{tuple(v.shape)},{v.device})' if isinstance(v , torch.Tensor) else f'{k}={v}')
        return f'{self.__class__.__name__}({", ".join(attr_repr)})'
    
class MultiFactor:
    def __init__(self , weight_scheme = 'ic', window_type = 'rolling', weight_decay= 'exp' , 
                 ir_window = 40 , roll_window = 40 , halflife  = 20 , 
                 insample = None , universe = None , min_coverage = 0.1 , **kwargs) -> None:
        assert weight_scheme in ['ew' , 'ic' , 'ir'] , weight_scheme
        assert window_type   in ['rolling' , 'insample'] , window_type
        assert weight_decay  in ['constant' , 'linear' , 'exp'] , weight_decay
        self.weight_scheme = weight_scheme
        self.window_type   = window_type
        self.weight_decay  = weight_decay
        self.ir_window     = ir_window
        self.roll_window   = roll_window
        self.halflife      = halflife
        self.insample      = insample
        self.universe      = universe
        self.min_coverage  = min_coverage

    def ts_decay(self , max_len , weight_decay = None , halflife = None):
        weight_decay = weight_decay if weight_decay else self.weight_decay
        halflife     = halflife     if halflife     else self.halflife
        return decay_weight(weight_decay , max_len , exp_halflife=halflife)

    @staticmethod
    def static_decorator(func , relative_weight_cap = 5.):
        def wrapper(data , time_slice = None):
            if time_slice is not None and data is not None: 
                data = data[time_slice]
            w = func(data).nan_to_num(torch.nan,torch.nan,torch.nan).reshape(1,1,-1)
            w /= w.abs().sum(-1,keepdim=True)
            w[w > (relative_weight_cap / w.shape[-1])] = relative_weight_cap / w.shape[-1]
            w /= w.abs().sum(-1,keepdim=True)
            return w
        return wrapper
    
    @staticmethod
    def dynamic_decorator(func , relative_weight_cap = 5. , method = 0):
        def wrapper(data , roll_window = 10):
            if method == 1:
                data = torch.nn.functional.pad(data,[0,0,roll_window-1,0],value=torch.nan).unfold(0,roll_window,1).permute(2,0,1)
                w = func(data).nan_to_num(torch.nan,torch.nan,torch.nan).permute(1,0,2)
            else:
                w = data * 0.
                for i in range(len(w)):
                    w[i] = func(data[i-roll_window:i]).nan_to_num(torch.nan,torch.nan,torch.nan)
                w = w.unsqueeze(1)
            w /= w.abs().sum(-1,keepdim=True)
            w[w > (relative_weight_cap / w.shape[-1])] = relative_weight_cap / w.shape[-1]
            w /= w.abs().sum(-1,keepdim=True)
            return w
        return wrapper
    
    def multi_factor(self , factor , window_type = None , **kwargs):
        weight = self.factor_weight(window_type , **kwargs)
        try:
            multi = (factor * weight).nanmean(-1)
        except torch.cuda.OutOfMemoryError:
            Logger.warning(f'OutOfMemoryError on multi factor calculation')
            multi = (factor.cpu() * weight.cpu()).nanmean(-1).to(factor)
        multi = MF.zscore(multi , -1)
        return MultiFactorValue(value = multi , weight = weight , inputs = factor)
    
    def factor_weight(self , window_type = None , **kwargs):
        window_type = window_type if window_type is not None else self.window_type
        if window_type == 'insample':
            weight_tensor = self.static_factor_weight(**kwargs)
        else:
            weight_tensor = self.dynamic_factor_weight(**kwargs)
        return weight_tensor

    def static_factor_weight(self , weight_scheme = None , weight_decay = None , insample = None , **kwargs):
        weight_scheme = weight_scheme  if weight_scheme is not None else self.weight_scheme
        weight_decay  = weight_decay   if weight_decay  is not None else self.weight_decay
        insample      = insample       if insample      is not None else self.insample
        assert weight_scheme in ['ew' , 'ic' , 'ir'] , weight_scheme
        assert weight_decay  in ['constant' , 'linear' , 'exp'] , weight_decay

        if weight_scheme == 'ew': 
            func = self.weight_ew
            data = kwargs['ic'] if 'ic' in kwargs.keys() else kwargs['ir']
        else:
            func = self.weight_icir
            data = kwargs[weight_scheme]
        func = self.static_decorator(func)
        return func(data , time_slice = insample)
    
    def dynamic_factor_weight(self , weight_scheme = None , weight_decay = None , roll_window = None , **kwargs):
        weight_scheme = weight_scheme  if weight_scheme is not None else self.weight_scheme
        weight_decay  = weight_decay   if weight_decay  is not None else self.weight_decay
        roll_window   = roll_window    if roll_window   is not None else self.roll_window
        assert weight_scheme in ['ew' , 'ic' , 'ir'] , weight_scheme
        assert weight_decay  in ['constant' , 'linear' , 'exp'] , weight_decay

        if weight_scheme == 'ew': 
            return self.static_factor_weight(weight_scheme , time_slice = self.insample, **kwargs)
        else:
            func = self.dynamic_decorator(self.weight_icir)
            return func(kwargs[weight_scheme] , roll_window = roll_window)

    def weight_ew(self , data):
        return data.nanmean(0).sign()

    def weight_icir(self, data):
        ts_w = self.ts_decay(len(data)).to(data).reshape(1,-1)
        fini = data.isfinite() * 1.
        data = data.nan_to_num(0,0,0)
        if data.dim() == 2:
            return (ts_w @ data) / (ts_w @ fini)
        elif data.dim() == 3:
            return torch.einsum('ij,jkl->ikl' , ts_w , data) / torch.einsum('ij,jkl->ikl' , ts_w , fini)

    def weighted_multi(self , singles , weight):
        assert singles.shape == weight.shapes , (singles.shape , weight.shape)
        weight = singles.isfinite() * weight
        wsum = torch.nansum(singles * weight , dim = -1) 
        return MF.zscore_inplace(wsum,-1)
    
    def calculate_icir(self , factors , labels , ir_window = None , universe = None , min_coverage = None , **kwargs):
        ir_window    = ir_window    if ir_window    is not None else self.ir_window
        universe     = universe     if universe     is not None else self.universe
        min_coverage = min_coverage if min_coverage is not None else self.min_coverage
        if labels.dim() == factors.dim():
            labels = labels.squeeze(-1)
        # rankic = torch.full((len(factors) , factors.shape[-1]) , fill_value=torch.nan).to(labels)
        rankic = []
        for i_factor in range(factors.shape[-1]):
            rankic.append(MF.rankic_2d(factors[...,i_factor] , labels , dim = 1 , universe = universe , min_coverage = min_coverage))
        rankic = torch.stack(rankic , dim = -1)
        rankir = MF.ma(rankic , ir_window) / MF.ts_stddev(rankic , ir_window)
        return {'ic' : rankic , 'ir' : rankir}