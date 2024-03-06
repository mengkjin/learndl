import torch
import gp_math_func as MF

def process_factor(value , stream = 'inf_trim_norm' , dim = 1 , trim_ratio = 7. , norm_tol = 1e-4,
                   neutral_factors = None , neutral_groups = None , **kwargs):
    '''
    ------------------------ process factor value ------------------------
    处理因子值 , 'inf_trim_winsor_norm_neutral_nan'
    input:
        value:         factor value to be processed
        process_key:   can be any of 'inf_trim/winsor_norm_neutral_nan'
        dim:           default to 1
        trim_ratio:    what extend can be identified as outlier? range is determined as med ± trim_ratio * brandwidth
        norm_tol:      if norm required, the tolerance to eliminate factor if standard deviation is too trivial
        neutral_factors: market cap (neutralize param)
        neutral_groups:  industry indicator (neutralize param)
    output:
        value:         processed factor value
    '''
    if MF.is_invalid(value) or MF.allna(value , inf_as_na = True): return MF.invalid

    assert 'inf' in stream or 'trim' in stream or 'winsor' in stream , stream
    if 'trim' in stream or 'winsor' in stream:
        med       = value.nanmedian(dim , keepdim=True).values
        bandwidth = (value.nanquantile(0.75 , dim , keepdim=True) - value.nanquantile(0.25 , dim , keepdim=True)) / 2
        lbound , ubound = med - trim_ratio * bandwidth , med + trim_ratio * bandwidth

    if 'norm' in stream:
        m = torch.nanmean(value , dim, keepdim=True)
        s = (value - m).square().nansum(dim,True).sqrt()
        trivial = s < norm_tol + (m.abs() * norm_tol > s)
        # the other axis
        #m = torch.nanmean(value , dim-1, keepdim=True)
        #s  = (value - m).square().nansum(dim-,True).sqrt()
        #trivial = trivial + s < norm_tol + (m.abs() * norm_tol > s)

    for _str in stream.split('_'):
        if _str == 'inf':
            value.nan_to_num_(torch.nan,torch.nan,torch.nan)
        elif _str == 'trim':
            value[(value > ubound) + (value < lbound)] = torch.nan
        elif _str == 'winsor':
            value = torch.where(value > ubound , ubound , value)
            value = torch.where(value < lbound , lbound , value)
        elif _str == 'norm': 
            value = torch.where(trivial , value * 0 , MF.zscore(value , dim))
        # elif _str == 'neutral': 
            # '市值中性化（对x做，现已取消，修改为对Y做）'
            # value = MF.neutralize_2d(value , neutral_factors , neutral_groups , dim = dim) 
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
            w = torch.arange(len(max_len)).div(exp_halflife).pow(2)
    else:
        raise KeyError(method)
    return w