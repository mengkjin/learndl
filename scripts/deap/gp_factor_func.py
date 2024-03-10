import torch
import gp_math_func as MF

def process_factor(value , stream = 'inf_trim_norm' , dim = 1 , trim_ratio = 7. , 
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
            w = torch.arange(max_len).div(exp_halflife).pow(2)
    else:
        raise KeyError(method)
    return w

def factor_coef_mean(x , mean_dim = 0, dim = -1 , weight = None):
    if MF.is_invalid(x): return x
    assert x.shape[dim] > 0
    if x.shape[dim] == 1: return torch.ones((1,1)).to(x)
    assert mean_dim >= 0 , f'mean_dim must be non-negative : {mean_dim}'
    assert dim >= -1 , f'dim must >= -1 : {dim}'
    if dim == -1: dim = x.dim() - 1
    assert mean_dim != dim , f'mean_dim and dim should be different'
    if weight is not None:
        assert len(weight) == x.shape[mean_dim] , f'length of weight and shape[mean_dim] should be the same'
    else:
        weight = torch.ones(x.shape[mean_dim])
    ij = torch.arange(x.shape[dim])
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
    if MF.is_invalid(x): return x
    assert x.shape[dim] > 0
    if x.shape[dim] == 1: return torch.ones((1,1)).to(x)
    ij = torch.arange(x.shape[dim])
    x = x.transpose(-1 , dim).reshape(-1 , len(ij))
    x = torch.corrcoef(x[~x.isnan().any(-1)].T)
    x[ij,ij] = 1
    return x

def factor_coef_with_y(x , y , corr_dim = 1, dim = -1):
    if MF.is_invalid(x): return x
    assert corr_dim >= 0 , f'corr_dim must be non-negative : {corr_dim}'
    assert dim >= -1 , f'dim must >= -1 : {dim}'
    if dim == -1: dim = x.dim() - 1
    assert corr_dim != dim , f'corr_dim and dim should be different'
    assert x.shape[dim] > 0
    if x.shape[dim] == 1: return torch.ones((1,1)).to(x)
    ij = torch.arange(x.shape[dim])
    new_dim = dim if dim < corr_dim else dim - 1
    try:
        # raise torch.cuda.OutOfMemoryError
        tscorr = MF.corrwith(x , y , dim = corr_dim)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        tscorr = []
        for i in range(x.shape[dim]):
            # print(x.select(dim , i).shape , y.select(dim , 0).shape)
            tscorr.append(MF.corrwith(x.select(dim , i) , y.select(dim , 0) , dim = corr_dim))
        tscorr = torch.stack(tscorr , dim = new_dim)
    except Exception as e:
        raise Exception(e)
    x = tscorr.transpose(-1 , new_dim)
    x = torch.corrcoef(x[~x.isnan().any(-1)].T)
    assert len(x) == len(ij) , print(x)
    x[ij,ij] = 1
    return x

def svd_factors(mat , raw_factor , top_n = -1 , top_ratio = 0. , dim = -1 , inplace = True):
    if MF.is_invalid(mat) or MF.is_invalid(raw_factor): return raw_factor
    assert mat.dim() == 2 
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] == raw_factor.shape[dim]
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
    for k in range(1, raw_factor.shape[dim]): finite_ij += raw_factor.select(dim,k).isfinite()
    vector = torch.einsum(einsum_formula , svd.U, raw_factor)
    vector[~finite_ij] = torch.nan
    return vector , svd

def top_svd_factors(mat , raw_factor , top_n = 1 , top_ratio = 0. , dim = -1 , inplace = True):
    '''
    mat1 = factor_coef_mean(raw_factor , dim = -1 , weight = None)
    mat2 = factor_coef_total(raw_factor, dim = -1)
    mat3 = factor_coef_with_y(raw_factor , y , corr_dim=1 , dim = -1)
    '''
    if MF.is_invalid(mat) or MF.is_invalid(raw_factor): return raw_factor
    vector , svd = svd_factors(mat , raw_factor , dim = dim , inplace = inplace)
    where  = svd.S.cumsum(0) / svd.S.sum() <= top_ratio
    where += torch.arange(vector.shape[-1] , device=where.device) < max(top_n , where.sum() + 1)
    print(svd.S.cumsum(0) / svd.S.sum())
    return vector[...,where]