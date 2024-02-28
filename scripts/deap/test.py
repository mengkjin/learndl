import pandas as pd
import numpy as np
from numba import jit , cuda
import statsmodels.api as sm
import torch
from torch import nn

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import os
#if os.name == 'posix': assert(device == torch.device('cuda'))
#print(__name__)

#%% 中性化函数

def neutralize_numpy(alphan_value_gp, size_gp, cs_indus_code, other_factor_list=[], silent= True):  # [tensor (TS*C), tensor (TS*C)]
    assert (alphan_value_gp.shape == size_gp.shape)
    resids_all = torch.zeros_like(alphan_value_gp) * np.nan
    for dd in range(alphan_value_gp.shape[0]):
        if dd%500 == 0 and not silent: print('neutralize by tradedate',dd)
        y = alphan_value_gp[[dd], :].T  # [C, 1]
        concat_xy = y
        size_dd = size_gp[[dd], :].T  # [C, 1]
        concat_xy = torch.cat((concat_xy, size_dd), 1)  # [C, 2]
        cs = cs_indus_code[[dd], :].T  # [C, 1]
        for cs_index in range(1, 29 + 1 - 1):  # 29个一级行业代码，去掉最后一列避免线性相关
            cs_dummy = torch.zeros_like(cs)  # [C, 1]
            cs_dummy[cs == cs_index] = 1
            if cs_dummy.sum() == 0:
                continue
            concat_xy = torch.cat((concat_xy, cs_dummy), 1)  # [C, 2+28+n]

        if len(other_factor_list) > 0:
            for factor_num, other_factor in enumerate(other_factor_list):
                other_factor_dd = other_factor[[dd], :].T
                #对缺失值用每个时间截面的均值填充
                other_factor_dd = torch.where(torch.isnan(other_factor_dd), torch.nanmean(other_factor_dd), other_factor_dd)
                concat_xy = torch.cat((concat_xy, other_factor_dd), 1)
        
        nan_bool_index = torch.isnan(concat_xy).any(1)  # [C, 1]
        concat_xy = concat_xy[~nan_bool_index, :]  # [C, 2+28+n]
        concat_xy = torch.cat((concat_xy, torch.ones(concat_xy.shape[0], 1).to(alphan_value_gp.device)),
                              1)  # [C, 1+28+n+1]
        concat_xy = concat_xy.cpu().numpy()

        if concat_xy.shape[0] >= 10:
            x = concat_xy[:, 1:]  # [C, 1+29]
            y = concat_xy[:, 0]  # [C, 1]
            try:
                resids = (y - x @ np.linalg.lstsq(x, y, rcond=None)[0]).T  # [1, C]
            except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
                print('neutralization error!')
                try:    
                    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                    resids = (y - x @ beta).T
                except:
                    print('neutralization error!')
                    resids = alphan_value_gp.copy()
                    
            resids_all[[dd], ~nan_bool_index] = torch.Tensor(resids).to(alphan_value_gp.device)

    return resids_all

from sklearn.linear_model import LinearRegression
import torch
import numpy as np

def one_hot(x):
    if not isinstance(x , torch.Tensor): x = torch.Tensor(x)
    return torch.nn.functional.one_hot(x.to(torch.long)).to(torch.float)

def _neutralize_yx(y , x_list = [] , x_group = None , no_intercept = True , index = None):
    if len(x_list) == 0 and x_group is None: 
        return y
    elif x_group is None:
        x = torch.stack(x_list,dim = -1)
    elif len(x_list) == 0:
        x = one_hot(x_group)[...,:-1]
    else:
        x = torch.cat([torch.stack(x_list,dim = -1),one_hot(x_group)[...,:-1]],dim=-1)
    if no_intercept: x = torch.nn.functional.pad(x , (1,0) , value = 1.)
    y = y.unsqueeze(-1)
    if index: x , y = x[index] , y[index]
    return y , x

def neutralize(y , x , method = 'torch'):
    assert len(y.shape) == len(x.shape)
    if len(y.shape) == 1 and len(x.shape) == 2: y = y.unsqueeze(-1)
    y_dev = y.device
    if method == 'sk':
        model = LinearRegression(fit_intercept=False).fit(x, y)
        coef  = torch.Tensor(model.coef_).to(x).reshape(1,-1)
        resids = y.reshape(1,-1) - torch.matmul(coef , x.permute(1,0)) - model.intercept_
    else:
        pack = globals()[method]
        if method == 'np': 
            y , x  = y.cpu().numpy() , x.cpu().numpy()
        try:
            model = pack.linalg.lstsq(x , y , rcond=None)
            resids = (y - x @ model[0]).T  # [1, C]
        except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
            try:    
                beta = pack.matmul(pack.matmul(pack.linalg.inv(pack.matmul(x.T, x)) , x.T) , y)
                resids = (y - x @ beta).T
            except:
                print('neutralization error!')
                resids = y.copy()
    if not isinstance(resids , torch.Tensor): resids = torch.Tensor(resids)
    return resids.to(y_dev).flatten()

def neutralize_2d(y , x_list = [] , x_group = None, method = 'torch' , no_intercept = True , silent= True):  # [tensor (TS*C), tensor (TS*C)]
    assert method in ['sk' , 'np' , 'torch' , 'torch_3d']
    y , x = _neutralize_yx(y , x_list , x_group , no_intercept)
    if method == 'torch_3d':
        # fastest, but cannot deal nan's
        model = torch.linalg.lstsq(x , y , rcond=None)
        resids_all = (y - x @ model[0])
    else:
        resids_all = torch.zeros_like(y.squeeze(-1)).fill_(torch.nan)

        # if you can make sure there is no nan in the data, torch.linalg.lstsq(x, y) is much faster in 3d
        for dd in range(len(y)):
            if dd % 500 == 0 and not silent: print('neutralize by tradedate',dd)
            y_ , x_ = y[dd] , x[dd]
            nan_bool_index = y_.isnan().any(dim=1) + x_.isnan().any(dim=1)
            y_ , x_ = y_[~nan_bool_index] , x_[~nan_bool_index]
            if len(y_) < 10: continue
            resids = neutralize(y_ , x_ , method = method)
            resids_all[dd,~nan_bool_index] = resids
    return resids_all

# %timeit neutralize_2d(r , o , b)

r = torch.Tensor(np.random.rand(50,1000))

a = torch.Tensor(np.random.rand(50,1000))
b1 = np.random.choice(30,1000,True)
b  = torch.Tensor(np.tile(b1 , 50).reshape(50,1000))

o = [torch.Tensor(np.random.rand(50,1000)) , torch.Tensor(np.random.rand(50,1000))]

import cProfile
cProfile.run("neutralize_2d(r , o , b , 'np')")
cProfile.run("neutralize_2d(r , o , b , 'np')")
