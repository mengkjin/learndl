import os
import pandas as pd
import numpy as np
import numba
from numba import cuda
from numba import jit
import statsmodels.api as sm
import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#if os.name == 'posix': assert(device == torch.device('cuda'))

print(__name__)

#%% 中性化函数
def neutralize(alphan_value_gp,size_gp):
    name2num = {}
    for i, col in enumerate(alphan_value_gp.columns):
        name2num[col] = i + 1  # 将索引从 1 开始，方便后续的处理
    num2name = {value: key for key, value in name2num.items()}
    alphan_value_gp_1col = pd.melt(alphan_value_gp.rename(columns=name2num).reset_index(), id_vars=['index'],
                                   var_name='stock', value_name='alphan')
    alphan_value_gp_1col = alphan_value_gp_1col.set_index(['index', 'stock'])
    size_gp_1col = pd.melt(size_gp.rename(columns=name2num).reset_index(), id_vars=['index'], var_name='stock',
                             value_name='size')
    size_gp_1col = size_gp_1col.set_index(['index', 'stock'])
    resids_all = None
    for dd in alphan_value_gp.index.unique():
        y = alphan_value_gp_1col.loc[(dd, slice(None)), :]
        x = size_gp_1col.loc[(dd, slice(None)), :]

        concat_xy = pd.concat([y, x], axis=1).dropna(how='any')
        if concat_xy.shape[0] < 10:
            resids = pd.DataFrame(np.nan, index=[dd], columns=alphan_value_gp.columns)
        else:
            x = concat_xy.iloc[:, 1]
            y = concat_xy.iloc[:, 0]
            x = sm.add_constant(x)
            resids = pd.DataFrame(y - x @ np.linalg.lstsq(x, y, rcond=None)[0]).unstack()

            # resids.index = [i for i in resids.index] #type: ignore
            resids.columns = [num2name[i[1]] for i in resids.columns]
            resids = resids.reindex(columns=alphan_value_gp.columns) #type: ignore

        if resids_all is None:
            resids_all = resids.to_numpy()
        else:
            resids_all = np.append(resids_all, resids.to_numpy(), axis=0)

    resids_all_df = pd.DataFrame(resids_all, index=alphan_value_gp.index.unique(), columns=alphan_value_gp.columns)
    return resids_all_df

def neutralize_torch(alphan_value_gp,size_gp,cs_indus_code,other_factor_list=[]):  #[tensor (TS*C), tensor (TS*C)]
    assert (alphan_value_gp.shape == size_gp.shape)
    resids_all = torch.zeros_like(alphan_value_gp) * np.nan
    for dd in range(alphan_value_gp.shape[0]):
        y = alphan_value_gp[[dd], :].T  # [C, 1]
        x = size_gp[[dd], :].T  # [C, 1]
        concat_xy = torch.cat((y, x), 1)  # [C, 2]
        cs = cs_indus_code[[dd], :].T  # [C, 1]
        for cs_index in range(1,29+1-1):  #29个一级行业代码，去掉最后一列避免线性相关
            cs_dummy = torch.zeros_like(cs)  # [C, 1]
            cs_dummy[cs == cs_index] = 1
            if cs_dummy.sum() == 0:
                continue
            concat_xy = torch.cat((concat_xy, cs_dummy), 1)  # [C, 2+28+n]
        for factor_num,other_factor in enumerate(other_factor_list):
            other_factor_dd = other_factor[[dd], :].T
            concat_xy = torch.cat((concat_xy, other_factor_dd), 1)

        nan_bool_index = torch.isnan(concat_xy).any(1)  # [C, 1]
        concat_xy = concat_xy[~nan_bool_index, :]  # [C, 2+28+n]
        if concat_xy.shape[0] >= 10:
            x = concat_xy[:, 1:]  # [C, 1+29]
            y = concat_xy[:, 0]  # [C, 1]
            x = torch.cat((x, torch.ones(x.shape[0], 1).to(alphan_value_gp.device)), 1)  # [C, 1+28+n+1]
            resids = (y - torch.matmul(x, torch.matmul(torch.matmul(torch.inverse(torch.matmul(x.T, x)), x.T), y))).T  # [1, C]
            resids_all[[dd], ~nan_bool_index] = resids
    return resids_all


def neutralize_numpy_(alphan_value_gp, size_gp, cs_indus_code, other_factor_list=[], silent=True):  # [tensor (TS*C), tensor (TS*C)]
    assert (alphan_value_gp.shape == size_gp.shape)
    alphan_value_gp = alphan_value_gp.values
    size_gp = size_gp.values
    cs_indus_code = cs_indus_code.values
    resids_all = np.zeros_like(alphan_value_gp) * np.nan
    for dd in range(alphan_value_gp.shape[0]):
        if dd%500 == 0 and not silent: print('neutralize by tradedate',dd)
        y = alphan_value_gp[[dd], :].T  # [C, 1]
        concat_xy = y

        if len(other_factor_list) == 0:
            size_dd = size_gp[[dd], :].T  # [C, 1]
            concat_xy = np.concatenate((concat_xy, size_dd), axis=1)  # [C, 2]
            cs = cs_indus_code[[dd], :].T  # [C, 1]
            for cs_index in range(1, 29 + 1 - 1):  # 29个一级行业代码，去掉最后一列避免线性相关
                cs_dummy = np.zeros_like(cs)  # [C, 1]
                cs_dummy[cs == cs_index] = 1
                if cs_dummy.sum() == 0:
                    continue
                concat_xy = np.concatenate((concat_xy, cs_dummy), axis=1)  # [C, 2+28+n]
        else:
            for factor_num, other_factor in enumerate(other_factor_list):
                other_factor_dd = other_factor[[dd], :].T
                #对缺失值用每个时间截面的均值填充
                other_factor_dd = np.where(np.isnan(other_factor_dd), np.nanmean(other_factor_dd), other_factor_dd)
                concat_xy = np.concatenate((concat_xy, other_factor_dd), axis=1)

        nan_bool_index = np.isnan(concat_xy).any(1)  # [C, 1]
        concat_xy = concat_xy[~nan_bool_index, :]  # [C, 2+28+n]
        concat_xy = np.concatenate((concat_xy, np.ones((concat_xy.shape[0],1))),
                              axis=1)  # [C, 1+28+n+1]
        #concat_xy = concat_xy.cpu().numpy()

        if concat_xy.shape[0] >= 10:
            x = concat_xy[:, 1:]  # [C, 1+29]
            y = concat_xy[:, 0]  # [C, 1]
            try:
                resids = (y - x @ np.linalg.lstsq(x, y, rcond=None)[0]).T  # [1, C]
            except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
                beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                resids = (y - x @ beta).T
                
            resids_all[[dd], ~nan_bool_index] = resids

    return resids_all


def neutralize_numpy(alphan_value_gp, size_gp, cs_indus_code, other_factor_list=[], silent= True):  # [tensor (TS*C), tensor (TS*C)]
    assert (alphan_value_gp.shape == size_gp.shape)
    resids_all = torch.zeros_like(alphan_value_gp) * np.nan
    for dd in range(alphan_value_gp.shape[0]):
        if dd%500 == 0 and not silent: print('neutralize by tradedate',dd)
        y = alphan_value_gp[[dd], :].T  # [C, 1]
        concat_xy = y

        if len(other_factor_list) == 0:
            size_dd = size_gp[[dd], :].T  # [C, 1]
            concat_xy = torch.cat((concat_xy, size_dd), 1)  # [C, 2]
            cs = cs_indus_code[[dd], :].T  # [C, 1]
            for cs_index in range(1, 29 + 1 - 1):  # 29个一级行业代码，去掉最后一列避免线性相关
                cs_dummy = torch.zeros_like(cs)  # [C, 1]
                cs_dummy[cs == cs_index] = 1
                if cs_dummy.sum() == 0:
                    continue
                concat_xy = torch.cat((concat_xy, cs_dummy), 1)  # [C, 2+28+n]
        else:
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
                try:    
                    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                    resids = (y - x @ beta).T
                except:
                    print('neutralization error!')
                    resids = alphan_value_gp.copy()
                    
            resids_all[[dd], ~nan_bool_index] = torch.Tensor(resids).to(alphan_value_gp.device)

    return resids_all


# 用梯度下降做市值中性化
from torch import nn

def neutralize_torch_nn(alphan_value_gp, size_gp, cs_indus_code, other_factor_list=[]):  # [tensor (TS*C), tensor (TS*C)]
    assert (alphan_value_gp.shape == size_gp.shape)
    resids_all = torch.zeros_like(alphan_value_gp) * np.nan
    for dd in range(alphan_value_gp.shape[0]):
        y = alphan_value_gp[[dd], :].T  # [C, 1]
        x = size_gp[[dd], :].T  # [C, 1]
        concat_xy = torch.cat((y, x), 1)  # [C, 2]
        cs = cs_indus_code[[dd], :].T  # [C, 1]
        for cs_index in range(1, 29 + 1 - 1):  # 29个一级行业代码，去掉最后一列避免线性相关
            cs_dummy = torch.zeros_like(cs)  # [C, 1]
            cs_dummy[cs == cs_index] = 1
            if cs_dummy.sum() == 0:
                continue
            concat_xy = torch.cat((concat_xy, cs_dummy), 1)  # [C, 2+28+n]
        for factor_num, other_factor in enumerate(other_factor_list):
            other_factor_dd = other_factor[[dd], :].T
            concat_xy = torch.cat((concat_xy, other_factor_dd), 1)

        nan_bool_index = torch.isnan(concat_xy).any(1)  # [C, 1]
        concat_xy = concat_xy[~nan_bool_index, :]  # [C, 2+28+n]
        if concat_xy.shape[0] >= 10:
            x = concat_xy[:, 1:]  # [C, 1+28+n]
            y = concat_xy[:, 0]  # [C, 1]
            x = torch.cat((x, torch.ones(x.shape[0], 1).to(alphan_value_gp.device)), 1)  # [C, 1+28+n+1]

            model = nn.Linear(in_features=1, out_features=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            # 训练模型
            num_epochs = 1000
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # 清空之前计算得到的梯度信息
                outputs = model(x)  # 通过模型预测结果
                loss = criterion(outputs, y)  # 计算损失
                loss.backward()  # 反向传播更新参数
                optimizer.step()  # 应用梯度更新

            resids = (y - model(x)).T  # [1, C]

            resids_all[[dd], ~nan_bool_index] = resids
    return resids_all


#%% 相关系数函数
def corrwith_torch(x,y):
    assert (x.shape == y.shape)
    corr_all = torch.zeros(x.shape[0], 1).to(x.device)
    x = x + y * 0
    y = y + x * 0
    for dd in range(x.shape[0]):
        x_ = x[[dd], :]  # [1, C]
        y_ = y[[dd], :]  # [1, C]
        x_xmean = x_ - torch.nanmean(x_, 1, keepdim=True)  # [1, C]
        y_ymean = y_ - torch.nanmean(y_, 1, keepdim=True)  # [1, C]
        corr = torch.nansum(x_xmean * y_ymean, 1) / (
                    torch.sqrt(torch.nansum(x_xmean ** 2, 1)) * torch.sqrt(torch.nansum(y_ymean ** 2, 1)))  # [1, 1]
        corr_all[dd] = corr

    return corr_all


# 将以上函数以矩阵形式改写
def corrwith_torch_matrix(x,y,dim):
    assert (x.shape == y.shape)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    corr = torch.nansum(x_xmean * y_ymean, dim) / (
                torch.sqrt(torch.nansum(x_xmean ** 2, dim)) * torch.sqrt(torch.nansum(y_ymean ** 2, dim)))  # [TS, 1]
    return corr


def covariance_torch_matrix(x,y,dim):
    assert (x.shape == y.shape)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    corr = torch.nanmean(x_xmean * y_ymean, dim)  # [TS, 1]
    return corr


#%% 其他函数
def ts_rolling_torch(x, roll_num, func, **kwargs):
    assert (len(x.shape) == 2)
    assert (roll_num > 0)
    assert (roll_num <= x.shape[0])

    # assert (len(x) == len(y))
    z = torch.zeros_like(x) * np.nan

    # for j in (range(x.shape[1])):
    #     tmp = np.concatenate((x[:,j:j+1], y[:,j:j+1]), axis=1)
    for i in range(x.shape[0]):
        if i >= roll_num - 1:
            tmp = x[i-roll_num+1:i+1,:]
            try:
                z[i,:] = func(tmp, **kwargs)
            except TypeError:
                z[i,:] = func(tmp, **kwargs)[0]
            except RuntimeError:
                z[i,:] = func(tmp, **kwargs)[[-1]]
    return z

def ts_corolling_torch(x, y, roll_num, func, **kwargs):
    assert (x.shape == y.shape)
    assert (len(x.shape) == 2)
    assert (roll_num > 0)
    assert (roll_num <= x.shape[0])

    z = torch.zeros_like(x) * np.nan
    for i in range(x.shape[0]):
        if i >= roll_num - 1:
            tmp_x = x[i-roll_num+1:i+1,:]
            tmp_y = y[i-roll_num+1:i+1,:]
            z[i,:] = func(tmp_x, tmp_y, **kwargs)
    return z

def rank_pct_torch(x,dim=1):
    assert (len(x.shape) == 2)
    assert (dim in [0,1])  # 0:按列排序，1：按行排序
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32)
    x_rank[x.isnan()] = np.nan
    # 做percentile处理
    x_rank = (x_rank+1) / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

def decay_linear_igrnan_torch(x):   #only for rolling
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    # z = torch.zeros_like(x) * np.nan   # [d,C]  np.nan

    z_arrange = torch.arange(1, x.shape[0]+1, 1).unsqueeze(-1).to(torch.float32).to(x.device)  # [d]

    return (x * z_arrange).nansum(dim=0) / ((z_arrange * (~x.isnan())).sum(dim=0))

def ts_decay_linear_igrnan_torch(x, d):
    assert (len(x.shape) == 2)
    assert (d > 0)
    assert (d <= x.shape[0])

    return ts_rolling_torch(x, d, decay_linear_igrnan_torch)

def ts_SubPosDecayLinear_torch(x, y, d):
    value = x - y
    value[value < 0] = 0
    return ts_decay_linear_igrnan_torch(value, d)

def ts_correlation_torch(x, y, d):
    assert (x.shape == y.shape)
    assert (len(x.shape) == 2)
    assert (d > 0)
    assert (d <= x.shape[0])

    return ts_corolling_torch(x, y, d, corrwith_torch_matrix, dim=0)

def ts_covariance_torch(x, y, d):
    assert (x.shape == y.shape)
    assert (len(x.shape) == 2)
    assert (d > 0)
    assert (d <= x.shape[0])

    return ts_corolling_torch(x, y, d, covariance_torch_matrix, dim=0)

def ts_rankcorr_torch(x, y, d):
    assert (x.shape == y.shape)
    assert (len(x.shape) == 2)
    assert (d > 0)
    assert (d <= x.shape[0])

    return ts_corolling_torch(rank_pct_torch(x), rank_pct_torch(y), d, corrwith_torch_matrix, dim=0)

def sign_torch(x):
    return torch.sign(x)

def ts_delay_torch(x, d):
    assert (len(x.shape) == 2)
    assert (abs(d) <= x.shape[0])

    z = x.roll(d, dims=0)
    if d >= 0:
        z[:d,:] = np.nan
    else:
        z[d:,:] = np.nan
    return z

def ts_delta_torch(x, d):
    assert (len(x.shape) == 2)
    assert (d > 0)
    assert (d <= x.shape[0])

    z = x - ts_delay_torch(x, d)
    return z

def scale_torch(x, c=1):
    assert (len(x.shape) == 2)
    assert (c > 0)

    return (x * c)/(abs(x).nansum(axis=1, keepdim=True))

def signedpower_torch(x, a):
    assert (len(x.shape) == 2)
    assert (a > 0)

    return (x ** a)

def ts_min_torch(x, d):
    x = x.nan_to_num(np.inf)
    return ts_rolling_torch(x, d, torch.min, dim=0, keepdim=True)

def ts_max_torch(x, d):
    x = x.nan_to_num(-np.inf)
    return ts_rolling_torch(x, d, torch.max, dim=0, keepdim=True)

def ts_argmin_torch(x, d):
    x = x.nan_to_num(np.inf)
    return ts_rolling_torch(x, d, torch.argmin, dim=0, keepdim=True)

def ts_argmax_torch(x, d):
    x = x.nan_to_num(-np.inf)
    return ts_rolling_torch(x, d, torch.argmax, dim=0, keepdim=True)

def ts_rank_torch(x, d):
    return ts_rolling_torch(x, d, rank_pct_torch, dim=0)

def sigmoid_torch(x):
    return 1 / (1 + torch.exp(-x))

def ts_stddev_torch(x, d):
    return ts_rolling_torch(x, d, torch.std, dim=0, keepdim=True)

def ts_sum_torch(x, d):
    return ts_rolling_torch(x, d, torch.sum, dim=0, keepdim=True)

def ts_product_torch(x, d):
    return ts_rolling_torch(x, d, torch.prod, dim=0, keepdim=True)

def rank_sub_torch(x,y):
    return rank_pct_torch(x,1) - rank_pct_torch(y,1)

def rank_add_torch(x,y):
    return rank_pct_torch(x,1) + rank_pct_torch(y,1)

def rank_div_torch(x,y):
    return rank_pct_torch(x,1) / rank_pct_torch(y,1)

def log_torch(x):
    return torch.log(x)

def sqrt_torch(x):
    return torch.sqrt(x)

def ts_delaypct_torch(x,d):
    return (x - ts_delay_torch(x,d)) / ts_delay_torch(x,d)

def rlb_torch(x, y, d, n, btm, sel_posneg=False):
    assert (x.shape == y.shape)
    z = torch.zeros_like(x) * np.nan
    for i in range(x.shape[0]):
        if i >= d - 1:
            # tmp_i = tmp.iloc[i-d+1:i+1,:]
            # tmp_i = tmp_i.sort_values(by='y')
            # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
            tmp_x = x[i - d + 1:i + 1, :]
            tmp_y = y[i - d + 1:i + 1, :]
            if btm == 'btm':
                z[i,:] = torch.where(tmp_x<=tmp_x.kthvalue(n, dim=0, keepdim=True)[0],tmp_y,torch.nan).nanmean(dim=0)  #kth是从1开始的
            elif btm == 'diff':
                z[i,:] = (torch.where(tmp_x>=tmp_x.kthvalue(d-n+1, dim=0, keepdim=True)[0],tmp_y,torch.nan).nanmean(dim=0) -
                          torch.where(tmp_x<=tmp_x.kthvalue(n, dim=0, keepdim=True)[0],tmp_y,torch.nan).nanmean(dim=0))
            elif btm == 'top':
                z[i,:] = torch.where(tmp_x>=tmp_x.kthvalue(d-n+1, dim=0, keepdim=True)[0],tmp_y,torch.nan).nanmean(dim=0)
            else:
                assert (False)
        # z[col] = (tmp.rolling(d, method='table').apply(lambda df: df[df[:,1].argsort()][:n,0].mean(), engine='numba', raw=True))['x']  #argsort是升序排，小的在上，大的在下
    return z

def rlb_torch_slow(x, y, d, n, btm, sel_posneg=False):
    assert (x.shape == y.shape)
    z = torch.zeros_like(x) * np.nan
    for j in (range(x.shape[1])):
        tmp = torch.cat((x[:, [j]], y[:, [j]]), dim=1)
        for i in range(len(tmp)):
            if i >= d - 1:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = tmp[i - d + 1:i + 1, :]
                tmp_i_new = tmp_i[tmp_i[:, 1].argsort(),:]
                if btm == 'btm':
                    z[i, j] = tmp_i_new[:n, 0].nanmean()
                elif btm == 'diff':
                    z[i, j] = tmp_i_new[-n:, 0].nanmean() - tmp_i_new[:n, 0].nanmean()
                elif btm == 'top':
                    z[i, j] = tmp_i_new[-n:, 0].nanmean()
                else:
                    assert (False)
        # z[col] = (tmp.rolling(d, method='table').apply(lambda df: df[df[:,1].argsort()][:n,0].mean(), engine='numba', raw=True))['x']  #argsort是升序排，小的在上，大的在下
    return z

def ts_grouping_ascsortavg_torch(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最小n个x的平均值'
    n = min(d, n)
    return rlb_torch(x, y, d, n, btm='btm')

def ts_grouping_decsortavg_torch(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值'
    n = min(d, n)
    return rlb_torch(x, y, d, n, btm='top')

def ts_grouping_diffsortavg_torch(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值与最小n个x的平均值的差值'
    n = min(d, n)
    return rlb_torch(x, y, d, n, btm='diff')

    #         if i < d - 1:
    #             z[i, j] = np.nan
    #         else:
    #             # tmp_i = tmp.iloc[i-d+1:i+1,:]
    #             # tmp_i = tmp_i.sort_values(by='y')
    #             # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
    #             tmp_i = tmp[i-d+1:i+1,:]
    #             tmp_i_new = tmp_i[tmp_i[:,1].argsort()]
    #             if btm=='btm':
    #                 z[i, j] = tmp_i_new[:n,0].mean()
    #             elif btm=='diff':
    #                 z[i, j] = tmp_i_new[-n:,0].mean() - tmp_i_new[:n, 0].mean()
    #             elif btm=='top':
    #                 z[i, j] = tmp_i_new[-n:,0].mean()
    #             else:
    #                 assert(False)
    #     # z[col] = (tmp.rolling(d, method='table').apply(lambda df: df[df[:,1].argsort()][:n,0].mean(), engine='numba', raw=True))['x']  #argsort是升序排，小的在上，大的在下
    # return z



#%% 原始函数
# _______以下为原始代码，未修改为torch形式_________

def correlation(x,y,d):
    '过去d日的x和过去d日的y时间序列相关系数'
    return x.rolling(d).corr(y)

def covariance(x, y, d):
    '过去d日的x和过去d日的y时间序列协方差'
    return x.rolling(d).cov(y)

def rank(x):
    '按照变量 x 截面值进行排序的序号'
    return x.rank(axis=1, pct=True)

def rankcorr(x,y,d):
    '过去d日的x和过去d日的y时间序列相关系数'
    return rank(x).rolling(d).corr(rank(y))

def sign(x):
    '符号函数'
    return np.sign(x)

def delay(x, d):
    '过去d日的x'
    return x.shift(d)

def scale(x, a=1):
    return (x * a) .div(abs(x).sum(axis=1), axis=0)

def delta(x, d):
    '变量 x 截面值与 d 日前截面值差值'
    return x.diff(d)

def signedpower(x, a):
    return x ** a #np.power(x,a)

def decay_linear(x, d):
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    w = np.arange(1,d+1,1)
    return x.rolling(d).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)  #, engine='numba'

def ts_min(x, d):
    '过去d日的x的最小值'
    return x.rolling(d).min()

def ts_max(x, d):
    '过去d日的x的最大值'
    return x.rolling(d).max()

def ts_arg_min(x, d):
    '过去d日的x的最小值对应的序号'
    return x.rolling(d).apply(np.argmin, raw=True)

def ts_arg_max(x, d):
    '过去d日的x的最大值对应的序号'
    return x.rolling(d).apply(np.argmax, raw=True)

def ts_rank(x, d):
    '过去d日的x的排名序号'
    # return x.rolling(d).apply(lambda x: x.rank().iloc[-1])
    return x.rolling(d).rank(pct=True)

def ts_zscore(x, d):
    '过去d日的x的z分数'
    return x.rolling(d).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def stddev(x, d ):
    '过去d日的x的标准差'
    return x.rolling(d).std()

def ts_sum(x, d):
    '过去d日的x的和'
    return x.rolling(d).sum()
    # return x.rolling(d).sum()

def product(x, d ):
    '过去d日的x的乘积'
    # if isinstance(x, float):
    #     return None
    return x.rolling(d).apply(np.prod, raw=True)

def rank_add(x, y):
    'x+y的序号'
    return x.rank(axis=1, pct=True) + y.rank(axis=1, pct=True)

def rank_sub(x, y):
    'x-y的序号'
    return x.rank(axis=1, pct=True) - y.rank(axis=1, pct=True)

def rank_div(x, y):
    'x/y的序号'
    return x.rank(axis=1, pct=True) / y.rank(axis=1, pct=True)

def add(x, y):
    'x+y'
    return x + y

def sub(x, y):
    'x-y'
    return x - y

def mul(x, y):
    'x*y'
    return x * y

def div(x, y):
    'x/y'
    return x / y

def add_int(x, y):
    'x+y'
    return x + y

def sub_int1(x, y):
    'x-y'
    return x - y

def sub_int2(x, y):
    'x-y'
    return x - y

def mul_int(x, y):
    'x*y'
    return x * y

def div_int1(x, y):
    'x/y'
    return x / y

def div_int2(x, y):
    'x/y'
    return x / y

def log(x):
    'log(x)'
    return np.log(x)

def sqrt(x):
    'sqrt(x)'
    return np.sqrt(x)

def neg(x):
    '-x'
    return -x

def neg_int(x):
    '-x'
    return -x

# @cuda.jit(device=True)
def numba_rlb(x, y, d, n, btm, sel_posneg=False):
    assert (len(x) == len(y))
    z = np.zeros_like(x)
    for j in (range(x.shape[1])):
        tmp = np.concatenate((x[:,j:j+1], y[:,j:j+1]), axis=1)
        for i in range(len(tmp)):
            if i < d - 1:
                z[i, j] = np.nan
            else:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = tmp[i-d+1:i+1,:]
                tmp_i_new = tmp_i[tmp_i[:,1].argsort()]
                if btm=='btm':
                    z[i, j] = tmp_i_new[:n,0].mean()
                elif btm=='diff':
                    z[i, j] = tmp_i_new[-n:,0].mean() - tmp_i_new[:n, 0].mean()
                elif btm=='top':
                    z[i, j] = tmp_i_new[-n:,0].mean()
                else:
                    assert(False)
        # z[col] = (tmp.rolling(d, method='table').apply(lambda df: df[df[:,1].argsort()][:n,0].mean(), engine='numba', raw=True))['x']  #argsort是升序排，小的在上，大的在下
    return z

def rolling_selmean_btm(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最小n个x的平均值'
    if isinstance(x, float):
        x = pd.DataFrame(np.ones_like(y) * x, index=y.index, columns=y.columns)
    if isinstance(y, float):
        y = pd.DataFrame(np.ones_like(x) * y, index=x.index, columns=x.columns)
    n = min(d, n)
    numba_values = numba_rlb(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='btm')
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

def rolling_selmean_top(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值'
    if isinstance(x, float):
        x = pd.DataFrame(np.ones_like(y) * x, index=y.index, columns=y.columns)
    if isinstance(y, float):
        y = pd.DataFrame(np.ones_like(x) * y, index=x.index, columns=x.columns)
    n = min(d, n)
    numba_values = numba_rlb(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='top')
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

def rolling_selmean_diff(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值与最小n个x的平均值的差值'
    if isinstance(x, float):
        x = pd.DataFrame(np.ones_like(y) * x, index=y.index, columns=y.columns)
    if isinstance(y, float):
        y = pd.DataFrame(np.ones_like(x) * y, index=x.index, columns=x.columns)
    n = min(d, n)
    numba_values = numba_rlb(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='diff')
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

# @cuda.jit(device=True)

@jit(nopython=True)
def decay_linear_igrnan(x_np, d):
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    z = np.zeros_like(x_np)
    for j in (range(x_np.shape[1])):
        tmp = x_np[:, j]
        for i in range(len(tmp)):
            if i < d - 1:
                z[i, j] = np.nan
            else:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = tmp[i - d + 1:i + 1]
                tmp_i_exnan = np.zeros_like(tmp_i)
                k_this = 0
                for k in range(len(tmp_i)):
                    if not np.isnan(tmp_i[k]):
                        tmp_i_exnan[k_this] = tmp_i[k]
                        k_this += 1
                tmp_i_exnan = tmp_i_exnan[:k_this]
                if len(tmp_i_exnan)==0:
                    z[i, j] = np.nan
                else:
                    w = np.arange(1,len(tmp_i_exnan)+1,1, dtype=np.float64)
                    # w = 0.8 ** (np.arange(len(tmp_i),0,-1))
                    z[i,j] = np.dot(tmp_i_exnan,w)/w.sum()
    return z

def SubPosDecayLinear(x, y, d):
    value = x - y
    value[value < 0] = 0
    value_np = np.array(value.values, dtype=np.float64)
    return pd.DataFrame(decay_linear_igrnan(value_np, d), index=value.index, columns=value.columns)

def delaypct(x,d):
    'x的d日前与当天相比的涨跌幅'
    return x/x.shift(d) - 1

def numba_rlb1(x, y, d, n, btm, sel_posneg=False):
    assert (len(x) == len(y))
    z = np.zeros_like(x)
    for j in (range(x.shape[1])):
        tmp = np.concatenate((x[:,j:j+1], y[:,j:j+1]), axis=1)
        for i in range(len(tmp)):
            if i < d - 1:
                z[i, j] = np.nan
            else:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = tmp[i-d+1:i+1,:]
                tmp_i_new = tmp_i[tmp_i[:,1].argsort()]
                if btm=='btm':
                    if sel_posneg:
                        tmp_i_new_1 = tmp_i_new[tmp_i_new[:,0]<0]
                        z[i, j] = tmp_i_new_1[:n,0].mean()
                    else:
                        z[i, j] = tmp_i_new[:n,0].mean()
                elif btm=='diff':
                    if sel_posneg:
                        tmp_i_new_1 = tmp_i_new[tmp_i_new[:,0] < 0]
                        tmp_i_new_2 = tmp_i_new[tmp_i_new[:,0] > 0]
                        z[i, j] = tmp_i_new_2[-n:,0].mean() - tmp_i_new_1[:n, 0].mean()
                    else:
                        z[i, j] = tmp_i_new[-n:,0].mean() - tmp_i_new[:n, 0].mean()
                elif btm=='top':
                    if sel_posneg:
                        tmp_i_new_2 = tmp_i_new[tmp_i_new[:,0] > 0]
                        z[i, j] = tmp_i_new_2[-n:,0].mean()
                    else:
                        z[i, j] = tmp_i_new[-n:,0].mean()
                else:
                    assert(False)
        # z[col] = (tmp.rolling(d, method='table').apply(lambda df: df[df[:,1].argsort()][:n,0].mean(), engine='numba', raw=True))['x']  #argsort是升序排，小的在上，大的在下
    return z

def rolling_selmean_btm_sel_posneg(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最小n个x的平均值'
    numba_values = numba_rlb1(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='btm', sel_posneg=True)
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

def rolling_selmean_top_sel_posneg(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值'
    numba_values = numba_rlb1(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='top', sel_posneg=True)
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

def rolling_selmean_diff_sel_posneg(x, y, d, n):
    '在过去d日上，根据y的值对x进行排序，取最大n个x的平均值与最小n个x的平均值的差值'
    numba_values = numba_rlb1(np.array(x.values,dtype=np.float64), np.array(y.values,dtype=np.float64), d, n, btm='diff', sel_posneg=True)
    return pd.DataFrame(numba_values, index=x.index, columns=x.columns)

