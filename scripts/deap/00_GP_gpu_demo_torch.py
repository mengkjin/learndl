# -*- coding: utf-8 -*-
"""
GPU demo

实验一
0.12849640846252441     numpy
0.09869718551635742     pandas
0.043421268463134766    torch+cpu
0.002076864242553711    torch+cuda

0.12733721733093262
0.10813021659851074
0.04917788505554199
0.0014736652374267578

实验二
166.66732501983643  numpy
2.919011354446411   numpy+numba
0.5333425998687744  torch+cuda

170.1861231327057
2.8308353424072266
0.49471592903137207

实验三
0.21473336219787598   pandas
0.4073503017425537    torch+cpu
1.5067682266235352    torch+cuda
0.08215570449829102   torch+cpu+matrix
0.0015797615051269531 torch+cuda+matrix

0.2359161376953125
0.4897339344024658
1.6294970512390137
0.08276009559631348
0.0015001296997070312

"""
#%% 
from numba import jit
from numba import cuda
import pandas as pd
import numpy as np
#import cupy as cp
import time
from numba import vectorize
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#%% 
price = pd.DataFrame(np.random.randn(5000,5000))
price_cuda = torch.Tensor(np.array(price,dtype='float32')).to(device)


#%% 实验一：简单相加（六个矩阵相加）
print('exp1')

# numpy直接操作
def array_plus(a,b):
    return a+b#+a+b+a+b

open_array = np.array(price,dtype='float32')
timenow = time.time()
for i in range(10):
    OUTPUT = array_plus(open_array,open_array)
    # print(OUTPUT)
print(time.time()-timenow)


#DataFrame直接相加
def df_plus(a,b):
    return a+b#+a+b+a+b

timenow = time.time()
for i in range(10):
    OUTPUT = df_plus(price,price)
    # print(OUTPUT)
print(time.time()-timenow)


#torch+cpu
def torch_plus(a,b): #type: ignore
    return a+b#+a+b+a+b

price_cuda = torch.tensor(np.array(price,dtype='float32')).cpu()
timenow = time.time()
for i in range(10):
    # OUTPUT = price_cuda + price_cuda  + price_cuda  + price_cuda + price_cuda + price_cuda
    OUTPUT = torch_plus(price_cuda,price_cuda)
    #print(OUTPUT)
print(time.time()-timenow)


#torch+cuda
def torch_plus(a,b):
    return a+b#+a+b+a+b

price_cuda = torch.tensor(np.array(price,dtype='float32')).to(device)
timenow = time.time()
for i in range(10):
    # OUTPUT = price_cuda + price_cuda  + price_cuda  + price_cuda + price_cuda + price_cuda
    OUTPUT = torch_plus(price_cuda,price_cuda)
    #print(OUTPUT)
print(time.time()-timenow)


#%% 实验二：衰减平均
print('exp2')

# numpy直接操作
def decay_linear_igrnan0(x_np, d):
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    z = np.zeros_like(x_np)
    for j in tqdm(range(x_np.shape[1])):
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
price_array = np.array(price,dtype='float32')
timenow = time.time()
#decay_linear_igrnan0(price_array, 10)
print(time.time()-timenow)


# numpy+numba
@jit(nopython=True)
def decay_linear_igrnan1(x_np, d):
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
                    w = np.arange(1,len(tmp_i_exnan)+1,1, dtype=np.float32)
                    # w = 0.8 ** (np.arange(len(tmp_i),0,-1))
                    z[i,j] = np.dot(tmp_i_exnan,w)/w.sum()
    return z
price_array = np.array(price,dtype='float32')
timenow = time.time()
decay_linear_igrnan1(price_array, 10)
print(time.time()-timenow)


# torch+cuda
def ts_rolling_torch(x, roll_num, func, **kwargs):
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

timenow = time.time()
z_torch = ts_decay_linear_igrnan_torch(price_cuda,10)
print(time.time()-timenow)


#%% 实验三 correlation
print('exp3')

def corrwith_torch(x,y):
    assert (x.shape == y.shape)
    corr_all = torch.zeros(x.shape[0], 1)
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


#将以上函数以矩阵形式改写
def corrwith_torch_matrix(x,y):
    assert (x.shape == y.shape)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, 1, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, 1, keepdim=True)  # [TS, C]
    corr = torch.nansum(x_xmean * y_ymean, 1) / (
                torch.sqrt(torch.nansum(x_xmean ** 2, 1)) * torch.sqrt(torch.nansum(y_ymean ** 2, 1)))  # [TS, 1]
    return corr


# pandas
timenow = time.time()
price.corrwith(price*2,axis=1)
print(time.time()-timenow)


# torch+cpu+未经矩阵化改写
price_cuda = price_cuda.cpu()
timenow = time.time()
corrwith_torch(price_cuda,price_cuda*2)
print(time.time()-timenow)


# torch+cuda+未经矩阵化改写
price_cuda = price_cuda.to(device)
timenow = time.time()
corrwith_torch(price_cuda,price_cuda*2)
print(time.time()-timenow)


# torch+cpu+矩阵化改写
price_cuda = price_cuda.cpu()
timenow = time.time()
corrwith_torch_matrix(price_cuda,price_cuda*2)
print(time.time()-timenow)


# torch+cuda+矩阵化改写
price_cuda = price_cuda.to(device)
timenow = time.time()
corrwith_torch_matrix(price_cuda,price_cuda*2)
print(time.time()-timenow)

