# -*- coding: utf-8 -*-
"""
GPU demo

实验一
0.041457176208496094   cupy
0.02668929100036621    cupy+numbacuda(vectorize)
0.0076291561126708984  cupy+numbacuda(jit)

0.04178261756896973
0.026834487915039062
0.007462024688720703

实验二
5.4 it/s                cupy
1.6689300537109375e-06  cupy+numbacuda

5.4 it/s
1.430511474609375e-06

"""

from numba import jit
from numba import cuda
import pandas as pd
import numpy as np
import cupy as cp
import time
from numba import vectorize
#import torch
from tqdm import tqdm

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
print(device)


#%% 
price = pd.DataFrame(np.random.randn(5000,5000))
#price_cuda = torch.Tensor(np.array(price,dtype='float32')).to(device)


#%% 实验一：简单相加（六个矩阵相加）
print('exp1')

# cupy直接操作
def array_plus(a,b):
    return a+b#+a+b+a+b

open_array = cp.array(price,dtype='float32')
timenow = time.time()
for i in range(10):
    OUTPUT = array_plus(open_array,open_array)
    OUTPUT
print(time.time()-timenow)


# cupy+numbacuda(vectorize)
@vectorize(["float32(float32, float32)"], target='cuda')
def numba_plus(a,b):
    return a+b#+a+b+a+b

open_array = cp.array(price,dtype='float32')
numba_plus(open_array,open_array)
timenow = time.time()
for i in range(10):
    OUTPUT = numba_plus(open_array,open_array)
    OUTPUT
print(time.time()-timenow)


# cupy+numbacuda(cuda.jit)
@cuda.jit
def numba_plus(a,b,output):
    i_start,j_start = cuda.grid(2)
    i_step,j_step = cuda.gridsize(2)
    for j in (range(j_start,a.shape[1],j_step)):
        for i in (range(i_start,a.shape[0],i_step)):
            output[i][j] = a[i][j]+b[i][j]#+a[i][j]+b[i][j]+a[i][j]+b[i][j]

open_array = cuda.to_device(cp.array(price,dtype='float32'))
output = cuda.to_device(cp.zeros_like(cp.array(price)))
numba_plus[(64, 64),(16,16)](open_array,open_array,output)
timenow = time.time()
for i in range(10):
    # output = cp.zeros_like(open_array)
    numba_plus[(64, 64),(16,16)](open_array,open_array,output)
    cuda.synchronize()
    OUTPUT
print(time.time()-timenow)


#%% 实验二：衰减平均
# cupy直接操作
def decay_linear_igrnan1(x_np, d):
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    z = cp.zeros_like(x_np)
    for j in tqdm(range(x_np.shape[1])):
        tmp = x_np[:, j]
        for i in range(len(tmp)):
            if i < d - 1:
                z[i, j] = cp.nan
            else:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = tmp[i - d + 1:i + 1]
                tmp_i_exnan = cp.zeros_like(tmp_i)
                k_this = 0
                for k in range(len(tmp_i)):
                    if not cp.isnan(tmp_i[k]):
                        tmp_i_exnan[k_this] = tmp_i[k]
                        k_this += 1
                tmp_i_exnan = tmp_i_exnan[:k_this]
                if len(tmp_i_exnan)==0:
                    z[i, j] = cp.nan
                else:
                    w = cp.arange(1,len(tmp_i_exnan)+1,1, dtype=np.float64)
                    # w = 0.8 ** (np.arange(len(tmp_i),0,-1))
                    z[i,j] = cp.dot(tmp_i_exnan,w)/w.sum()
    return z
price_array = cp.array(price,dtype='float32')
timenow = time.time()
#decay_linear_igrnan1(price_array, 10)
print(time.time()-timenow)


# cupy+numbacuda
@cuda.jit  #(device=True)
def decay_linear_igrnan0(x_np, d,w,z):
    'd日衰减加权平均，加权系数为 d, d-1,...,1'

    i_start,j_start = cuda.grid(2)
    i_step,j_step = cuda.gridsize(2)
    for j in (range(j_start,x_np.shape[1],j_step)):
        for i in (range(i_start,x_np.shape[0],i_step)):

            if i < d - 1:
                z[i, j] = np.nan
            else:
                # tmp_i = tmp.iloc[i-d+1:i+1,:]
                # tmp_i = tmp_i.sort_values(by='y')
                # z.iloc[i, j] = tmp_i.iloc[:n,0].mean()
                tmp_i = x_np[i - d + 1:i + 1, j]
                # tmp_i_exnan = cp.zeros(tmp_i.shape[0],tmp_i.shape[1])
                # k_this = 0
                # for k in range(len(tmp_i)):
                #     if not np.isnan(tmp_i[k]):
                #         tmp_i_exnan[k_this] = tmp_i[k]
                #         k_this += 1
                # tmp_i_exnan = tmp_i_exnan[:k_this]
                # if len(tmp_i_exnan)==0:
                #     z[i, j] = np.nan
                # else:
                #     w = cp.arange(1,len(tmp_i_exnan)+1,1, dtype=cp.float64)
                #     # w = 0.8 ** (np.arange(len(tmp_i),0,-1))
                #     z[i,j] = cp.dot(tmp_i_exnan,w)/w.sum()
                # z[i,j] = tmp_i.sum(axis=0)
                w_sum = 0
                k_sum = 0
                for k in range(d):
                    if tmp_i[k]==tmp_i[k]:  #即tmp_i[k] is not nan
                        w_sum += w[k]
                        k_sum += tmp_i[k] * w[k]
                z[i,j] = k_sum / w_sum
                
z = cuda.to_device(cp.zeros_like(cp.array(price)))
w = cuda.to_device(cp.arange(1,10+1,1, dtype=cp.float32))

decay_linear_igrnan0[(64, 64),(16,16)](cuda.to_device(cp.array(price)), 10,w,z)

timenow = time.time()
decay_linear_igrnan0[(64, 64),(16,16)](cuda.to_device(cp.array(price)), 10,w,z)
cuda.synchronize()
print(time.time()-timenow)


     