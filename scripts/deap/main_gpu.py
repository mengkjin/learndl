'''
---------环境配置要求---------------
deap==1.3.1，不建议采用1.4版本（可能会报错）
torch>=1.12.0，并确保cuda可用
parquet文件读取需要安装fastparquet，版本不限

'''
#%%
import sys
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
import numpy as np
import array
import random
import json
import operator
import time
import platform
plat = platform.system().lower()

from deap import base,creator,tools,gp,algorithms
from deap.algorithms import varAnd
from torch.multiprocessing import Pool

from tqdm import tqdm

import joblib
from math_func_gpu import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print('cuda_available',torch.cuda.is_available())


#%% 并行参数初始化，无需修改
if plat == 'linux':
    #assert(device == torch.device('cuda'))
    print('device:',device, torch.cuda.get_device_name(0), torch.cuda.is_available())
    # 单机多进程设置（实际未启用），参考https://zhuanlan.zhihu.com/p/600801803
    torch.multiprocessing.set_start_method('forkserver',force = True)
else:
    torch.multiprocessing.set_start_method('spawn', force=True)


#%% 参数初始化
'''
---------参数初始化---------------
【windows下是调试参数，linux下是正式参数，两套参数可分别设置】
【注：为方便跑多组实验，linux下需设置job_id参数（设置方式为: python xxx.py --job_id 123456），windows下不需要设置】

以下参数均为全局参数，需在此处修改
slice_date:         修改数据切片区间，前两个为样本内的起止点，后两个为样本外的起止点【均需要是交易日】'
dir:                input路径，即原始因子所在路径。
dir_pop:            output路径，即保存因子库、因子值、因子表达式的路径。
pool_num:           并行任务数量，建议设为1，即不并行，使用单显卡单进程运行。若并行，通信成本过高，效率提升不大。
pop_num, hof_num:   分别为种群数量、精英数量。一般精英数量设为种群数量的1/6左右即可。
niter:              【大循环】的迭代次数，每次迭代重新开始一次遗传规划、重新创立全新的种群，以上一轮的残差收益率作为优化目标。
ir_lowestvalue:     【大循环】中因子入库所需的最低rankIR值，低于此值的因子不入库。
cor_uplimit:        【大循环】中新因子与老因子的最高相关系数，相关系数绝对值高于此值的因子不入库。
ngen:               【小循环】的迭代次数，即每次遗传规划进行几轮繁衍进化。
max_tree_depth:     【小循环】中个体算子树的最大深度，即因子表达式的最大复杂度。
cxpb:               【小循环】中交叉概率，即两个个体之间进行交叉的概率。
mutpb:              【小循环】中变异概率，即个体进行突变变异的概率。
'''

#slice_date = ['2010-01-04', '2021-12-31', '2022-01-04', '2023-12-29']   # 注意均需要是交易日
slice_date = ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29']   # 注意均需要是交易日
niter = 5
ir_lowestvalue = 2.5
cor_uplimit = 0.7
ngen = 2  #实际运行为n+1轮
max_tree_depth = 3
cxpb = 0.35
mutpb = 0.25

if plat == 'windows':
    dir = './data/features/parquet'
    dir_pop = './pop/bendi'
    pool_num, pop_num, hof_num = 1, 4, 3

elif plat == 'linux':

    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--job_id", type=int, required=True)
        parser.add_argument("--poolnm", type=int, required=False)
        args, _ = parser.parse_known_args()
        job_id_arg = args.job_id
        if args.poolnm is None:
            poolnm_arg = 1
        else:
            poolnm_arg = args.poolnm

    dir = './data/features/parquet'
    dir_pop = f'./pop/{job_id_arg}'   #linux系统下，job_id需在运行程序时自行定义 （设置方式为: python xxx.py --job_id 123456）
    pool_num, pop_num, hof_num = poolnm_arg, 3000, 500

# 查看output路径是否存在，不存在则创建
if not os.path.exists(dir_pop):
    os.mkdir(dir_pop)
if not os.path.exists(f'{dir_pop}/factor'):
    os.mkdir(f'{dir_pop}/factor')
print(dir_pop)


#%% 读取数据
'转换index为datetime格式，选取切片日期，选取股票列'
def dfindex_to_dt(df, input_freq='D', stockdata=True,firstdata=False):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[slice_date[0]:slice_date[-1]] # 训练集首日至测试集末日
    # if freq!='D':
    #     df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    if stockdata:
        df = df[stockcol] # 选取指定股票
    if firstdata:
        global slice_date_indexnum, timerow
        slice_date_indexnum = [df.index.get_loc(i) for i in slice_date] # slice_date在全部日期中的索引
        timerow = df.index.tolist() # 日期行
    df = torch.FloatTensor(df.values).to(device)
    df.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return df


timenow = time.time()
'''-----------读取数据中：带ori后缀的是原始的，不带ori后缀的是【滚动60天tszscore标准化】后的----------'''
# global close, turn, volume, amt, open1, high, low, vwap, bp, ep, ocfp, dp, return1, adv20, adv60, 
# close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori, bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori, cs_indus_code, close_adj, size
close = pd.read_parquet(f'{dir}/close_adj_zscore_day.parquet', engine='fastparquet')
global stockcol
stockcol = close.columns.tolist()

close = dfindex_to_dt(close,firstdata=True)
print('success: close')
turn = pd.read_parquet(f'{dir}/turn_zscore_day.parquet', engine='fastparquet')
turn = dfindex_to_dt(turn)
volume = pd.read_parquet(f'{dir}/volume_zscore_day.parquet', engine='fastparquet')
volume = dfindex_to_dt(volume)
amt = pd.read_parquet(f'{dir}/amt_zscore_day.parquet', engine='fastparquet')
amt = dfindex_to_dt(amt)
open1 = pd.read_parquet(f'{dir}/open_zscore_day.parquet', engine='fastparquet')
open1 = dfindex_to_dt(open1)
high = pd.read_parquet(f'{dir}/high_zscore_day.parquet', engine='fastparquet')
high = dfindex_to_dt(high)
low = pd.read_parquet(f'{dir}/low_zscore_day.parquet', engine='fastparquet')
low = dfindex_to_dt(low)
vwap = pd.read_parquet(f'{dir}/vwap_zscore_day.parquet', engine='fastparquet')
vwap = dfindex_to_dt(vwap)
print('success: turn, volume, amt, open1, high, low, vwap')

bp = pd.read_parquet(f'{dir}/bp_lf_zscore_day.parquet', engine='fastparquet')
bp = dfindex_to_dt(bp)
ep = pd.read_parquet(f'{dir}/ep_ttm_zscore_day.parquet', engine='fastparquet')
ep = dfindex_to_dt(ep)
ocfp = pd.read_parquet(f'{dir}/ocfp_ttm_zscore_day.parquet', engine='fastparquet')
ocfp = dfindex_to_dt(ocfp)
dp = pd.read_parquet(f'{dir}/dividendyield2_zscore_day.parquet', engine='fastparquet')
dp = dfindex_to_dt(dp)
return1 = pd.read_parquet(f'{dir}/return1_day.parquet', engine='fastparquet')
return1 = dfindex_to_dt(return1)
adv20 = pd.read_parquet(f'{dir}/adv20_zscore_day.parquet', engine='fastparquet')
adv20 = dfindex_to_dt(adv20)
adv60 = pd.read_parquet(f'{dir}/adv60_zscore_day.parquet', engine='fastparquet')
adv60 = dfindex_to_dt(adv60)
print('success: bp, ep, ocfp, dp, return1, adv20, adv60')

close_ori = pd.read_parquet(f'{dir}/close_adj_day.parquet', engine='fastparquet')
close_ori = dfindex_to_dt(close_ori)
turn_ori = pd.read_parquet(f'{dir}/turn_day.parquet', engine='fastparquet')
turn_ori = dfindex_to_dt(turn_ori)
volume_ori = pd.read_parquet(f'{dir}/volume_day.parquet', engine='fastparquet')
volume_ori = dfindex_to_dt(volume_ori)
amt_ori = pd.read_parquet(f'{dir}/amt_day.parquet', engine='fastparquet')
amt_ori = dfindex_to_dt(amt_ori)
open1_ori = pd.read_parquet(f'{dir}/open_day.parquet', engine='fastparquet')
open1_ori = dfindex_to_dt(open1_ori)
high_ori = pd.read_parquet(f'{dir}/high_day.parquet', engine='fastparquet')
high_ori = dfindex_to_dt(high_ori)
low_ori = pd.read_parquet(f'{dir}/low_day.parquet', engine='fastparquet')
low_ori = dfindex_to_dt(low_ori)
vwap_ori = pd.read_parquet(f'{dir}/vwap_day.parquet', engine='fastparquet')
vwap_ori = dfindex_to_dt(vwap_ori)

print('success: close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori')

bp_ori = pd.read_parquet(f'{dir}/bp_lf_day.parquet', engine='fastparquet')
bp_ori = dfindex_to_dt(bp_ori)
ep_ori = pd.read_parquet(f'{dir}/ep_ttm_day.parquet', engine='fastparquet')
ep_ori = dfindex_to_dt(ep_ori)
ocfp_ori = pd.read_parquet(f'{dir}/ocfp_ttm_day.parquet', engine='fastparquet')
ocfp_ori = dfindex_to_dt(ocfp_ori)
dp_ori = pd.read_parquet(f'{dir}/dividendyield2_day.parquet', engine='fastparquet')
dp_ori = dfindex_to_dt(dp_ori)
return1_ori = pd.read_parquet(f'{dir}/return1_day.parquet', engine='fastparquet')
return1_ori = dfindex_to_dt(return1_ori)
adv20_ori = pd.read_parquet(f'{dir}/adv20_day.parquet', engine='fastparquet')
adv20_ori = dfindex_to_dt(adv20_ori)
adv60_ori = pd.read_parquet(f'{dir}/adv60_day.parquet', engine='fastparquet')
adv60_ori = dfindex_to_dt(adv60_ori)

cs_indus_code = pd.read_parquet(f'{dir}/cs_indus_code_day.parquet', engine='fastparquet')
cs_indus_code = dfindex_to_dt(cs_indus_code)

size = pd.read_parquet(f'{dir}/size_day.parquet', engine='fastparquet')
size = dfindex_to_dt(size)

print('success: bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori, cs_indus_code, size')

print('load data done: %.4f seconds'%(time.time()-timenow))


#%% 指数收益率（未启用）
# index_name = '000852.SH'
# index_name = '000300.SH'
# close_index_300 = close_index[index_name]
# close_index_300_wk = close_index_300.groupby([pd.Grouper(level=0, freq=freq)]).last()
# close_index_300_wk

    
#%% 计算未来收益率并中性化
timenow = time.time()

# size = size.groupby([pd.Grouper(level=0, freq='D')]).last()
# return1_next = close_ori.groupby([pd.Grouper(level=0, freq='D')]).last()
size = size
return1_next_ori = close_ori
# return1_next = return1_next.pct_change()
# return1_next = return1_next.shift(-1)
# return1_next = delaypct_torch(return1_next, 1)   # t-1至t的收益率
# return1_next = delay_torch(return1_next, -1)  # t至t+1的收益率
return1_next_ori = ts_delaypct_torch(return1_next_ori, 10)  # t-10至t的收益率
return1_next_ori = ts_delay_torch(return1_next_ori, -11)  # t+1至t+11的收益率
return1_next = neutralize_numpy(return1_next_ori, size, cs_indus_code)  # 市值行业中性化
return1_next_resid = return1_next

print('neutralize y done: %.4f seconds'%(time.time()-timenow))


#%% 适应度函数：计算单因子适应度
'''定义遗传算法中的适应度函数，即信息比率的绝对值abs(rankIR)'''
def evaluate(individual, pool_skuname, 
             close, turn, volume, amt, open1, high, low, vwap, 
             bp, ep, ocfp, dp, adv20, adv60, 
             close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori,
             bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori, 
             size, return1_next, return1_next_resid, slice_date_indexnum):
    
    # individual: 如sigmoid_torch(rank_sub_torch(ts_grouping_decsortavg_torch(turn, dp_ori, 15, 4), high)) 
    # pool_skuname: 如z_iter0_gen0_0
    
    '记录开始时间并输出txt'
    # print(str(individual))
    if int(pool_skuname.split('_')[-1])%100 == 0:
        start_time_sku = time.time()
        output_path = f'{dir_pop}/z_{pool_skuname}.txt'
        with open(output_path, 'w', encoding='utf-8') as file1:
            print(str(individual),'\n start_time',time.ctime(start_time_sku),file=file1)

    '根据迭代出的因子表达式，计算因子值'
    func = toolbox.compile(expr=individual) # 基于函数表达式构建函数：deap.base.Toolbox().compile
    #timenow = time.time()
    func_value = func(close, turn, volume, amt, open1, high, low, vwap, bp, ep, ocfp, dp, adv20, adv60, close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori, bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori)
    #print('compute x done: %.4f seconds'%(time.time()-timenow))
    if isinstance(func_value, float):
        # func_value = pd.DataFrame(np.ones_like(close),index=close.index,columns=close.columns) * func_value
        func_value = torch.ones_like(close) * func_value

    if plat == 'windows': print('func_value_end')

    # 异常值处理：inf转换为nan
    func_value_gp = func_value
    # func_value_gp = func_value_gp.applymap(lambda x: np.nan if (x == np.inf) or (x == -np.inf) else x)
    func_value_gp = torch.where(func_value_gp == torch.inf, torch.nan, func_value_gp)
    func_value_gp = torch.where(func_value_gp == -torch.inf, torch.nan, func_value_gp)

    # '市值中性化（对x做，现已取消，修改为对Y做）'
    # func_value_gp = neutralize_torch(func_value_gp,size,cs_indus_code)
    
    const_annual = 24 # 年化常数
    
    '计算原始IC IR（x是原始x，y是只与风格因子做回归得到的残差收益率）'
    #timenow = time.time()
    # ic_t = func_value_gp.corrwith(return1_next,axis=1)
    ic_t = corrwith_torch_matrix(rank_pct_torch(func_value_gp), rank_pct_torch(return1_next),dim=1)
    #print('compute ic: %.4f seconds'%(time.time()-timenow))
    # 样本内IC、IC标准差、年化ICIR
    ic_t_in = ic_t[slice_date_indexnum[0]:slice_date_indexnum[1]+1]
    ic_std_in = (((ic_t_in - ic_t_in.nanmean()) ** 2).nanmean()) ** 0.5
    ir_in = ic_t_in.nanmean() / ic_std_in * np.sqrt(const_annual) # 年化
    # 样本外IC、IC标准差、年化ICIR
    ic_t_out = ic_t[slice_date_indexnum[2]:slice_date_indexnum[3]+1]
    ic_std_out = (((ic_t_out - ic_t_out.nanmean()) ** 2).nanmean()) ** 0.5
    ir_out = ic_t_out.nanmean() / ic_std_out * np.sqrt(const_annual)

    '计算中性化IC IR（x是原始x，y是与风格因子和上一代挖出的新因子一起做回归得到的残差收益率）'
    ic_t_resid = corrwith_torch_matrix(rank_pct_torch(func_value_gp), rank_pct_torch(return1_next_resid), dim=1)
    # 样本内IC、IC标准差、年化ICIR
    ic_t_in_resid = ic_t_resid[slice_date_indexnum[0]:slice_date_indexnum[1]]
    ic_std_in_resid = (((ic_t_in_resid - ic_t_in_resid.nanmean()) ** 2).nanmean()) ** 0.5
    ir_in_resid = ic_t_in_resid.nanmean() / ic_std_in_resid * np.sqrt(const_annual)
    # 样本外IC、IC标准差、年化ICIR
    ic_t_out_resid = ic_t_resid[slice_date_indexnum[2]:slice_date_indexnum[3]]
    ic_std_out_resid = (((ic_t_out_resid - ic_t_out_resid.nanmean()) ** 2).nanmean()) ** 0.5
    ir_out_resid = ic_t_out_resid.nanmean() / ic_std_out_resid * np.sqrt(const_annual)

    # '记录结束时间并输出txt'
    # end_time_sku = time.time()
    # print(str(individual),ir,'end_time',time.ctime(end_time_sku),'time_cost',time.strftime("%H:%M:%S", time.gmtime(end_time_sku-start_time_sku)))
    # output_path = f'{dir_pop}/z_{pool_skuname}_end_{time.strftime("%H_%M_%S", time.gmtime(end_time_sku-start_time_sku))}.txt'
    # with open(output_path, 'w', encoding='utf-8') as file1:
    #     print(str(individual), '\n',ir,'\n', 'start_time', time.ctime(start_time_sku),file=file1)
    #     print(' end_time', time.ctime(end_time_sku), '\n time_cost',
    #           time.strftime("%H:%M:%S", time.gmtime(end_time_sku - start_time_sku)),file=file1)

    '返回IR，绝对值越大越好，0为最差。如果返回nan值，则输出0'
    if ir_in_resid.isnan(): ir_in_resid=0
    if ir_out_resid.isnan(): ir_out_resid=0
    if ir_in.isnan(): ir_in=0
    if ir_out.isnan(): ir_out=0
    return abs(ir_in_resid),ir_in_resid,ir_out_resid,ir_in,ir_out,


#%% 单次循环
'遗传算法全流程，依次完成选种、交叉、突变操作'
def eaSimple_shr(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, start_gen=None,):  #evaluate, open1, close, high, low, volume, return1, vwap, adv20, adv60, trade_count,
    """
    代码的粗略结构如下：
        evaluate(population)     # 对随机生成的初代种群评估IR值
        for g in range(ngen):
            population = select(population, len(population))    # 选取abs(IR)值较高的个体，以产生后代
            offspring = varAnd(population, toolbox, cxpb, mutpb)   # 交叉、变异
            evaluate(offspring)   # 对新种群评估IR值
            population = offspring    # 更新种群
    """

    if start_gen is None:
        start_gen = 1
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        '评估个体abs(IR)值'
        print('muti'+str(pool_num))
        pool_list = [f'iter{overall_iter_num}_gen0_{i}' for i in range(len(invalid_ind))]
        if pool_num != 1:
            pool = Pool(pool_num)
            # fitnesses = list(tqdm(pool.map(toolbox.evaluate, invalid_ind, pool_list), total=len(invalid_ind), desc="gen"+str(0)))
            fitnesses = pool.starmap(toolbox.evaluate, zip(invalid_ind, pool_list), chunksize=1)
            # fitnesses = pool.map(myfunc,range(10))
            pool.close()
            pool.join()
            # pool.clear()
        else:
            # 调用evaluate
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, pool_list)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = tuple([fit[0]])
            ind.icir_in_resid = fit[1]
            ind.icir_out_resid = fit[2]
            ind.icir_in = fit[3]
            ind.icir_out = fit[4]

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        joblib.dump(population, f'{dir_pop}/pop_iter{overall_iter_num}_' + str(0) + '.pkl')
        joblib.dump(halloffame, f'{dir_pop}/hof_iter{overall_iter_num}_' + str(0) + '.pkl')
        joblib.dump(logbook, f'{dir_pop}/log_iter{overall_iter_num}_' + str(0) + '.pkl')
    
    else:
        population = joblib.load(f'{dir_pop}/pop_iter{overall_iter_num}_' + str(start_gen-1) + '.pkl')
        halloffame = joblib.load(f'{dir_pop}/hof_iter{overall_iter_num}_' + str(start_gen-1) + '.pkl')
        logbook = joblib.load(f'{dir_pop}/log_iter{overall_iter_num}_' + str(start_gen-1) + '.pkl')

    # Begin the generational process
    for gen in range(start_gen, ngen + 1):
        # Select the next generation individuals，abs(IR)越大，则被选中的概率越大
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        timenow = time.time()
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        print('varAnd done: %.4f seconds'%(time.time()-timenow))

        # Evaluate the individuals with an invalid fitness（对于发生改变的个体，重新评估abs(IR)值）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pool_list = [f'iter{overall_iter_num}_gen{gen}_{i}' for i in range(len(invalid_ind))]
        if pool_num!=1:
            pool = Pool(pool_num)
            # fitnesses = list(tqdm(pool.imap(toolbox.evaluate, invalid_ind, pool_list), total=len(invalid_ind), desc="gen"+str(gen)))
            fitnesses = pool.starmap(toolbox.evaluate, zip(invalid_ind, pool_list), chunksize=1)
            pool.close()
            pool.join()
            # pool.clear()
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, pool_list)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = tuple([fit[0]])
            ind.icir_in_resid = fit[1]
            ind.icir_out_resid = fit[2]
            ind.icir_in = fit[3]
            ind.icir_out = fit[4]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        joblib.dump(population, f'{dir_pop}/pop_iter{overall_iter_num}_{gen}.pkl')
        joblib.dump(halloffame, f'{dir_pop}/hof_iter{overall_iter_num}_{gen}.pkl')
        joblib.dump(logbook, f'{dir_pop}/log_iter{overall_iter_num}_{gen}.pkl')
    
    return population, logbook


#%% 大循环
saved_times = pd.DataFrame(columns=['settings','gp','selection','neu_dump','all','avg_compute_factors'], dtype=float)

'''-------大循环，每次循环重新开始一次遗传规划、重新创立全新的种群，以上一轮的残差收益率作为优化目标-------'''
for overall_iter_num in range(niter):
    
    
    #%% 设置
    timenow0 = time.time()
    
    #对数据分类可自由定义、自由命名。如下方代码定义了几种正整数，和两种初始因子（ori是没做标准化的，不带ori是做了标准化的）
    class POSINT0(): pass
    class POSINT1(): pass
    class POSINT2(): pass
    class POSINT3(): pass
    class POSINT4(): pass
    class POSINT5(): pass
    class DF_ori(): pass
    class DF(): pass

    #对于每个数据分类，均需定义returnself算子，以便遗传算法识别
    def _returnself(input):
        return input

    '''
    --------定义遗传算法中的基础算子（由math_func_gpu.py导入，每个函数均可作为基础算子），并注明每个算子的输入输出类型-------------
    定义方法为：pset.addPrimitive(func_name, [第1个输入数据的数据类型, 第2个输入数据的数据类型……], 输出数据的数据类型)
    '''
    pset = gp.PrimitiveSetTyped("main", [DF] * 14 + [DF_ori] * 15, DF)
    if True:
        pset.addPrimitive(ts_correlation_torch, [DF_ori, DF_ori, POSINT2], DF)
        pset.addPrimitive(ts_covariance_torch, [DF, DF, POSINT2], DF)
        pset.addPrimitive(rank_pct_torch, [DF], DF)
        pset.addPrimitive(sign_torch, [DF_ori], DF)
        pset.addPrimitive(ts_delay_torch, [DF, POSINT1], DF)
        pset.addPrimitive(scale_torch, [DF], DF)
        pset.addPrimitive(ts_delta_torch, [DF, POSINT1], DF)
        # pset.addPrimitive(signedpower, [pd.DataFrame,int], pd.DataFrame)
        pset.addPrimitive(ts_decay_linear_igrnan_torch, [DF, POSINT2], DF)
        pset.addPrimitive(ts_min_torch, [DF, POSINT2], DF)
        pset.addPrimitive(ts_max_torch, [DF, POSINT2], DF)
        pset.addPrimitive(ts_argmin_torch, [DF_ori, POSINT2], DF)
        pset.addPrimitive(ts_argmax_torch, [DF_ori, POSINT2], DF)
        pset.addPrimitive(ts_rank_torch, [DF_ori, POSINT2], DF)
        # pset.addPrimitive(ts_zscore, [pd.DataFrame,POSINT2], pd.DataFrame)
        pset.addPrimitive(sigmoid_torch, [DF], DF)
        pset.addPrimitive(ts_stddev_torch, [DF, POSINT2], pd.DataFrame)
        pset.addPrimitive(ts_sum_torch, [DF, POSINT2], DF)
        # pset.addPrimitive(ts_product_torch, [DF, POSINT2], DF)
        pset.addPrimitive(rank_sub_torch, [DF, DF], DF)
        pset.addPrimitive(rank_div_torch, [DF, DF], DF)
        pset.addPrimitive(rank_add_torch, [DF, DF], DF)
        # pset.addPrimitive(add_int, [pd.DataFrame, int], pd.DataFrame)
        # pset.addPrimitive(sub_int1, [pd.DataFrame, int], pd.DataFrame)
        # pset.addPrimitive(sub_int2, [int, pd.DataFrame], pd.DataFrame)
        # pset.addPrimitive(mul_int, [pd.DataFrame, int], pd.DataFrame)
        # pset.addPrimitive(div_int1, [pd.DataFrame, int], pd.DataFrame)
        # pset.addPrimitive(div_int2, [int, pd.DataFrame], pd.DataFrame)
        # pset.addPrimitive(neg_int, [int], int)
    pset.addPrimitive(add, [DF, DF], DF)
    pset.addPrimitive(sub, [DF, DF], DF)
    pset.addPrimitive(mul, [DF, DF], DF)
    pset.addPrimitive(div, [DF, DF], DF)
    pset.addPrimitive(log_torch, [DF], DF)
    pset.addPrimitive(sqrt_torch, [DF], DF)
    pset.addPrimitive(neg, [DF], DF)

    pset.addPrimitive(ts_grouping_ascsortavg_torch, [DF, DF_ori, POSINT3, POSINT1], DF)
    pset.addPrimitive(ts_grouping_decsortavg_torch, [DF, DF_ori, POSINT3, POSINT1], DF)
    pset.addPrimitive(ts_grouping_diffsortavg_torch, [DF, DF_ori, POSINT3, POSINT1], DF)
    # pset.addPrimitive(rolling_selmean_btm, [DF, DF_ori, POSINT5, POSINT4], DF,name='rolling_selmean_btm_long')
    # pset.addPrimitive(rolling_selmean_top, [DF, DF_ori, POSINT5, POSINT4], DF,name='rolling_selmean_top_long')
    # pset.addPrimitive(rolling_selmean_diff, [DF, DF_ori, POSINT5, POSINT4], DF,name='rolling_selmean_diff_long')
    pset.addPrimitive(ts_rankcorr_torch, [DF, DF, POSINT2], DF)
    pset.addPrimitive(ts_delaypct_torch, [DF, POSINT1], DF)
    pset.addPrimitive(ts_SubPosDecayLinear_torch, [DF, DF, POSINT3], DF)

    '''
    --------添加遗传算法中的随机正整数----------
    '''
    # pset.addEphemeralConstant('POSINT0',lambda: np.random.randint(0,10+1),POSINT0)
    # pset.addPrimitive(_returnself, [POSINT0], POSINT0,name='int_')

    pset.addEphemeralConstant(f'POSINT1_{overall_iter_num}', lambda: np.random.randint(1,10+1), POSINT1)
    pset.addPrimitive(_returnself, [POSINT1], POSINT1, name='_int')

    pset.addEphemeralConstant(f'POSINT2_{overall_iter_num}', lambda: np.random.randint(2,10+1), POSINT2)
    pset.addPrimitive(_returnself, [POSINT2], POSINT2, name='_int_')

    pset.addTerminal(10, POSINT3)
    pset.addTerminal(15, POSINT3)
    pset.addTerminal(20, POSINT3)
    pset.addTerminal(40, POSINT3)
    pset.addPrimitive(_returnself, [POSINT3], POSINT3, name='_int__')

    # pset.addTerminal(1, POSINT4)
    # pset.addTerminal(5, POSINT4)
    # pset.addTerminal(10, POSINT4)
    # pset.addTerminal(20, POSINT4)
    # pset.addTerminal(40, POSINT4)
    # pset.addTerminal(60, POSINT4)
    # pset.addPrimitive(_returnself, [POSINT4], POSINT4, name='_int___')
    #
    # pset.addTerminal(60, POSINT5)
    # pset.addTerminal(120, POSINT5)
    # pset.addTerminal(180, POSINT5)
    # pset.addTerminal(200, POSINT5)
    # pset.addTerminal(240, POSINT5)
    # pset.addPrimitive(_returnself, [POSINT5], POSINT5, name='_int____')

    pset.addPrimitive(_returnself, [DF], DF, name='_df')
    pset.addPrimitive(_returnself, [DF_ori], DF_ori, name='_df_')

    '''
    ------------定义遗传算法中的初始因子（初始输入变量）--------------
    依次为close, turn, volume, amt, open1, high, low, vwap, 
    bp, ep, ocfp, dp, adv20, adv60,
    close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori, 
    bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori
    '''
    pset.renameArguments(ARG0="close")
    pset.renameArguments(ARG1="turn")
    pset.renameArguments(ARG2="volume")
    pset.renameArguments(ARG3="amt")
    pset.renameArguments(ARG4="open1")
    pset.renameArguments(ARG5="high")
    pset.renameArguments(ARG6="low")
    pset.renameArguments(ARG7="vwap")
    pset.renameArguments(ARG8="bp")
    pset.renameArguments(ARG9="ep")
    pset.renameArguments(ARG10="ocfp")
    pset.renameArguments(ARG11="dp")
    pset.renameArguments(ARG12="adv20")
    pset.renameArguments(ARG13="adv60")
    pset.renameArguments(ARG14="close_ori")
    pset.renameArguments(ARG15="turn_ori")
    pset.renameArguments(ARG16="volume_ori")
    pset.renameArguments(ARG17="amt_ori")
    pset.renameArguments(ARG18="open1_ori")
    pset.renameArguments(ARG19="high_ori")
    pset.renameArguments(ARG20="low_ori")
    pset.renameArguments(ARG21="vwap_ori")
    pset.renameArguments(ARG22="bp_ori")
    pset.renameArguments(ARG23="ep_ori")
    pset.renameArguments(ARG24="ocfp_ori")
    pset.renameArguments(ARG25="dp_ori")
    pset.renameArguments(ARG26="return1_ori")
    pset.renameArguments(ARG27="adv20_ori")
    pset.renameArguments(ARG28="adv60_ori")

    '''创建遗传算法基础模块，以下参数不建议更改，如需更改，可参考deap官方文档'''
    # https://zhuanlan.zhihu.com/p/72130823
    creator.create("FitnessMin", base.Fitness, weights=(+1.0,))   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset) # 个体编码：pset，预设的
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, close=close, turn=turn, volume=volume, amt=amt, open1=open1, high=high, low=low, vwap=vwap,
                     bp=bp, ep=ep, ocfp=ocfp, dp=dp, adv20=adv20, adv60=adv60, 
                     close_ori=close_ori, turn_ori=turn_ori, volume_ori=volume_ori, amt_ori=amt_ori, open1_ori=open1_ori, high_ori=high_ori, low_ori=low_ori, vwap_ori=vwap_ori,
                     bp_ori=bp_ori, ep_ori=ep_ori, ocfp_ori=ocfp_ori, dp_ori=dp_ori, return1_ori=return1_ori, adv20_ori=adv20_ori, adv60_ori=adv60_ori, 
                     size=size, return1_next=return1_next, return1_next_resid=return1_next_resid, slice_date_indexnum=slice_date_indexnum)
    toolbox.register("select", tools.selTournament, tournsize=3) # 锦标赛：第一轮随机选择3个，取最大
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_tree_depth)  # genFull
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3

    '创建遗传算法种群，定义初始人口数及选取的最优个体数'
    pop = toolbox.population(n=pop_num)
    hof = tools.HallOfFame(hof_num)
    
    saved_times.loc[overall_iter_num,'settings'] = time.time()-timenow0


    #%% 遗传规划核心环节：因子计算+适应度计算
    timenow1 = time.time()
    
    '''运行迭代主程序'''
    pop, log = eaSimple_shr(pop, toolbox, cxpb=cxpb,mutpb=mutpb, ngen=ngen,   #0.5,0.1,40
                            halloffame=hof, verbose=True)  #, start_gen=6   #algorithms.eaSimple
    
    saved_times.loc[overall_iter_num,'gp'] = time.time()-timenow1
    
    
    #%% 因子入库
    timenow1 = time.time()
    
    '''good_log文件为因子库，存储了因子表达式、rankIR值等信息。在具体因子值存储在factor文件夹的parquet文件中'''
    if overall_iter_num == 0:
        good_log = pd.DataFrame(columns=['overall_iter_num','str_hof','rankir_in_resid','rankir_out_resid','rankir_in_noresid','rankir_out_noresid'])
    else:
        good_log = pd.read_csv(f'{dir_pop}/good_log.csv',index_col=0)

    '''因子库存储在hof_good_list中，每个因子为一个二维tensor（时间*股票）'''
    hof_good_list = []

    '''对rankIR大于设定值的因子，进行相关性检验，如果相关性低，则加入因子库'''
    t_compute_factors = 0
    count_factors = 0
    for hof_single in hof:
        if hof_single.fitness.values[0] > ir_lowestvalue:
            print(str(hof_single))
            #------------------------------------------#
            # 计算因子
            timenow2 = time.time()
            hof_func = toolbox.compile(expr=hof_single)
            hof_value = hof_func(close, turn, volume, amt, open1, high, low, vwap, bp, ep, ocfp, dp, adv20, adv60, close_ori, turn_ori, volume_ori, amt_ori, open1_ori, high_ori, low_ori, vwap_ori, bp_ori, ep_ori, ocfp_ori, dp_ori, return1_ori, adv20_ori, adv60_ori)
            t_compute_factors = t_compute_factors + time.time() - timenow2
            count_factors = count_factors + 1
            #------------------------------------------#
            hof_value = torch.where(hof_value == torch.inf, torch.nan, hof_value)
            hof_value = torch.where(hof_value == -torch.inf, torch.nan, hof_value)
            good_tag = 1
            # 与已有的因子库做相关性检验，如果相关性大于预设值，则不加入因子库
            for hof_good in hof_good_list:
                corr_value = corrwith_torch_matrix(hof_value, hof_good, dim=1).nanmean()
                # print(corr_value)
                if abs(corr_value) > cor_uplimit:
                    good_tag = 0
                    break
                if np.isinf(hof_single.fitness.values[0].cpu()):
                    good_tag = 0
                    break
                if hof_single.icir_out_resid.cpu()==0:
                    good_tag = 0
                    break
            # 如果通过相关性检验，则加入因子库
            if good_tag:
                hof_good_list.append(hof_value)
                hof_value_df = pd.DataFrame(hof_value.cpu(), index=timerow, columns=stockcol)
                str_hof = str(hof_single)
                good_log = pd.concat([good_log,pd.DataFrame([[overall_iter_num,str_hof,float(hof_single.icir_in_resid),float(hof_single.icir_out_resid),float(hof_single.icir_in),float(hof_single.icir_out)]],columns=['overall_iter_num','str_hof','rankir_in_resid','rankir_out_resid','rankir_in_noresid','rankir_out_noresid'])],axis=0)
                hof_value_df.to_parquet(f'{dir_pop}/factor/{str_hof}.parquet', engine='fastparquet')

    good_log.to_csv(f'{dir_pop}/good_log.csv')
    
    saved_times.loc[overall_iter_num,'selection'] = time.time()-timenow1
    
    
    #%% 收益中性化和保存
    timenow1 = time.time()
    
    return1_next_resid = neutralize_numpy(return1_next_resid, size, cs_indus_code, hof_good_list)

    '''保存该轮迭代的最终种群、最优个体、迭代日志'''
    joblib.dump(pop,f'{dir_pop}/pop_iter{overall_iter_num}_overall.pkl')
    joblib.dump(hof,f'{dir_pop}/hof_iter{overall_iter_num}_overall.pkl')
    joblib.dump(log,f'{dir_pop}/log_iter{overall_iter_num}_overall.pkl')

    saved_times.loc[overall_iter_num,'neu_dump'] = time.time()-timenow1
    
    saved_times.loc[overall_iter_num,'all'] = time.time()-timenow0
    saved_times.loc[overall_iter_num,'avg_compute_factors'] = t_compute_factors / count_factors
    saved_times.to_csv(f'{dir_pop}/saved_times.csv')

