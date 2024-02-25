# %%
import pandas as pd
import numpy as np
import os , sys , copy
import torch
import argparse
import array , random , json , operator , time, platform , joblib

from copy import deepcopy
from deap import base , creator , tools , gp , algorithms
from deap.algorithms import varAnd
from torch.multiprocessing import Pool

from tqdm import tqdm
# from math_func_gpu import *

import math_func_gpu as F

_plat      = platform.system().lower()
_test_code = True or _plat == 'windows'
_DIR_data = './data/features/parquet' #input路径，即原始因子所在路径。
_DIR_pop  = './pop'                   #output路径，即保存因子库、因子值、因子表达式的路径。
_DIR_job  = f'{_DIR_pop}/bendi'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
torch.multiprocessing.set_start_method('spawn' if _plat == 'windows' else 'forkserver', force=True)
# %%
# environment setting
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=-1)
parser.add_argument("--poolnm", type=int, default=1)
args, _ = parser.parse_known_args()

def gp_parameters(test_code = None , job_id = None):
    global _DIR_job
    if _plat == 'linux':
        # 单机多进程设置（实际未启用），参考https://zhuanlan.zhihu.com/p/600801803
        if job_id is not None: args.job_id = job_id
        if args.job_id < 0:
            old_jobs = np.array(os.listdir('./pop/') if os.path.exists('./pop/') else []).astype(int)
            args.job_id = np.setdiff1d(np.arange(max(old_jobs) + 2) , old_jobs).min() if len(old_jobs) else 0
        _DIR_job = f'{_DIR_pop}/{args.job_id}'   #linux系统下，job_id需在运行程序时自行定义 （设置方式为: python xxx.py --job_id 123456）

    if test_code is None: test_code = _test_code
    # 查看output路径是否存在，不存在则创建
    if os.path.exists(_DIR_job):
        if input(f'{_DIR_job} exists , press "yes" to confirm continuation:')[0].lower() == 'y':
            pass
        else:
            raise Exception(f'{_DIR_job} exists!')
    else:
        os.makedirs(f'{_DIR_job}/factor' , exist_ok=True)
    '''
    ---------参数初始化---------------
    【windows下是调试参数，linux下是正式参数，两套参数可分别设置】
    【注：为方便跑多组实验，linux下需设置job_id参数（设置方式为: python xxx.py --job_id 123456），windows下不需要设置】

    以下参数均为全局参数，需在此处修改
    slice_date:         修改数据切片区间，前两个为样本内的起止点，后两个为样本外的起止点【均需要是交易日】'
    pool_num:           并行任务数量，建议设为1，即不并行，使用单显卡单进程运行。若并行，通信成本过高，效率提升不大。
    pop_num:            种群数量
    hof_num:            精英数量。一般精英数量设为种群数量的1/6左右即可。
    n_iter:             【大循环】的迭代次数，每次迭代重新开始一次遗传规划、重新创立全新的种群，以上一轮的残差收益率作为优化目标。
    ir_floor:           【大循环】中因子入库所需的最低rankIR值，低于此值的因子不入库。
    corr_cap:           【大循环】中新因子与老因子的最高相关系数，相关系数绝对值高于此值的因子不入库。
    n_gen:              【小循环】的迭代次数，即每次遗传规划进行几轮繁衍进化。
    max_depth:          【小循环】中个体算子树的最大深度，即因子表达式的最大复杂度。
    cxpb:               【小循环】中交叉概率，即两个个体之间进行交叉的概率。
    mutpb:              【小循环】中变异概率，即个体进行突变变异的概率。
    '''
    print('Device name:', torch.cuda.get_device_name(0), ', Available:' ,torch.cuda.is_available())
    print(f'Pop directory is : "{_DIR_job}"')

    gp_params = {
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu') ,
        'verbose' : False ,
        'pool_num' : 1 , # args.poolnm #并行任务数量，建议设为1，即不并行，使用单显卡单进程运行。若并行，通信成本过高，效率提升不大。
        'pop_num': 3000 , #种群数量
        'hof_num': 500 ,  #精英数量。一般精英数量设为种群数量的1/6左右即可。
        'slice_date' : ['2010-01-04', '2021-12-31', '2022-01-04', '2023-12-29'] ,  # 注意均需要是交易日
        'n_iter' :  5 ,     #【大循环】的迭代次数，每次迭代重新开始一次遗传规划、重新创立全新的种群，以上一轮的残差收益率作为优化目标。
        'ir_floor' : 2.5 ,  #【大循环】中因子入库所需的最低rankIR值，低于此值的因子不入库
        'corr_cap' : 0.7 ,  #【大循环】中新因子与老因子的最高相关系数，相关系数绝对值高于此值的因子不入库
        'n_gen' : 2  ,      #【小循环】的迭代次数，即每次遗传规划进行几轮繁衍进化,实际运行为n+1轮
        'max_depth' : 3 ,   #【小循环】中个体算子树的最大深度，即因子表达式的最大复杂度。
        'cxpb' : 0.35 ,     #【小循环】中交叉概率，即两个个体之间进行交叉的概率。
        'mutpb' : 0.25 ,    #【小循环】中变异概率，即个体进行突变变异的概率。
    }
    if test_code:
        gp_params.update({'verbose' : True , 'pop_num': 4 , 'hof_num': 3 , 'n_iter' : 2 ,
                        'slice_date' : ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29']}) 
    return gp_params

# %%
class gpTimer:
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {}
    class ptimer:
        def __init__(self , key , record = False , target_dict = {} , print = False):
            self.key = key
            self.record = record
            self.target_dict = target_dict
            self.print = print
        def __enter__(self):
            if self.print: print(f'{self.key} ... start!')
            self.start_time = time.time()
        def __exit__(self, type, value, trace):
            time_cost = time.time() - self.start_time
            if self.record: self.append_time(self.target_dict , self.key , time_cost)
            if self.print: print(f'{self.key} ... done, cost {time_cost:.2f} secs')
        @staticmethod
        def append_time(target_dict , key , time_cost):
            if key not in target_dict.keys():
                target_dict[key] = [time_cost]
            else:
                target_dict[key].append(time_cost)
    def __call__(self , key , print = False):
        return self.ptimer(key , self.recording , self.recorder , print = print)
    def __repr__(self):
        return self.recorder.__repr__()
    def __bool__(self): return True
    def append_time(self , key , time_cost):
        self.ptimer.append_time(self.recorder , key , time_cost)
    def save_to_csv(self , path , columns = [] , dtype = float):
        if columns:
            df = pd.DataFrame(data = {col:self.recorder[col] for col in columns} , dtype=dtype) 
            df.to_csv(path)

class gpAccTimer:
    def __init__(self , key = ''):
        self.key   = key
        self.time  = 0.
        self.count = 0
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, type, value, trace):
        self.time  += time.time() - self.start_time
        self.count += 1
    def __repr__(self) -> str:
        return f'time : {self.time} , count {self.count}'
    def avgtime(self):
        return self.time if self.count == 0 else self.time / self.count


# %%
def gp_dictionary(gp_params , gp_timer):
    with gp_timer('load data' , print = True):
        gp_dict = copy.deepcopy(gp_params)
        slice_date = gp_dict['slice_date']

        df_list = ['close', 'turn', 'volume', 'amt', 'open', 'high', 'low', 'vwap', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']
        df_ori_list = [f'{_d}_ori' for _d in df_list] + ['return1_ori']
        gp_args = df_list + df_ori_list # gp args sequence

        gp_values = []
        nrowchar = 0
        for i , gp_key in enumerate(gp_args):
            df , first_data_return = read_gp_data(gp_key,slice_date,gp_dict.get('stockcol'),first_data=(i==0),device=gp_dict.get('device'))
            if first_data_return: gp_dict.update(first_data_return)
            gp_values.append(df)
            if gp_dict.get('verbose'):
                nrowchar += len(gp_key) + 1
                print(gp_key , end='\n' if nrowchar >= 100 else ',')
                if i == len(gp_args) - 1: print()
                if nrowchar >= 100: nrowchar = 0

        gp_dict['gp_args']   = gp_args
        gp_dict['gp_values'] = gp_values
        for key in ['size' , 'cs_indus_code']: gp_dict[key] = read_gp_data(key,slice_date,gp_dict.get('stockcol'),device=gp_dict.get('device'))[0]

        cp_ori = gp_values[gp_args.index('close_ori')]
        labels = F.ts_delaypct_torch(cp_ori, 10)  # t-10至t的收益率
        labels = F.ts_delay_torch(labels, -11)  # t+1至t+11的收益率
        labels_resid = F.neutralize_numpy(labels, gp_dict['size'], gp_dict['cs_indus_code'])  # 市值行业中性化

        gp_dict['labels'] = gp_dict['labels_resid'] = labels_resid

    gp_dict['gp_timer'] = gp_timer
    return gp_dict

def gp_data_filename(gp_key : str):
    ori = gp_key.endswith('_ori')
    k   = gp_key[:-4] if ori else gp_key
    if k == 'open1':
        v = f'open'
    elif k == 'bp':
        v = f'{k}_lf'
    elif k in ['ep' , 'ocfp']:
        v = f'{k}_ttm'
    elif k == 'dp':
        v = 'dividendyield2'
    elif k in ['close']:
        v = f'{k}_adj'
    else:
        v = k
    return f'{v}_day' if ori or v in ['cs_indus_code' , 'size'] else f'{v}_zscore_day'

def read_gp_data(gp_key,slice_date,stockcol=None,first_data=False,device=None,input_freq='D'):
    file = f'{_DIR_data}/{gp_data_filename(gp_key)}.parquet'
    df = pd.read_parquet(file, engine='fastparquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: df = df[slice_date[0]:slice_date[-1]] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    df = df[stockcol] if stockcol is not None else df # 选取指定股票

    if first_data:
        first_data_return = {
            'stockcol' : df.columns.tolist() ,
            'slice_indexnum':[df.index.get_loc(i) for i in slice_date] , # slice_date在全部日期中的索引 
            'timerow': df.index.tolist() , # 日期行
        }
    else:
        first_data_return = {}

    df = torch.FloatTensor(df.values)
    if device: df = df.to(device)
    df.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return (df , first_data_return)


# %%
class gpTypes:
    # types:
    class POSINT0(): pass
    class POSINT1(): pass
    class POSINT2(): pass
    class POSINT3(): pass
    class POSINT4(): pass
    class POSINT5(): pass
    class DF_ori(): pass
    class DF(): pass

    @staticmethod
    def _returnself(input): return input

    # methods (or Primatives):
    @classmethod
    def primatives_self(cls):
        return [
            (cls._returnself, [cls.DF], cls.DF , '_df') ,
            (cls._returnself, [cls.DF_ori], cls.DF_ori , '_dfo') ,
            #(cls._returnself, [cls.POSINT0], cls.POSINT0, '_int0') ,
            (cls._returnself, [cls.POSINT1], cls.POSINT1, '_int1') ,
            (cls._returnself, [cls.POSINT2], cls.POSINT2, '_int2') ,
            (cls._returnself, [cls.POSINT3], cls.POSINT3, '_int3') ,
            #(cls._returnself, [cls.POSINT4], cls.POSINT4, '_int4') ,
            #(cls._returnself, [cls.POSINT5], cls.POSINT5, '_int5') ,
        ]
    @classmethod
    def primatives_1d(cls):
        # (primative , in_types , out_type , name)
        return [
            (F.log_torch, [cls.DF], cls.DF) ,
            (F.sqrt_torch, [cls.DF], cls.DF) ,
            (F.neg, [cls.DF], cls.DF) ,
            (F.rank_pct_torch, [cls.DF], cls.DF) ,
            (F.sign_torch, [cls.DF_ori], cls.DF) ,
            (F.scale_torch, [cls.DF], cls.DF) ,
            (F.sigmoid_torch, [cls.DF], cls.DF) ,
            # (F.signedpower, [pd.DataFrame,int], pd.DataFrame) ,
            # (F.neg_int, [int], int) ,
        ]
    @classmethod
    def primatives_2d(cls):
        # (primative , in_types , out_type , name)
        return [
            (F.add, [cls.DF, cls.DF], cls.DF) ,
            (F.sub, [cls.DF, cls.DF], cls.DF) ,
            (F.mul, [cls.DF, cls.DF], cls.DF) ,
            (F.div, [cls.DF, cls.DF], cls.DF) ,
            (F.ts_delaypct_torch, [cls.DF, cls.POSINT1], cls.DF) ,
            (F.ts_stddev_torch, [cls.DF, cls.POSINT2], pd.DataFrame) ,
            (F.ts_sum_torch, [cls.DF, cls.POSINT2], cls.DF) ,
            # (F.ts_product_torch, [cls.DF, cls.POSINT2], cls.DF) ,
            (F.ts_delay_torch, [cls.DF, cls.POSINT1], cls.DF) ,
            (F.ts_delta_torch, [cls.DF, cls.POSINT1], cls.DF) ,
            (F.ts_decay_linear_igrnan_torch, [cls.DF, cls.POSINT2], cls.DF) ,
            (F.ts_min_torch, [cls.DF, cls.POSINT2], cls.DF) ,
            (F.ts_max_torch, [cls.DF, cls.POSINT2], cls.DF) ,
            (F.ts_argmin_torch, [cls.DF_ori, cls.POSINT2], cls.DF) ,
            (F.ts_argmax_torch, [cls.DF_ori, cls.POSINT2], cls.DF) ,
            (F.ts_rank_torch, [cls.DF_ori, cls.POSINT2], cls.DF) ,
            # (F.ts_zscore, [pd.DataFrame,cls.POSINT2], pd.DataFrame) ,
            (F.rank_sub_torch, [cls.DF, cls.DF], cls.DF) ,
            (F.rank_div_torch, [cls.DF, cls.DF], cls.DF) ,
            (F.rank_add_torch, [cls.DF, cls.DF], cls.DF) ,
            # (F.add_int, [pd.DataFrame, int], pd.DataFrame) ,
            # (F.sub_int1, [pd.DataFrame, int], pd.DataFrame) ,
            # (F.sub_int2, [int, pd.DataFrame], pd.DataFrame) ,
            # (F.mul_int, [pd.DataFrame, int], pd.DataFrame) ,
            # (F.div_int1, [pd.DataFrame, int], pd.DataFrame) ,
            # (F.div_int2, [int, pd.DataFrame], pd.DataFrame) ,
        ]
    @classmethod
    def primatives_3d(cls):
        # (primative , in_types , out_type , name)
        return [
            (F.ts_rankcorr_torch, [cls.DF, cls.DF, cls.POSINT2], cls.DF) ,
            (F.ts_SubPosDecayLinear_torch, [cls.DF, cls.DF, cls.POSINT3], cls.DF) ,
            (F.ts_correlation_torch, [cls.DF_ori, cls.DF_ori, cls.POSINT2], cls.DF) ,
            (F.ts_covariance_torch, [cls.DF, cls.DF, cls.POSINT2], cls.DF) ,
        ]
    @classmethod
    def primatives_4d(cls):
        # (primative , in_types , out_type , name)
        return [
            (F.ts_grouping_ascsortavg_torch, [cls.DF, cls.DF_ori, cls.POSINT3, cls.POSINT1], cls.DF) ,
            (F.ts_grouping_decsortavg_torch, [cls.DF, cls.DF_ori, cls.POSINT3, cls.POSINT1], cls.DF) ,
            (F.ts_grouping_decsortavg_torch, [cls.DF, cls.DF_ori, cls.POSINT3, cls.POSINT1], cls.DF) ,
            #(F.rolling_selmean_btm, [cls.DF, cls.DF_ori, cls.POSINT5, cls.POSINT4], cls.DF,name='rolling_selmean_btm_long') ,
            #(F.rolling_selmean_top, [cls.DF, cls.DF_ori, cls.POSINT5, cls.POSINT4], cls.DF,name='rolling_selmean_top_long') ,
            #(F.rolling_selmean_diff, [cls.DF, cls.DF_ori, cls.POSINT5, cls.POSINT4], cls.DF,name='rolling_selmean_diff_long') ,
        ]
    @classmethod
    def primatives_all(cls):
        return [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['self','1d','2d','3d','4d']] for prima in plist]



# %%
def evaluate(individual, pool_skuname, labels , labels_resid, slice_indexnum , size ,
             gp_values , compile_func , **kwargs):
    '''定义遗传算法中的适应度函数，即信息比率的绝对值abs(rankIR)'''
    # individual: 如sigmoid_torch(rank_sub_torch(ts_grouping_decsortavg_torch(turn, dp_ori, 15, 4), high)) 
    # pool_skuname: 如z_iter0_gen0_0
    '记录开始时间并输出txt'
    # print(str(individual))
    if int(pool_skuname.split('_')[-1])%100 == 0:
        start_time_sku = time.time()
        output_path = f'{_DIR_job}/z_{pool_skuname}.txt'
        with open(output_path, 'w', encoding='utf-8') as file1:
            print(str(individual),'\n start_time',time.ctime(start_time_sku),file=file1)

    '根据迭代出的因子表达式，计算因子值'
    func = compile_func(expr=individual) # type: ignore 基于函数表达式构建函数：deap.base.Toolbox().compile
    #timenow = time.time()
    func_value = func(*gp_values)
    #print('compute x done: %.4f seconds'%(time.time()-timenow))
    if isinstance(func_value, float):
        # func_value = pd.DataFrame(np.ones_like(close),index=close.index,columns=close.columns) * func_value
        func_value = torch.ones_like(gp_values[0]) * func_value

    if _plat == 'windows': print('func_value_end')

    # 异常值处理：inf转换为nan
    func_value_gp = func_value
    func_value_gp[func_value_gp.isinf()] = torch.nan

    # '市值中性化（对x做，现已取消，修改为对Y做）'
    # func_value_gp = neutralize_torch(func_value_gp,size,cs_indus_code)
    
    const_annual = 24 # 年化常数
    '计算原始IC IR（x是原始x，y是只与风格因子做回归得到的残差收益率）'

    ir_list = []
    for resid in [True , False]:
        ic = F.corrwith_torch_matrix(F.rank_pct_torch(func_value_gp), F.rank_pct_torch(labels_resid if resid else labels),dim=1)
        for in_sample in [True , False]:
            ic_samp = ic[slice_indexnum[0]:slice_indexnum[1]+1] if in_sample else ic[slice_indexnum[2]:slice_indexnum[3]+1]
            ic_std = (((ic_samp - ic_samp.nanmean()) ** 2).nanmean()) ** 0.5 + 1e-6
            ir_samp = (ic_samp.nanmean() / ic_std * np.sqrt(const_annual)).nan_to_num() # 年化
            ir_list.append(ir_samp.cpu())
    # '记录结束时间并输出txt'
    # end_time_sku = time.time()
    # print(str(individual),ir,'end_time',time.ctime(end_time_sku),'time_cost',time.strftime("%H:%M:%S", time.gmtime(end_time_sku-start_time_sku)))
    # output_path = f'{_DIR_job}/z_{pool_skuname}_end_{time.strftime("%H_%M_%S", time.gmtime(end_time_sku-start_time_sku))}.txt'
    # with open(output_path, 'w', encoding='utf-8') as file1:
    #     print(str(individual), '\n',ir,'\n', 'start_time', time.ctime(start_time_sku),file=file1)
    #     print(' end_time', time.ctime(end_time_sku), '\n time_cost',
    #           time.strftime("%H:%M:%S", time.gmtime(end_time_sku - start_time_sku)),file=file1)
    return abs(ir_list[0]) , ir_list


# %%
def gp_toolbox(gp_args , i_iter , max_depth = 5 , **kwargs):
    def _delnames(obj , names):
        [(delattr(obj , n) if hasattr(obj , n) else None) for n in names]
            
    pset = gp.PrimitiveSetTyped("main", [gpTypes.DF] * 14 + [gpTypes.DF_ori] * 15, gpTypes.DF)
    
    # ------------定义遗传算法中的初始因子（初始输入变量）--------------
    for i , v in enumerate(gp_args): pset.renameArguments(**{f'ARG{i}':v})
    
    #--------添加遗传算法中的随机正整数----------
    _delnames(gp , [f'POSINT{i}_{i_iter}' for i in range(6)])
    # pset.addEphemeralConstant('gpTypes.POSINT0',lambda: np.random.randint(0,10+1),gpTypes.POSINT0)
    # pset.addPrimitive(_returnself, [gpTypes.POSINT0], gpTypes.POSINT0,name='int_')
    pset.addEphemeralConstant(f'POSINT1_{i_iter}', lambda: np.random.randint(1,10+1), gpTypes.POSINT1)
    pset.addEphemeralConstant(f'POSINT2_{i_iter}', lambda: np.random.randint(2,10+1), gpTypes.POSINT2)
    for v in [10 , 15 , 20 , 40]:  pset.addTerminal(v, gpTypes.POSINT3)
    #for v in [1 , 5 , 10 , 20 , 40 , 60]:  pset.addTerminal(v, gpTypes.POSINT4)
    #for v in [60 , 120 , 180 , 200 , 240]: pset.addTerminal(v, gpTypes.POSINT5)

    # add primatives
    for prima in gpTypes.primatives_all(): pset.addPrimitive(*prima)

    '''创建遗传算法基础模块，以下参数不建议更改，如需更改，可参考deap官方文档'''
    # https://zhuanlan.zhihu.com/p/72130823
    _delnames(creator , ['FitnessMin' , 'Individual'])
    creator.create("FitnessMin", base.Fitness, weights=(+1.0,))   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset) # type:ignore 个体编码：pset，预设的

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_= max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)# type:ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)# type:ignore
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, compile_func = toolbox.compile , **kwargs) # type: ignore
    toolbox.register("select", tools.selTournament, tournsize=3) # 锦标赛：第一轮随机选择3个，取最大
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_= max_depth)  # genFull
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # type:ignore
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3

    '创建遗传算法种群，定义初始人口数及选取的最优个体数'
    return toolbox



# %%
def gp_eaSimple(toolbox , population , i_iter, pool_num=1,
                n_gen=2,cxpb=0.35,mutpb=0.25,hof_num=10,verbose=__debug__,stats=None,start_gen=None,gp_timer=None,**kwargs):  
    """
    Evolutionary Algorithm simple
    遗传算法全流程，依次完成选种、交叉、突变操作
    代码的粗略结构如下：
        evaluate(population)     # 对随机生成的初代种群评估IR值
        for g in range(n_gen)):
            population = select(population, len(population))    # 选取abs(IR)值较高的个体，以产生后代
            offspring = varAnd(population, toolbox, cxpb, mutpb)   # 交叉、变异
            evaluate(offspring)   # 对新种群评估IR值
            population = offspring    # 更新种群
    """

    """
    if start_gen is None:
        i_gen = 0
        logbook = tools.Logbook()
        logbook.header = ['i_gen', 'nevals'] + (stats.fields if stats else []) # type: ignore
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        '评估个体abs(IR)值'
        pool_list = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(invalid_ind))]

        if pool_num > 1:
            if verbose and i_gen == 0: print('muti'+str(pool_num))
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
            ind.ir_list = fit[1]

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(i_gen=i_gen, nevals=len(invalid_ind), **record)
        if verbose: print(logbook.stream)
        joblib.dump(population, f'{_DIR_job}/pop_iter{i_iter}_' + str(0) + '.pkl')
        joblib.dump(halloffame, f'{_DIR_job}/hof_iter{i_iter}_' + str(0) + '.pkl')
        joblib.dump(logbook   , f'{_DIR_job}/log_iter{i_iter}_' + str(0) + '.pkl')
        start_gen = i_gen + 1
    else:
        population = joblib.load(f'{_DIR_job}/pop_iter{i_iter}_' + str(start_gen-1) + '.pkl')
        halloffame = joblib.load(f'{_DIR_job}/hof_iter{i_iter}_' + str(start_gen-1) + '.pkl')
        logbook    = joblib.load(f'{_DIR_job}/log_iter{i_iter}_' + str(start_gen-1) + '.pkl')
    """

    if start_gen is not None and start_gen > 0:
        population = joblib.load(f'{_DIR_job}/pop_iter{i_iter}_{start_gen-1}.pkl')
        halloffame = joblib.load(f'{_DIR_job}/hof_iter{i_iter}_{start_gen-1}.pkl')
        logbook    = joblib.load(f'{_DIR_job}/log_iter{i_iter}_{start_gen-1}.pkl')
    else:
        start_gen = 0
        logbook = tools.Logbook()
        logbook.header = ['i_gen', 'nevals'] + (stats.fields if stats else []) #type:ignore
        halloffame = tools.HallOfFame(hof_num)

    # Begin the generational process
    for i_gen in range(start_gen, n_gen + 1):
        if i_gen == 0:
            offspring = population
        else:
            # Select the next generation individuals，abs(IR)越大，则被选中的概率越大
            offspring = toolbox.select(population, len(population))
            # Vary the pool of individuals: varAnd means variation part (crossover and mutation)
            varAnd_timer  = gpAccTimer()
            with varAnd_timer:
                offspring = varAnd(offspring, toolbox, cxpb , mutpb)

        # Evaluate the individuals with an invalid fitness（对于发生改变的个体，重新评估abs(IR)值）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pool_list = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(invalid_ind))]
        if pool_num > 1:
            pool = Pool(pool_num)
            # fitnesses = list(tqdm(pool.imap(toolbox.evaluate, invalid_ind, pool_list), total=len(invalid_ind), desc="gen"+str(i_gen)))
            fitnesses = pool.starmap(toolbox.evaluate, zip(invalid_ind, pool_list), chunksize=1)
            pool.close()
            pool.join()
            # pool.clear()
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, pool_list)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = tuple([fit[0]])
            ind.ir_list = fit[1]

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(i_gen=i_gen, nevals=len(invalid_ind), **record)
        if verbose: print(logbook.stream)
        joblib.dump(population, f'{_DIR_job}/pop_iter{i_iter}_{i_gen}.pkl')
        joblib.dump(halloffame, f'{_DIR_job}/hof_iter{i_iter}_{i_gen}.pkl')
        joblib.dump(logbook   , f'{_DIR_job}/log_iter{i_iter}_{i_gen}.pkl')
        
    if gp_timer: gp_timer.append_time('avg_varAnd' , varAnd_timer.avgtime())
    return population, halloffame , logbook


# %%
def gp_hof_eval(toolbox , halloffame, i_iter , gp_values ,
                ir_floor=2.0,corr_cap=0.5,verbose=__debug__,gp_timer=None,**kwargs):
    compute_timer  = gpAccTimer()
    '''因子库存储在hof_good_list中，每个因子为一个二维tensor（时间*股票）'''
    hof_good_list = []
    if i_iter > 0 and os.path.exists(f'{_DIR_job}/good_log.csv'): 
        good_log = pd.read_csv(f'{_DIR_job}/good_log.csv',index_col=0)
    else:
        good_log = pd.DataFrame(columns=['i_iter','str_hof','rankir_in_resid','rankir_out_resid','rankir_in','rankir_out'])

    '''对rankIR大于设定值的因子，进行相关性检验，如果相关性低，则加入因子库'''
    for hof_single in halloffame:
        if ((hof_single.fitness.values[0] > ir_floor) and 
            (not hof_single.fitness.values[0].isinf()) and 
            (hof_single.ir_list[1] != 0)): # icir_out_resid
            
            with compute_timer:
                hof_func = toolbox.compile(expr=hof_single) # type: ignore
                hof_value = hof_func(*gp_values)
                hof_value[hof_value.isinf()] = torch.nan
            
            # 与已有的因子库做相关性检验，如果相关性大于预设值，则不加入因子库
            good_tag = True
            for hof_good in hof_good_list:
                corr_value = F.corrwith_torch_matrix(hof_value, hof_good, dim=1).nanmean().abs()
                # print(corr_value)
                if corr_value > corr_cap: 
                    good_tag = False
                    break

            # 如果通过相关性检验，则加入因子库
            if good_tag:
                hof_good_list.append(hof_value)
                str_hof = str(hof_single)
                if verbose or True: print('good : ' + str_hof)
                
                good_log = pd.concat([good_log,pd.DataFrame([[i_iter,str_hof,*hof_single.ir_list]],columns=good_log.columns)],axis=0)
                
                hof_value_df = pd.DataFrame(hof_value.cpu(), index=kwargs['timerow'], columns=kwargs['stockcol']) # type: ignore
                hof_value_df.to_parquet(f'{_DIR_job}/factor/{str_hof}.parquet', engine='fastparquet')
    
    if gp_timer: gp_timer.append_time('avg_compute_factors' , compute_timer.avgtime())
    good_log.to_csv(f'{_DIR_job}/good_log.csv')
    return halloffame , hof_good_list


# %%
def main_loop(i_iter , gp_dict , gp_timer):
    timenow = time.time()
    gp_dict['i_iter'] = i_iter
    
    '''Setting GP toolbox and pop'''
    with gp_timer('setting' , print = False):
        toolbox = gp_toolbox(**gp_dict)
        population = toolbox.population(n = gp_dict['pop_num']) # type:ignore
    
    '''运行迭代主程序'''
    with gp_timer('gp' , print = False):
        population, halloffame, logbook = gp_eaSimple(toolbox , population , **gp_dict)  #, start_gen=6   #algorithms.eaSimple

    '''good_log文件为因子库，存储了因子表达式、rankIR值等信息。在具体因子值存储在factor文件夹的parquet文件中'''
    with gp_timer('selection' , print = False):
        halloffame, hof_good_list = gp_hof_eval(toolbox , halloffame, **gp_dict)

    # update labels_resid
    with gp_timer('neu_dump' , print = False):
        gp_dict['labels_resid'] = F.neutralize_numpy(gp_dict['labels_resid'], gp_dict['size'], gp_dict['cs_indus_code'], hof_good_list)
        '''保存该轮迭代的最终种群、最优个体、迭代日志'''
        joblib.dump(population,f'{_DIR_job}/pop_iter{i_iter}_overall.pkl')
        joblib.dump(halloffame,f'{_DIR_job}/hof_iter{i_iter}_overall.pkl')
        joblib.dump(logbook   ,f'{_DIR_job}/log_iter{i_iter}_overall.pkl')

    gp_timer.append_time('all' , time.time() - timenow)
    return population , halloffame , logbook
    

# %%
# main process
def main(test_code = None , job_id = None):
    gp_timer = gpTimer()
    gp_dict = gp_dictionary(gp_parameters(test_code , job_id) , gp_timer)
    for i_iter in range(gp_dict['n_iter']):
        main_loop(i_iter , gp_dict , gp_timer)
    gp_timer.save_to_csv(f'{_DIR_job}/saved_times.csv' , 
                         ['setting','gp','selection','neu_dump','all','avg_compute_factors','avg_varAnd'])

if __name__ == '__main__':
    main()
