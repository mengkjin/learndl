# %%
import pandas as pd
import numpy as np
import os , sys , copy , tqdm , shutil , gc , re
import torch
import array , random , json , operator , time, platform , joblib
from argparse import ArgumentParser , Namespace
from tqdm import tqdm
import cProfile

from deap import base , creator , tools , gp
from deap.algorithms import varAnd
from torch.multiprocessing import Pool

import gp_math_func as F
from gp_utils import gpHandler , gpTimer , gpContainer

# %%
# ------------------------ environment setting ------------------------

# np.seterr(over='raise')

_plat      = platform.system().lower()                                    # Windows or Linux
_device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda or cpu
_test_code = True or (_device == torch.device('cpu'))                     # True if just to test code is valid
_DIR_data  = './data/features/parquet'                                    # input路径，即原始因子所在路径。
_DIR_pack  = './data/package'                                             # input路径2, 加载原始因子后保存pt文件加速存储。
_DIR_pop   = './pop'                                                      # output路径，即保存因子库、因子值、因子表达式的路径。
_DIR_job   = f'{_DIR_pop}/bendi'                                          # 测试代码时的job path 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                               # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099


parser = ArgumentParser()
parser.add_argument("--job_id", type=int, default=-1)
parser.add_argument("--poolnm", type=int, default=1)
args, _ = parser.parse_known_args()

assert args.poolnm == 1 , args.poolnm # 若并行，通信成本过高，效率提升不大。
# multiprocessing method, 单机多进程设置（实际未启用），参考https://zhuanlan.zhihu.com/p/600801803
torch.multiprocessing.set_start_method('spawn' if _plat == 'windows' else 'forkserver', force=True) 

# ------------------------ gp parameters ------------------------
def gp_parameters(test_code = None , job_id = None , train = True):
    '''
    ------------------------ gp parameters ------------------------
    主要遗传规划参数初始化
    注：为方便跑多组实验，需设置job_id参数，设置方式为: python xxx.py --job_id 123456
    input:
        test_code: if only to test code validity
        job_id:    when test_code is not True, determines _DIR_job = f'{_DIR_pop}/{job_id}' 
        train:     if True, will first check dirs and device  
    output:
        gp_params: dict that includes all gp parameters
    '''
        
    # directory setting and making
    global _DIR_job
    if test_code is None: test_code = _test_code
    if train:
        if job_id is None: job_id = args.job_id
        if test_code:
            _DIR_job = f'{_DIR_pop}/bendi'
            if os.path.exists(_DIR_job): shutil.rmtree(_DIR_job)
        else:
            if job_id < 0:
                old_job_df = os.listdir('./pop/') if os.path.exists('./pop/') else []
                old_job_id = np.array([id for id in old_job_df if id.isdigit()]).astype(int)
                job_id = np.setdiff1d(np.arange(max(old_job_id) + 2) , old_job_id).min() if len(old_job_id) else 0
            _DIR_job = f'{_DIR_pop}/{job_id}'   
            # if _DIR_job exists, make sure you know it is for continuation computing
            if os.path.exists(_DIR_job) and not input(f'{_DIR_job} exists , press "yes" to confirm continuation:')[0].lower() == 'y':
                raise Exception(f'{_DIR_job} exists!')
        os.makedirs(f'{_DIR_job}/factor' , exist_ok=True)
        os.makedirs(_DIR_pack , exist_ok=True)

        if (_device == torch.device('cpu')):
            print('**Cuda not available')
        else:
            print('**Device name:', torch.cuda.get_device_name(0), ', Available:' ,torch.cuda.is_available())
        print(f'  --> Pop directory is : "{_DIR_job}"')

    '''
    test_code     是不是仅仅作为代码测试，若是会有一个小很多的参数组合覆盖本参数
    gp_fac_list   作为GP输入时，标准化因子的数量
    gp_raw_list   作为GP输入时，原始指标的数量
    slice_date:   修改数据切片区间，前两个为样本内的起止点，后两个为样本外的起止点【均需要是交易日】'
    device:       用cuda还是cpu计算
    verbose:      训练过程是否输出细节信息
    pool_num:     并行任务数量，建议设为1，即不并行，使用单显卡单进程运行。
    pop_num:      种群数量，初始化时生成多少个备选公式
    hof_num:      精英数量。一般精英数量设为种群数量的1/6左右即可。
    n_iter:       【大循环】的迭代次数，每次迭代重新开始一次遗传规划、重新创立全新的种群，以上一轮的残差收益率作为优化目标。
    ir_floor:     【大循环】中因子入库所需的最低rankIR值，低于此值的因子不入库。
    corr_cap:     【大循环】中新因子与老因子的最高相关系数，相关系数绝对值高于此值的因子不入库。
    n_gen:        【小循环】的迭代次数，即每次遗传规划进行几轮繁衍进化。
    max_depth:    【小循环】中个体算子树的最大深度，即因子表达式的最大复杂度。
    cxpb:         【小循环】中交叉概率，即两个个体之间进行交叉的概率。
    mutpb:        【小循环】中变异概率，即个体进行突变变异的概率。
    '''

    gp_fac_list = ['cp', 'turn', 'vol', 'amt', 'op', 'hp', 'lp', 'vp', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']# in lower case
    gp_raw_list = [v.upper() for v in gp_fac_list] + ['RTN'] # in upper case
    slice_date  = ['2010-01-04', '2021-12-31', '2022-01-04', '2099-12-31']

    gp_params = gpContainer(
        test_code = test_code  ,     # just to check code, 
        gp_fac_list = gp_fac_list ,   # gp intial factor list 
        gp_raw_list = gp_raw_list ,   # gp intial raw data list
        slice_date  = slice_date ,    # must be trade date
        device = _device ,            # training device, cuda or cpu
        verbose = False ,             # if show some text
        pool_num = args.poolnm ,      # multiprocessing pool number
        pop_num= 3000 ,               # [parameter] population number
        hof_num= 500 ,                # [parameter] halloffame number
        n_iter =  5 ,                 # [outer loop] loop number
        ir_floor = 3.0 ,              # [outer loop] rankir threshold
        corr_cap = 0.6 ,              # [outer loop] cap of correlation with existing factors
        n_gen = 5  ,                  # [inner loop] generation number
        max_depth = 5 ,               # [inner loop] max tree depth of gp
        survive_rate = 0.8 ,          # [inner loop] rate last generation can go down to next generation
        cxpb = 0.35 ,                 # [inner loop] crossover probability
        mutpb = 0.25 ,                # [inner loop] mutation probability
    )
    if test_code:
        # when test code, change some parameters
        gp_params.update(verbose = True , pop_num = 20 , hof_num = 5 , n_iter = 2 , max_depth = 3 , ir_floor = 2.0 , corr_cap = 0.7 , 
                         slice_date = ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29']) 
    gp_params.apply('slice_date' , lambda x:pd.to_datetime(x).values)
    F.invalid = F.invalid.to(gp_params.get('device'))
    return gp_params

# %%
class MemoryManager():
    unit = 1024**3

    def __init__(self , device_no = 0) -> None:
        self.device_no = device_no
        self.unit = type(self).unit
        self.gmem_total = torch.cuda.mem_get_info()[1] / self.unit
        self.record = {}
        self.check(showoff = True)

    def check(self , key = None, showoff = False , critical_ratio = 0.5):
        gmem_free = torch.cuda.mem_get_info(self.device_no)[0] / self.unit
        if gmem_free > critical_ratio * self.gmem_total: return gmem_free

        torch.cuda.empty_cache() # collect graphic memory 
        gc.collect() # collect memory
        gmem_freed = torch.cuda.mem_get_info(self.device_no)[0] / self.unit - gmem_free
        gmem_free += gmem_freed
        gmem_allo  = torch.cuda.memory_allocated(self.device_no) / self.unit
        gmem_rsrv  = torch.cuda.memory_reserved(self.device_no) / self.unit
        
        if key is not None:
            if key not in self.record.keys(): self.record[key] = []
            self.record[key].append(gmem_freed)
        if showoff: print(f'**Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 
        return gmem_free

    def __bool__(self):
        return True
    
    @classmethod
    def object_memory(cls , object):
        if isinstance(object , torch.Tensor):
            return cls.tensor_memory(object)
        elif isinstance(object , (list,tuple)):
            return sum([cls.object_memory(obj) for obj in object])
        elif isinstance(object , dict):
            return sum([cls.object_memory(obj) for obj in object.values()])
        else:
            return 0.
    
    @classmethod
    def tensor_memory(cls , tensor):
        total_memory = tensor.element_size() * tensor.numel()
        return total_memory / cls.unit
    
    def print_memeory_record(self):
        if len(self.record):
            print(f'  --> Avg Freed Cuda Memory: ')
            for key , value in self.record.items():
                print(f'     --> {key} : {len(value)} counts, on average freed {np.mean(value):.2f}G')
    
# %%
def gp_namespace(gp_params , gp_timer = gpTimer()):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    基于遗传规划的参数字典，读取各类主要数据，并放在同一字典中传回
    input:
        gp_params: gpContainer of all gp_parameters
        gp_timer:  gpTimer to record time cost
    output:
        gp_space:  gpContainer that includes gp parameters, gp datas and gp arguements, and various other datas
    '''
    
    with gp_timer('Data' , df_cols = False , print_str= '**Load Data'):
        gp_space = gp_params.copy().update(gp_values = [] , df_columns = None)
        gp_space.gp_args = gp_space.gp_fac_list + gp_space.gp_raw_list
        gp_space.n_args  = (len(gp_space.gp_fac_list) , len(gp_space.gp_raw_list))
        
        package_path = f'{_DIR_pack}/gp_data_package' + '_test' * gp_space.test_code + '.pt'
        package_require = ['gp_args' , 'gp_values' , 'size' , 'indus' , 'labels_raw' , 'labels_res' , 'df_index' , 'df_columns' , 'universe']
        if os.path.exists(package_path):
            if gp_space.verbose: print(f'  --> Directly load {package_path}')
            package_data = torch.load(package_path)

            assert np.isin(gp_space.gp_args , package_data['gp_args']).all() , np.setdiff1d(gp_space.gp_args , package_data['gp_args'])
            assert np.isin(package_require , list(package_data.keys())).all() , np.setdiff1d(package_require , list(package_data.keys()))

            for gp_key in gp_space.gp_args:
                gp_val = package_data['gp_values'][package_data['gp_args'].index(gp_key)]
                gp_val = df_to_ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_values.append(gp_val)

            for gp_key in ['size' , 'indus' , 'labels_raw' , 'labels_res' , 'universe']: 
                gp_val = package_data[gp_key]
                gp_val = df_to_ts(gp_val , gp_key , gp_space.device)
                gp_space.set(gp_key , gp_val)

            for gp_key in ['df_index' , 'df_columns']: 
                gp_val = package_data[gp_key]
                gp_space.set(gp_key , gp_val)

        else:
            if gp_space.verbose: print(f'  --> Load from parquet files')
            gp_filename = gp_filename_converter()
            nrowchar = 0
            for i , gp_key in enumerate(gp_space.gp_args):
                if gp_space.verbose and nrowchar == 0: print('    --> ' , end='')
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.slice_date,gp_space.df_columns)
                if i == 0: gp_space.update({'df_columns' : gp_val.columns.values ,'df_index': gp_val.index.values})
                gp_val = df_to_ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_values.append(gp_val)
                
                if gp_space.verbose:
                    print(gp_key , end=',')
                    nrowchar += len(gp_key) + 1
                    if nrowchar >= 100 or i == len(gp_space.gp_args):
                        print()
                        nrowchar = 0

            for gp_key in ['size' , 'indus']: 
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.slice_date,gp_space.df_columns)
                gp_val = df_to_ts(gp_val , gp_key , gp_space.device)
                gp_space.set(gp_key , gp_val)

            CP = gp_space.gp_values[gp_space.gp_args.index('CP')]      
            gp_space.universe   = ~CP.isnan() # type:ignore
            gp_space.labels_raw = gp_get_labels(CP , gp_space.size , gp_space.indus)
            gp_space.labels_res = copy.deepcopy(gp_space.labels_raw)
            torch.save(gp_space.subset(package_require) , package_path)

        if gp_space.verbose: print(f'  --> {len(gp_space.gp_fac_list)} factors, {len(gp_space.gp_raw_list)} raw data loaded!')

    gp_space.insample  = (gp_space.df_index >= gp_space.slice_date[0]) * (gp_space.df_index <= gp_space.slice_date[1])
    gp_space.outsample = (gp_space.df_index >= gp_space.slice_date[2]) * (gp_space.df_index <= gp_space.slice_date[3])
    gp_space.gp_timer = gp_timer
    gp_space.memory_manager = MemoryManager()
    return gp_space

def gp_get_labels(CP = None , neutral_factor = None , neutral_group = None , nday = 10 , delay = 1 , 
                  slice_date = None, df_columns = None , device = None):
    if CP is None:
        CP = df_to_ts(read_gp_data(gp_filename_converter()('CP'),slice_date,df_columns) , 'CP' , device)    
    labels = F.ts_delay(F.pctchg(CP, nday) , -nday-delay)  # t+1至t+11的收益率
    labels = F.neutralize_2d(labels, neutral_factor, neutral_group)  # 市值行业中性化
    return labels

def gp_filename_converter():
    '''
    ------------------------ gp input data filenames ------------------------
    原始因子名与文件名映射
    input:

    output:
        wrapper: function to convert gp_key into parquet filename 
    '''

    filename_dict = {'op':'open','hp':'high','lp':'low','vp':'vwap','cp':'close_adj',
                     'vol':'volume','bp':'bp_lf','ep':'ep_ttm','ocfp':'ocfp_ttm',
                     'dp':'dividendyield2','rtn':'return1','op':'open','indus':'cs_indus_code'}
    def wrapper(gp_key):
        assert gp_key.isupper() or gp_key.islower() , gp_key

        rawkey = gp_key.lower()
        if rawkey in filename_dict.keys(): rawkey = filename_dict[rawkey]

        zscore = gp_key.islower() and rawkey not in ['cs_indus_code' , 'size']

        return f'{_DIR_data}/{rawkey}' + '_zscore' * zscore + '_day.parquet'
    return wrapper

def read_gp_data(filename,slice_date=None,df_columns=None,df_index=None,input_freq='D'):
    '''
    ------------------------ read gp data and convert to torch.tensor ------------------------
    读取单个原始因子文件并转化成tensor，额外返回df表格的行列字典
    input:
        filename:    filename gp data
        slice_date:  [insample_start, insample_end, outsample_start, outsample_end]
        df_columns:  if not None, filter columns
        df_index     if not None, filter rows, cannot be used if slice_date given
        input_freq:  freq faster than day should code later
    output:
        ts:          result tensor
        df_indexes:  index and columns if first_data is True
    '''
    df = pd.read_parquet(filename, engine='fastparquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: 
        df = df[(slice_date[0] <= df.index.values) & (df.index.values <= slice_date[-1])] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    if df_columns is not None: df = df.loc[:,df_columns]# 选取指定股票
    if slice_date is None and df_index is not None: df = df.loc[df_index]
    return df

def df_to_ts(x , gp_key = '' , device = None , share_memory = True):
    # additional treatment based by gp_key
    if isinstance(x , pd.DataFrame): x = torch.FloatTensor(x.values)
    if gp_key.startswith('dp'): # dividend factor , nan means 0
        x = x.nan_to_num()
    if isinstance(x , torch.Tensor):
        if device is not None: x = x.to(device)
        if share_memory: x.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return x

def save_gp_data(data , filename , index = None , columns = None):
    '''
    ------------------------ save gp data as parquet ------------------------
    save单个原始因子文件
    input:
        data:        canbe tensor in cuda
        filename:    filename gp data
        index , columns: indexes
    output:
    '''
    if isinstance(data , torch.Tensor): data = data.cpu().numpy()
    df = pd.DataFrame(data=data , index=index,columns=columns)
    df.to_parquet(filename , engine='fastparquet')

# %%
def gp_population(toolbox , pop_num , max_round = 100 , last_generation = [], **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    初始化种群
    input:
        gp_args:    initial gp factor names
        i_iter:     i of outer loop
        max_round:  max iterations to approching 99% of pop_num
        pop_num:    population number
        last_generation: starting population
    output:
        toolbox:    toolbox that contains all gp utils
        population: initial population of syntax
    '''
    if last_generation: 
        population = toolbox.prune(last_generation)
    else:
        population = []
    for _ in range(max_round):
        new_comer  = toolbox.population(n = pop_num - len(population))
        new_comer  = toolbox.prune(new_comer) 
        population = population + new_comer #type:ignore
        population = toolbox.duplicate(population) 
        if len(population) >= 0.99 * pop_num: break

    return population

# %%
def evaluate(individual, pool_skuname, compiler , gp_values , labels_raw , labels_res, universe, insample , outsample , gp_timer , 
             const_annual = 24 , min_coverage = 0.5 , i_iter = -1 , **kwargs):
    '''
    ------------------------ evaluate individual syntax fitness ------------------------
    从因子表达式起步，生成因子并计算适应度
    input:
        individual:     individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        pool_skuname:   pool skuname in pool.imap, e.g. iter0_gen0_0
        compiler:       compiler function to realize syntax computation, i.e. return factor function of given syntax
        gp_values:      initial population factor values
        labels_raw:     raw label, e.g. 20 day future return
        labels_res:     residual label, neutralized by indus and size
        insample:       insample_dates
        outsample:      outsample_dates
        const_annual:   constant or annualization
        min_coverage:   minimum daily coverage
    output:
        tuple of (
            abs_rankir: (abs(insample_resid), ) # !! Fitness definition 
            rankir:     (insample_resid, outsample_resid, insample_raw, outsample_raw)
        )
    '''
    if int(pool_skuname.split('_')[-1])%100 == 0:
        start_time_sku = time.time()
        output_path = f'{_DIR_job}/z_{pool_skuname}.txt'
        with open(output_path, 'w', encoding='utf-8') as file1:
            print(str(individual),'\n start_time',time.ctime(start_time_sku),file=file1)

    # compile syntax and calculate factors
    factor_func  = gp_compile(compiler , individual , gp_timer.acc_timer('compile'))
    factor_value = gp_factor(factor_func , gp_values , 'inf' , gp_timer.acc_timer('eval'))
    # first return must be list or tuple, since fitness can be weighted
    # "Assigned values have not the same length than fitness weights"
    if F.is_invalid(factor_value): return [F.invalid] * 4

    rankir_list = []
    for labels in [labels_res , labels_raw]:
        rankic_full = F.rankic_2d(factor_value , labels , dim = 1 , universe = universe , min_coverage = min_coverage)
        if F.is_invalid(rankic_full): 
            rankir_list += [F.invalid] * 2
            continue
        
        for sample in [insample , outsample]:
            rankic = rankic_full[sample]
            if rankic.isnan().sum() > 0.5 * len(rankic): # if too many nan rank_ic (due to low coverage)
                rankir_list.append(0.)
            else:
                rankic_std = (rankic - rankic.nanmean()).square().nanmean().sqrt() + 1e-6
                rankir_samp = (rankic.nanmean() / rankic_std * np.sqrt(const_annual)) # 年化 ir
                rankir_samp[rankir_samp.isinf()] = 0.
                rankir_list.append(rankir_samp.nan_to_num().cpu())
            if rankir_list[-1] == 0 and i_iter >= 1:
                print(factor_value)
                print(rankir_list)
                raise Exception()
    return rankir_list

def gp_compile(compiler , individual , timer = gpTimer.EmptyTimer() , **kwargs):
    '''
    ------------------------ calculate individual syntax factor value ------------------------
    根据迭代出的因子表达式，计算因子值 , 基于函数表达式构建函数：deap.base.Toolbox().compile
    input:
        compiler:     compiler function to realize syntax computation, i.e. return factor function of given syntax
        individual:   individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        timer:        record compile time
    output:
        func:         function of a syntax
    '''
    with timer: 
        func = compiler(individual)
    return func

def gp_factor(func , gp_values , stream = 'inf' , timer = gpTimer.EmptyTimer() , **kwargs):
    '''
    ------------------------ calculate individual factor value ------------------------
    根据迭代出的因子表达式，计算因子值 , 基于函数表达式构建函数：deap.base.Toolbox().compile
    input:
        func:         function of syntax
        gp_values:    initial population factor values
        timer:        record compile time
    output:
        factor_value: 2d tensor
    '''
    with timer:
        factor_value = func(*gp_values)
        factor_value = process_factor(factor_value , stream , dim = 1)
        
    return factor_value

def process_factor(value , stream = 'inf_trim_norm' , dim = 1 , trim_ratio = 7. , norm_tol = 1e-4,
                   size = None , indus = None , **kwargs):
    '''
    ------------------------ process factor value ------------------------
    处理因子值 , 'inf_trim_winsor_norm_neutral_nan'
    input:
        value:         factor value to be processed
        process_key:   can be any of 'inf_trim/winsor_norm_neutral_nan'
        dim:           default to 1
        trim_ratio:    what extend can be identified as outlier? range is determined as med ± trim_ratio * brandwidth
        norm_tol:      if norm required, the tolerance to eliminate factor if standard deviation is too trivial
        size:          market cap (neutralize param)
        indus:         industry indicator (neutralize param)
    output:
        value:         processed factor value
    '''
    if F.is_invalid(value) or F.allna(value , inf_as_na = True): return F.invalid

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
            value = torch.where(trivial , value * 0 , F.zscore(value , dim))
        # elif _str == 'neutral': 
            # '市值中性化（对x做，现已取消，修改为对Y做）'
            # value = F.neutralize_2d(value , size , indus , dim = dim) 
        elif _str == 'nan': 
            value = value.nan_to_num_()
    return value

# %%
def gp_eaSimple(toolbox , i_iter, pop_num  , pool_num=1,
                n_gen=2,cxpb=0.35,mutpb=0.25,hof_num=10, survive_rate=0.8,
                start_gen=None,gp_timer=gpTimer(),verbose=__debug__,stats=None,**kwargs):  
    """
    ------------------------ Evolutionary Algorithm simple ------------------------
    变异/进化小循环，从初始种群起步计算适应度并变异，重复n_gen次
    input:
        toolbox:    toolbox that contains all gp utils
        i_iter:     i of outer loop
        pop_num:    population number
        pool_num:   multiprocessing pool number
        n_gen:      [inner loop] generation number
        cxpb:       [inner loop] crossover probability
        mutpb:      [inner loop] mutation probability
        hof_num:    halloffame number
        survive_rate: how many last generation survivors can go down to next generation
        start_gen:  which gen to start, if None start a new
        gp_timer:   gpTimer to record time cost
    output:
        population: updated population of syntax
        halloffame: container of individuals with best fitness (no more than hof_num)
        logbook:    gp logbook

    ------------------------ basic code structure ------------------------
    evaluate(population)     # 对随机生成的初代种群评估IR值
    for g in range(n_gen)):
        evaluate(population)   # 对新种群评估IR值
        population = select(population, len(population))    # 选取abs(IR)值较高的个体，以产生后代
        offspring = varAnd(population, toolbox, cxpb, mutpb)   # 交叉、变异
        population = offspring    # 更新种群
    """

    if start_gen is not None and start_gen > 0:
        offspring  = joblib.load(f'{_DIR_job}/pop_iter{i_iter}_{start_gen-1}.pkl')
        halloffame = joblib.load(f'{_DIR_job}/hof_iter{i_iter}_{start_gen-1}.pkl')
        logbook    = joblib.load(f'{_DIR_job}/log_iter{i_iter}_{start_gen-1}.pkl')
    else:
        start_gen = 0
        offspring = []
        halloffame = tools.HallOfFame(hof_num)
        logbook = tools.Logbook()
        logbook.header = ['i_gen', 'n_evals'] + (stats.fields if stats else []) #type:ignore

    for i_gen in range(start_gen, n_gen + 1):
        population = gp_population(toolbox , pop_num , last_generation = offspring)
        # population = offspring # old method
        if i_gen == 0:
            print(f'**A Population({len(population)}) has been Initialized')
        else:
            print(f'     --> Survive {len(offspring)} Offsprings, Populate to {len(population)} ones')
        
        # Evaluate the individuals with an invalid fitness（对于发生改变的个体，重新评估abs(IR)值）
        survivors = [ind for ind in population if not ind.fitness.valid] # 'individual' arg for evaluate
        pool_list = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(survivors))] # 'pool_skuname' arg for evaluate
        desc = f'  --> Evolve Generation {str(i_gen)}'
        fitnesses = gp_fitnesses(pool_num , toolbox , toolbox.evaluate , survivors , pool_list , desc = desc)
        for ind, fit in zip(survivors, fitnesses):
            ind.factor_valid = not F.is_invalid(fit[0])
            ind.ir_list = [(0 if F.is_invalid(rankir) else rankir) for rankir in fit]
            ind.fitness.values = (abs(ind.ir_list[0]),)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(i_gen=i_gen, n_evals=len(survivors), **record)

        # Update the hall of fame with the generated individuals
        survivors = [ind for ind in survivors if ind.factor_valid] # 'individual' arg for evaluate
        halloffame.update(survivors)

        # if verbose: [print('     -->   ' + s) for s in str(logbook.stream).split('\n')]
        joblib.dump(population, f'{_DIR_job}/pop_iter{i_iter}_{i_gen}.pkl')
        joblib.dump(halloffame, f'{_DIR_job}/hof_iter{i_iter}_{i_gen}.pkl')
        joblib.dump(logbook   , f'{_DIR_job}/log_iter{i_iter}_{i_gen}.pkl')
            
        # Select the next generation individuals，based on abs(IR)
        offspring = toolbox.select(survivors, min(int(survive_rate * pop_num) , len(survivors)))
        # Vary the pool of individuals: varAnd means variation part (crossover and mutation)
        with gp_timer.acc_timer('varAnd'):
            offspring = varAnd(offspring, toolbox, cxpb , mutpb)

    print(f'**A HallofFame({len(halloffame)}) has been Evolutionized')
    return population, halloffame, logbook

def gp_fitnesses(pool_num , toolbox , eval_func , population , pool_skunames = None , desc = ''):
    if pool_skunames is None: pool_skunames = [f'pool_sku{i}' for i in range(len(population))]
    if pool_num > 1:
        pool = Pool(pool_num)
        fitnesses = list(tqdm(pool.imap(eval_func, zip(population, pool_skunames)), 
                              total=len(population), desc=desc)) #type:ignore
        #fitnesses = pool.starmap(toolbox.evaluate, zip(survivors, pool_list), chunksize=1)
        pool.close()
        pool.join()
        # pool.clear()
    else:
        fitnesses = list(tqdm(toolbox.map(eval_func, population, pool_skunames),
                              total=len(population), desc=desc))
    return fitnesses

# %%
def gp_hof_eval(toolbox , halloffame, i_iter , gp_values , df_index , df_columns , 
                ir_floor=2.5,corr_cap=0.7,gp_timer=gpTimer(),memory_manager=None,
                verbose=__debug__,**kwargs):
    """
    ------------------------ gp halloffame evaluation ------------------------
    筛选因子表达式，并加入名人堂中，标准是高ir、低相关
    input:
        toolbox:       toolbox that contains all gp utils
        halloffame:    container of individuals with best fitness
        i_iter:        i of outer loop
        gp_values:     initial population factor values
        ir_floor:      [outer loop] rankir threshold
        corr_cap:      [outer loop] cap of correlation with existing factors
        gp_timer:      gpTimer to record time cost
    output:
        halloffame:    new container of individuals with best fitness
        hof_good_list: good hof values who pass the criterions
    """
    
    good_hofs = F.invalid
    good_cols = ['i_iter','i_good','sytax','max_corr','rankir_in_resid','rankir_out_resid','rankir_in','rankir_out']
    good_path = f'{_DIR_job}/good_log.csv'
    good_logs = pd.read_csv(good_path,index_col=0) if i_iter > 0 and os.path.exists(good_path) else pd.DataFrame()
    i_good = len(good_logs)

    hof_valid   = torch.Tensor([hof_single.factor_valid      for hof_single in halloffame]) # icir valid
    hof_rankir  = torch.Tensor([hof_single.fitness.values[0] for hof_single in halloffame]) # icir larger than ir_floor
    hof_rankir2 = torch.Tensor([hof_single.ir_list[1]        for hof_single in halloffame]) # icir_out_resid not zero (i.e. not nan)
    good_sign   = hof_valid * (hof_rankir > ir_floor) * ~hof_rankir2.isnan()
    print(f'  --> HallofFame({len(halloffame)}) Contains {good_sign.sum().int()} Promising Candidates with RankIR >= {ir_floor:.2f}')
    if good_sign.sum() <= 0.1 * len(halloffame):
        # Failure of finding promising offspring , check if code has bug
        print(f'  --> Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ')
        print(f'  --> Valid hof({hof_valid.sum()}), insample max ir({hof_rankir.max():.4f}), outsample nonnan({(~hof_rankir2.isnan()).sum()})')

    for i , hof_single in enumerate(halloffame):
        if not good_sign[i]: continue

        # 根据迭代出的因子表达式，计算因子值
        factor_func  = gp_compile(toolbox.compile , hof_single , gp_timer.acc_timer('compile'))
        factor_value = gp_factor(factor_func , gp_values , 'inf_trim_norm' , gp_timer.acc_timer('eval'))
        if memory_manager: memory_manager.check('factor')

        # 与已有的因子库做相关性检验，如果相关性大于预设值corr_cap，则不加入因子库
        corr_values = torch.ones(good_hofs.shape[-1]).to(good_hofs) * 10000
        if not F.is_invalid(factor_value): 
            for i_hof in range(good_hofs.shape[-1]):
                corr = F.corrwith(factor_value, good_hofs[...,i_hof], dim=1).nanmean().abs()  # F.corrwith(factor_value, good_hofs[...,i_hof]).abs()
                if corr > corr_cap: break
                corr_values[i_hof] = corr
                if memory_manager: memory_manager.check('corr')
        
        max_corr = 0. if corr_values.numel() == 0 else corr_values.nan_to_num().max().item()
        good_sign[i] = max_corr < corr_cap
        if not good_sign[i]: continue

        # 如果通过相关性检验，则加入因子库
        hof_str = str(hof_single).replace(' ','')
        print(f'     --> Good {i_good}: RankIR {hof_rankir[i]:.2f} MaxCorr {max_corr:.2f}: {hof_str}')

        good_hofs = F.concat_factors(good_hofs , factor_value)
        save_gp_data(factor_value,f'{_DIR_job}/factor/iter{i_iter}_good{i_good}.parquet',df_index,df_columns)

        new_log   = pd.DataFrame([[i_iter,i_good,hof_str,max_corr,*hof_single.ir_list]],columns =good_cols)
        good_logs = pd.concat([good_logs , new_log],axis=0) if len(good_logs) else new_log
        i_good += 1

    del hof_valid , hof_rankir , hof_rankir2
    # halloffame = [hof_single for good , hof_single in zip(good_sign , halloffame) if good]
    good_logs.to_csv(good_path)
    print(f'Cuda Memories of good_logs , gp_values and others take {MemoryManager.object_memory([good_logs , gp_values , kwargs]):.2f}G')

    return halloffame , good_hofs

# %%
def outer_loop(i_iter , gp_space):
    """
    ------------------------ gp outer loop ------------------------
    大循环主程序，初始化种群、变异、筛选、更新残差labels、记录
    input:
        i_iter:   i of outer loop
        gp_space:  dict that includes gp parameters, gp datas and gp arguements, and various other datas
    output:
        None
    process:
        gp_population : create initial population
        gp_eaSimple   : Evolutionary Algorithm simple
        gp_hof_eval   : gp halloffame evaluation
        update labels : update labels_res , according to hof_good_list
    """
    timenow = time.time()
    gp_space.i_iter = i_iter
    
    '''Initialize GP toolbox'''
    with gp_space.gp_timer('Setting' , print_str = f'**Initialize GP Toolbox'):
        toolbox = gpHandler.Toolbox(eval_func = evaluate , **gp_space)

    '''运行进化主程序'''
    with gp_space.gp_timer('Evolution' , print_str = f'**{gp_space.n_gen+1} Generations of Evolution'):
        mem_free = gp_space.memory_manager.check(showoff = True)
        population, halloffame, logbook = gp_eaSimple(toolbox , **gp_space)  #, start_gen=6   #algorithms.eaSimple

    '''good_log文件为因子库，存储了因子表达式、rankIR值等信息。在具体因子值存储在factor文件夹的parquet文件中'''
    with gp_space.gp_timer('Selection' , print_str = f'**Select HallofFamers Save Factor Values'):
        mem_free = gp_space.memory_manager.check(showoff = True)
        halloffame, good_hofs = gp_hof_eval(toolbox , halloffame, **gp_space)
        
    print(f'**The HallofFame({len(halloffame)}) Eventually has {good_hofs.shape[-1]} Outstanders')

    '''更新labels_res，并保存本轮循环的最终结果'''
    with gp_space.gp_timer('Neu_dump' , print_str = f'**Update Residual Labels and Dump Results'):
        # update labels_resid, according to hof_good_list (maybe no need to neutralize according to size and indus)
        mem_free = gp_space.memory_manager.check(showoff = True)
        nan1 = gp_space.labels_res.isnan().sum()
        gp_space.labels_res = F.neutralize_2d(gp_space.labels_res, good_hofs , device = torch.device('cpu') if mem_free < 10 else None) # out of memory issue
        nan2 = gp_space.labels_res.isnan().sum()
        print(f'old and new nans: {nan1} and {nan2}')
        del good_hofs

        '''保存该轮迭代的最终种群、最优个体、迭代日志'''
        joblib.dump(population,f'{_DIR_job}/pop_iter{i_iter}_overall.pkl')
        joblib.dump(halloffame,f'{_DIR_job}/hof_iter{i_iter}_overall.pkl')
        joblib.dump(logbook   ,f'{_DIR_job}/log_iter{i_iter}_overall.pkl')

    gp_space.gp_timer.append_time('AvgVarAnd' , gp_space.gp_timer.acc_timer('varAnd').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('AvgCompile', gp_space.gp_timer.acc_timer('compile').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('AvgEval',    gp_space.gp_timer.acc_timer('eval').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('All' , time.time() - timenow)
    return 
    
def gp_factor_generator(**kwargs):
    '''
    ------------------------ gp factor generator ------------------------
    构成因子生成器，返回一个函数，输入因子表达式则输出历史因子值
    input:
        kwargs:  specific gp parameters, suggestion is to leave it alone
    output:
        wrapper: lambda syntax:factor value
    '''
    gp_space = gp_namespace(gp_parameters(train = False).update(kwargs))
    toolbox = gpHandler.Toolbox(eval_func=evaluate , **gp_space)
        
    def wrapper(syntax , process_key = 'inf_trim_norm'):
        func = toolbox.compile(syntax) #type:ignore
        value = func(*gp_space.gp_values)
        value = process_factor(value , process_key , dim = 1)
        return value
    
    return wrapper
    

def func_str_decompose(func_string):
    # Define the regular expression pattern to extract information
    pattern = {
        r'<code object (.+) at (.+), file (.+), line (\d+)>' : ['function' , (0,) , (2,3) , (1,)] ,
        r'<function (.+) at (.+)>' : ['function' , (0,) , () , ()] ,
        r'<method (.+) of (.+) objects>' : ['method' , (0,) , (1,) , ()] ,
        r'<built-in method (.+)>' : ['built-in-method' , (0,) , () , ()] ,
        r'<fastparquet.cencoding.from_fields>' : ['built-in-method' , () , () , ()] ,
        # r'<(.+)>' : ['other' , (0,) , ()],
    }
    data = None
    for pat , use in pattern.items():
        match = re.match(pat, func_string)
        if match:
            data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , 
                    ','.join(match.group(i+1) for i in use[2]) , ','.join(match.group(i+1) for i in use[3])]
            #try:
            #    data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , ','.join(match.group(i+1) for i in use[2])]
            #except:
            #    print(func_string)
            break
    if data is None: 
        print(func_string)
        data = [''] * 4
    return data

def gp_multifactor(job_id , from_saving = True , weight_scheme = 'ew' , 
                   window_type = 'rolling' , window_len = 480 , weight_decay = 'constant' , 
                   expdecay_halflife = 240 , ir_window = 240):
    assert weight_scheme in ['ew' , 'ic' , 'ir']
    assert window_type   in ['rolling' , 'full'] # 'insample' 
    assert weight_decay  in ['constant' , 'linear' , 'exp']
    if job_id < 0:
        old_job_df = os.listdir('./pop/') if os.path.exists('./pop/') else []
        old_job_id = np.array([id for id in old_job_df if id.isdigit()]).astype(int)
        job_id = max(old_job_id)
    DIR_job = f'{_DIR_pop}/{job_id}'
    good_path = f'{DIR_job}/good_log.csv'
    fac_paths = [f'{DIR_job}/factor/{p}' for p in os.listdir(f'{DIR_job}/factor')]
    gp_factor = torch.Tensor()
    if from_saving:
        gp_filename = gp_filename_converter()

        for path in tqdm(fac_paths , desc='Loading factor parquets'):
            factor_df = read_gp_data(path)
            gp_factor = F.concat_factors(gp_factor , df_to_ts(factor_df , share_memory=False)) 

        df_columns = factor_df.columns.values
        df_index   = factor_df.index.values

        
    else:
        good_log  = pd.read_csv(good_path,index_col=0)
        gp_space = gp_namespace(gp_parameters(False , -1 , True))
        toolbox = gpHandler.Toolbox(eval_func = evaluate , **gp_space)
        population = gp_population(toolbox , **gp_space)
        'ts_corr(TURN, TURN, 7)'

    size  = df_to_ts(read_gp_data(gp_filename('size'),df_columns=df_columns,df_index=df_index) , 'size')
    indus = df_to_ts(read_gp_data(gp_filename('indus'),df_columns=df_columns,df_index=df_index) , 'indus')
    CP    = df_to_ts(read_gp_data(gp_filename('CP'),df_columns=df_columns,df_index=df_index) , 'CP')  
    univ  = ~CP.isnan()
    labels = gp_get_labels(CP , size , indus)
    gp_factor[~univ] = torch.nan

    n_factor = gp_factor.shape[-1]
    if weight_scheme == 'ew':
        multifactor = F.zscore(gp_factor.nanmean(-1),-1)
    else:
        # rankic first
        metric_full = torch.zeros(len(labels),n_factor).to(labels)
        for i_factor in range(n_factor):
            rankic = F.rankic_2d(gp_factor[...,i_factor] , labels , dim = 1 , universe = univ , min_coverage = 0.)
            metric_full[:,i_factor] = rankic
        if weight_scheme == 'ir': metric_full = F.ts_zscore(metric_full , ir_window)
        
        multifactor = torch.zeros_like(gp_factor[...,0])
        if weight_decay == 'constant':
            ts_weight = torch.ones(1,len(multifactor))
        elif weight_decay == 'linear':
            ts_weight = torch.arange(len(multifactor))
        else:
            ts_weight = torch.arange(len(multifactor)).div(expdecay_halflife).pow(2)
        for i in range(len(multifactor)):
            if i < 10: continue
            d = min(window_len , i) if window_type == 'rolling' else i
            f_weight = ts_weight[:d].reshape(1,-1) @ metric_full[i-d:i]
            factor_of_i = gp_factor[i] @ f_weight.reshape(-1,1)
            multifactor[i] = factor_of_i.flatten()
    
    multifactor = pd.DataFrame(multifactor.cpu().numpy() , columns = df_columns , index = df_index)
    return multifactor

            




# %%
def main(test_code = None , job_id = None , profiling = False , profiler_out = 'cprofile.csv'):
    """
    ------------------------ gp main process ------------------------
    input:
        test_code: if only to test code validity
        job_id:    when test_code is not True, determines _DIR_job = f'{_DIR_pop}/{job_id}'   
    output:
        None
    process:
        for i_iter in n_iter:
            outer_loop()
    """
    if test_code and profiling:
        with cProfile.Profile() as pfr:
            main(test_code , job_id , False)

        profiler = pd.DataFrame(pfr.getstats(), #type:ignore
                                columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime' , 'caller'])
        profiler = profiler.iloc[:,:-1]
        profiler.tottime = profiler.tottime.round(4)
        profiler.cumtime = profiler.cumtime.round(4)
        profiler.func    = profiler.func.astype(str)
        func_df = pd.DataFrame([func_str_decompose(s) for s in profiler.func] , 
                                columns = ['type' , 'name' , 'where' , 'memory'])
        profiler = pd.concat([func_df , profiler.iloc[:,1:]],axis=1).sort_values('cumtime',ascending=False)
        if isinstance(profiler_out , str): profiler.to_csv(profiler_out)
        # df[df.func.astype(str).str.find('gp_math_func.py') > 0][:20].to_csv('result.csv')
        
        return profiler

    time0 = time.time()
    gp_space = gp_namespace(gp_parameters(test_code , job_id))
    for i_iter in range(gp_space.n_iter):
        print(f' ------------------------------- Outer Loop {i_iter} -------------------------------')
        outer_loop(i_iter , gp_space)

    hours, secs = divmod(time.time() - time0, 3600)
    print(f'------------------------------- Total Time Cost :{hours:.0f} hours {secs/60:.1f} -------------------------------')
    gp_space.gp_timer.save_to_csv(f'{_DIR_job}/saved_times.csv' , print_out = True)
    gp_space.memory_manager.print_memeory_record()

if __name__ == '__main__':
    main()
