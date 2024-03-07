# %%
import pandas as pd
import numpy as np
import os , sys , copy , tqdm , shutil , gc , re , traceback
import torch
import array , random , json , operator , time, platform , joblib
from argparse import ArgumentParser , Namespace
from tqdm import tqdm
import cProfile

from deap import base , creator , tools , gp
from deap.algorithms import varAnd
from torch.multiprocessing import Pool

import gp_math_func as MF
import gp_factor_func as FF
from gp_utils import gpHandler , gpTimer , gpContainer , gpFileManager , MemoryManager , gpEliteGroup
from gp_affiliates import Profiler , EmptyTM

# %%
# ------------------------ environment setting ------------------------

# np.seterr(over='raise')

_plat      = platform.system().lower()                                    # Windows or Linux
_device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda or cpu
_test_code = True or (_device == torch.device('cpu'))                     # 是否只是测试代码有无bug,默认False
_noWith    = True                                                        # 是否取消所有计时器,默认False,有计时器时报错会出问题
_DIR_data  = './data/features/parquet'                                    # input路径1,原始因子所在路径
_DIR_pack  = './data/package'                                             # input路径2,加载原始因子后保存pt文件加速存储
_DIR_pop   = './pop'                                                      # output路径,即保存因子库、因子值、因子表达式的路径
_DIR_job   = f'{_DIR_pop}/bendi'                                          # 测试代码时的job path 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                               # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099

parser = ArgumentParser()
parser.add_argument("--job_id", type=int, default=-1)
parser.add_argument("--poolnm", type=int, default=1)
args, _ = parser.parse_known_args()

assert args.poolnm == 1 , args.poolnm # 若并行,通信成本过高,效率提升不大
# multiprocessing method, 单机多进程设置(实际未启用),参考https://zhuanlan.zhihu.com/p/600801803
torch.multiprocessing.set_start_method('spawn' if _plat == 'windows' else 'forkserver', force=True) 

def text_manager(object):
    if _noWith:
        return EmptyTM(object)
    else:
        return object

def gp_job_dir(job_id = None , train = True , continuation = False , test_code = False):
    # directory setting and making
    
    if job_id is None: job_id = args.job_id
    if test_code:
        job_dir = _DIR_job
    else:
        if job_id < 0:
            old_job_df = os.listdir('./pop/') if os.path.exists('./pop/') else []
            old_job_id = np.array([id for id in old_job_df if id.isdigit()]).astype(int)
            if train:
                job_id = np.setdiff1d(np.arange(max(old_job_id) + 2) , old_job_id).min() if len(old_job_id) else 0
            else:
                job_id = old_job_id
        job_dir = f'{_DIR_pop}/{job_id}'
    print(f'**Job Directory is : "{job_dir}"')
    if train:
        if os.path.exists(job_dir) and not continuation: 
            if not test_code and not input(f'Path "{job_dir}" exists , press "yes" to confirm Deletion:')[0].lower() == 'y':
                raise Exception(f'Deletion Denied!')
            shutil.rmtree(job_dir)
            print(f'  --> Start New Training in "{job_dir}"')
        elif os.path.exists(job_dir) and not input(f'Path "{job_dir}" exists , press "yes" to confirm Continuation:')[0].lower() == 'y':
            raise Exception(f'Continuation Denied!')
        elif not os.path.exists(job_dir) and continuation:
            raise Exception(f'No existing "{job_dir}" for Continuation!')
        else:
            print(f'  --> Continue Training in "{job_dir}"')
    return job_dir

def gp_parameters(job_id = None , train = True , continuation = False , test_code = False , **kwargs):
    '''
    ------------------------ gp parameters ------------------------
    主要遗传规划参数初始化
    注：为方便跑多组实验,需设置job_id参数,设置方式为: python xxx.py --job_id 123456
    input:
        test_code: if only to test code validity
        job_id:    when test_code is not True, determines f'{_DIR_pop}/{job_id}' 
        train:     if True, will first check dirs and device  
    output:
        gp_params: dict that includes all gp parameters
    '''
    job_dir = gp_job_dir(job_id , train , continuation , test_code)
    if train and _device == torch.device('cpu'):
        print('**Cuda not available')
    elif train:
        print('**Device name:', torch.cuda.get_device_name(0), ', Available:' ,torch.cuda.is_available())
        
    '''
    job_dir       工作目录
    test_code     是不是仅仅作为代码测试,若是会有一个小很多的参数组合覆盖本参数
    gp_fac_list   作为GP输入时,标准化因子的数量
    gp_raw_list   作为GP输入时,原始指标的数量
    slice_date:   修改数据切片区间,前两个为样本内的起止点,后两个为样本外的起止点均需要是交易日
    device:       用cuda还是cpu计算
    verbose:      训练过程是否输出细节信息
    pool_num:     并行任务数量,建议设为1,即不并行,使用单显卡单进程运行
    pop_num:      种群数量,初始化时生成多少个备选公式
    hof_num:      精英数量,一般精英数量设为种群数量的1/6左右即可
    n_iter:       [大循环]的迭代次数,每次迭代重新开始一次遗传规划、重新创立全新的种群,以上一轮的残差收益率作为优化目标
    ir_floor:     [大循环]中因子入库所需的最低rankIR值,低于此值的因子不入库
    corr_cap:     [大循环]中新因子与老因子的最高相关系数,相关系数绝对值高于此值的因子不入库
    n_gen:        [小循环]的迭代次数,即每次遗传规划进行几轮繁衍进化
    max_depth:    [小循环]中个体算子树的最大深度,即因子表达式的最大复杂度
    cxpb:         [小循环]中交叉概率,即两个个体之间进行交叉的概率
    mutpb:        [小循环]中变异概率,即个体进行突变变异的概率
    '''

    gp_fac_list = ['cp', 'turn', 'vol', 'amt', 'op', 'hp', 'lp', 'vp', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']# in lower case
    gp_raw_list = [v.upper() for v in gp_fac_list] + ['RTN'] # in upper case
    slice_date  = ['2010-01-04', '2021-12-31', '2022-01-04', '2099-12-31']

    gp_params = gpContainer(
        job_dir = job_dir ,           # the main directory
        test_code = test_code  ,      # just to check code, will save parquet in this case
        gp_fac_list = gp_fac_list ,   # gp intial factor list 
        gp_raw_list = gp_raw_list ,   # gp intial raw data list
        slice_date  = slice_date ,    # must be trade date
        device = _device ,            # training device, cuda or cpu
        verbose = False ,             # if show some text
        pool_num = args.poolnm ,      # multiprocessing pool number
        pop_num= 100 ,               # [parameter] population number
        hof_num= 500 ,                # [parameter] halloffame number
        n_iter =  5 ,                 # [outer loop] loop number
        ir_floor = 3.0 ,              # [outer loop] rankir threshold
        corr_cap = 0.6 ,              # [outer loop] cap of correlation with existing factors
        n_gen = 6  ,                  # [inner loop] generation number
        max_depth = 5 ,               # [inner loop] max tree depth of gp
        surv_rate = 0.8 ,             # [inner loop] survive rate last offspring can go down to next generation
        cxpb = 0.35 ,                 # [inner loop] crossover probability
        mutpb = 0.25 ,                # [inner loop] mutation probability
    )
    if test_code:
        # when test code, change some parameters
        gp_params.update(gp_fac_list = gp_fac_list[:2] , gp_raw_list= gp_raw_list[:2] ,
                         verbose = True , pop_num = 10 , hof_num = 5 , n_gen = 2 , n_iter = 1 , 
                         max_depth = 2 , ir_floor = 1.5 , corr_cap = 0.7 , 
                         slice_date = ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29'])
    gp_params.update(**kwargs)
    gp_params.apply('slice_date' , lambda x:pd.to_datetime(x).values)
    MF.invalid = MF.invalid.to(gp_params.get('device')) # should test if speed up
    return gp_params
    
# %%
def gp_namespace(gp_params):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    基于遗传规划的参数字典,读取各类主要数据,并放在同一字典中传回
    input:
        gp_params: gpContainer of all gp_parameters
        gp_timer:  gpTimer to record time cost
    output:
        gp_space:  gpContainer that includes gp parameters, gp datas and gp arguements, and various other datas
    '''
    file_manager = gpFileManager(gp_params.get('job_dir')) # gpFileManager managing loading and dumping
    gp_timer = gpTimer(True)                               # gpTimer to record time cost

    gp_space = gp_params.copy().update(file_manager = file_manager , gp_timer = gp_timer , gp_values = [] , df_columns = None)
    with text_manager(gp_space.gp_timer('Data' , df_cols = False , print_str= '**Load Data')):
        gp_space.gp_args = gp_space.gp_fac_list + gp_space.gp_raw_list
        gp_space.n_args  = (len(gp_space.gp_fac_list) , len(gp_space.gp_raw_list))
        
        package_path = f'{_DIR_pack}/gp_data_package' + '_test' * gp_space.test_code + '.pt'
        package_require = ['gp_args' , 'gp_values' , 'size' , 'indus' , 'labels_raw' , 'df_index' , 'df_columns' , 'universe']

        load_finished = False
        package_data = torch.load(package_path) if os.path.exists(package_path) else {}
        if not np.isin(package_require , list(package_data.keys())).all() or not np.isin(gp_space.gp_args , package_data['gp_args']).all():
            if gp_space.verbose: print(f'  --> Exists "{package_path}" but Lack Required Data!')
        else:
            assert np.isin(package_require , list(package_data.keys())).all() , np.setdiff1d(package_require , list(package_data.keys()))
            assert np.isin(gp_space.gp_args , package_data['gp_args']).all() , np.setdiff1d(gp_space.gp_args , package_data['gp_args'])
            assert package_data['df_index'] is not None

            if gp_space.verbose: print(f'  --> Directly load "{package_path}"')
            for gp_key in gp_space.gp_args:
                gp_val = package_data['gp_values'][package_data['gp_args'].index(gp_key)]
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_values.append(gp_val)

            for gp_key in ['size' , 'indus' , 'labels_raw' , 'universe']: 
                gp_val = package_data[gp_key]
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.set(gp_key , gp_val)

            for gp_key in ['df_index' , 'df_columns']: 
                gp_val = package_data[gp_key]
                gp_space.set(gp_key , gp_val)

            load_finished = True

        if not load_finished:
            if gp_space.verbose: print(f'  --> Load from Parquet Files:')
            gp_filename = gp_filename_converter()
            nrowchar = 0
            for i , gp_key in enumerate(gp_space.gp_args):
                if gp_space.verbose and nrowchar == 0: print('  --> ' , end='')
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.slice_date,gp_space.df_columns)
                if i == 0: gp_space.update(df_columns = gp_val.columns.values , df_index = gp_val.index.values)
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_values.append(gp_val)
                
                if gp_space.verbose:
                    print(gp_key , end=',')
                    nrowchar += len(gp_key) + 1
                    if nrowchar >= 100 or i == len(gp_space.gp_args):
                        print()
                        nrowchar = 0

            for gp_key in ['size' , 'indus']: 
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.slice_date,gp_space.df_columns)
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.set(gp_key , gp_val)

            if 'CP' in gp_space.gp_args:
                CP = gp_space.gp_values[gp_space.gp_args.index('CP')]      
            else:
                df2ts(read_gp_data(gp_filename('CP'),gp_space.slice_date,gp_space.df_columns) , 'CP' , gp_space.device)    
            gp_space.universe   = ~CP.isnan() # type:ignore
            gp_space.labels_raw = gp_get_labels(CP , gp_space.size , gp_space.indus)
            os.makedirs(_DIR_pack , exist_ok=True)
            torch.save(gp_space.subset(package_require , require = True) , package_path)

    gp_space.labels_res = copy.deepcopy(gp_space.labels_raw)
    if gp_space.verbose: print(f'  --> {len(gp_space.gp_fac_list)} factors, {len(gp_space.gp_raw_list)} raw data loaded!')
    file_manager.save_states({'params':gp_params.__dict__,'df_axis':gp_space.subset(['df_index' , 'df_columns'])} , i_iter = 0) # useful to assert same index as package data

    gp_space.insample  = (gp_space.df_index >= gp_space.slice_date[0]) * (gp_space.df_index <= gp_space.slice_date[1])
    gp_space.outsample = (gp_space.df_index >= gp_space.slice_date[2]) * (gp_space.df_index <= gp_space.slice_date[3])
    gp_space.mem_manager = MemoryManager(0)
    return gp_space

def gp_get_labels(CP = None , neutral_factor = None , neutral_group = None , nday = 10 , delay = 1 , 
                  slice_date = None, df_columns = None , device = None):
    if CP is None:
        CP = df2ts(read_gp_data(gp_filename_converter()('CP'),slice_date,df_columns) , 'CP' , device)    
    labels = MF.ts_delay(MF.pctchg(CP, nday) , -nday-delay)  # t+1至t+11的收益率
    neutral_x = MF.neutralize_xdata_2d(neutral_factor, neutral_group)
    return MF.neutralize_2d(labels, neutral_x , inplace=True)  # 市值行业中性化

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
    读取单个原始因子文件并转化成tensor,额外返回df表格的行列字典
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

def df2ts(x , gp_key = '' , device = None , share_memory = True):
    # additional treatment based by gp_key
    if isinstance(x , pd.DataFrame): x = torch.FloatTensor(x.values)
    if gp_key.startswith('dp'): # dividend factor , nan means 0
        x = x.nan_to_num()
    if isinstance(x , torch.Tensor):
        if device is not None: x = x.to(device)
        if share_memory: x.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return x

# %%
def gp_syntax2value(compiler , individual , gp_values , process_stream = 'inf' , gp_timer = gpTimer() , **kwargs):
    '''
    ------------------------ calculate individual syntax factor value ------------------------
    根据迭代出的因子表达式,计算因子值 , 基于函数表达式构建函数：deap.base.Toolbox().compile
    input:
        compiler:     compiler function to realize syntax computation, i.e. return factor function of given syntax
        individual:   individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        gp_values:    initial population factor values
        timer:        record compile time
    output:
        factor_value: 2d tensor
    '''
    #print(individual)
    with text_manager(gp_timer.acc_timer('compile')): 
        func = compiler(individual)
    with text_manager(gp_timer.acc_timer('eval')):
        factor_value = func(*gp_values)
        factor_value = FF.process_factor(factor_value , process_stream , dim = 1)
    return factor_value

# %%
def evaluate(individual, pool_skuname, compiler , gp_values , labels_raw , labels_res, universe, insample , outsample , 
             gp_timer = gpTimer(), file_manager = gpFileManager() , memory_manager = MemoryManager() ,
             const_annual = 24 , min_coverage = 0.5 , i_iter = -1 , **kwargs):
    '''
    ------------------------ evaluate individual syntax fitness ------------------------
    从因子表达式起步,生成因子并计算适应度
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
            abs_rankir: (abs(insample_res), ) # !! Fitness definition 
            rankir:     (insample_res, outsample_res, insample_raw, outsample_raw)
        )
    '''
    file_manager.update_sku(individual, pool_skuname)
    
    # compile syntax and calculate factors
    factor_value = gp_syntax2value(compiler , individual , gp_values , 'inf' , gp_timer)
    # memory_manager.check('factor')

    # first return must be list or tuple, since fitness can be weighted , 
    # Assigned values have not the same length than fitness weights
    rankir_list = torch.zeros(4)
    if is_invalid := MF.is_invalid(factor_value): 
        return not is_invalid , rankir_list # if_valid , ir's

    for i , labels in enumerate([labels_res , labels_raw]):
        rankic_full = MF.rankic_2d(factor_value , labels , dim = 1 , universe = universe , min_coverage = min_coverage)
        if is_invalid := MF.is_invalid(rankic_full): continue
        for j , sample in enumerate([insample , outsample]):
            rankic = rankic_full[sample]
            if rankic.isnan().sum() < 0.5 * len(rankic): # if too many nan rank_ic (due to low coverage)
                rankic_avg  = rankic.nanmean()
                rankic_std  = (rankic - rankic_avg).square().nanmean().sqrt() 
                rankir_list[i * 2 + j] = (rankic_avg / (rankic_std + 1e-6) * np.sqrt(const_annual)) #.cpu() # 年化 ir
    # memory_manager.check('rankic')
    return not is_invalid , rankir_list.nan_to_num(0,0,0)

def evaluate_pop(population , toolbox , i_iter = 0, i_gen = 0, pool_num = 1 , desc = 'Evolve Generation' , **kwargs):
    changed_pop = [ind for ind in population if not ind.fitness.valid]
    pool_skunames = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(changed_pop))] # 'pool_skuname' arg for evaluate
    desc = f'  --> {desc} {str(i_gen)}'

    if pool_num > 1:
        pool = Pool(pool_num)
        #fitnesses = pool.starmap(toolbox.evaluate, zip(survivors, pool_list), chunksize=1)
        fitnesses = list(tqdm(pool.imap(toolbox.evaluate, zip(changed_pop, pool_skunames)), total=len(changed_pop), desc=desc)) #type:ignore
        pool.close()
        pool.join()
        # pool.clear()
    else:
        fitnesses = list(tqdm(toolbox.map(toolbox.evaluate, changed_pop, pool_skunames), total=len(changed_pop), desc=desc))
        #fitnesses = []
        #for ind , sku in tqdm(zip(changed_pop, pool_skunames) , desc=desc , total=len(changed_pop)):
        #    fitnesses.append(toolbox.evaluate(ind , sku))

    # Evaluate the individuals with an invalid fitness(对于发生改变的个体,重新评估abs(IR)值)
    for ind, fit in zip(changed_pop, fitnesses):
        ind.if_valid , ind.rankirs = fit[0] , fit[1]
        ind.fitness.values = (abs(fit[1][0]),)

    assert all([ind for ind in population if not ind.fitness.valid])
    return population

# %%
def gp_population(toolbox , pop_num , max_round = 100 , last_gen = [], omitalways = [] , **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    初始化种群
    input:
        gp_args:    initial gp factor names
        i_iter:     i of outer loop
        max_round:  max iterations to approching 99% of pop_num
        pop_num:    population number
        last_gen:   starting population
        omitalways: all individuals with invalid return or goes in the the hof once
    output:
        toolbox:    toolbox that contains all gp utils
        population: initial population of syntax
    '''
    if last_gen: 
        population = toolbox.indpop_prune(last_gen)
        population = toolbox.deduplicate(population)
    else:
        population = []

    for _ in range(max_round):
        new_comer  = toolbox.population(n = int(pop_num * 1.2) - len(population))
        new_comer  = toolbox.indpop_prune(new_comer)
        new_comer  = toolbox.deduplicate(new_comer , exclude = population + omitalways) 
        population = population + new_comer[:pop_num - len(population)]
        if len(population) >= 0.99 * pop_num: break
    
    print([str(ind) for ind in population[:20]])
    population = toolbox.indpop2syxpop(population)
    print([str(ind) for ind in population[:20]])
    return population

# %%
def gp_evolution(toolbox , i_iter, pop_num , file_manager , pool_num=1,
                 n_gen=5,cxpb=0.35,mutpb=0.25,hof_num=10, surv_rate=0.8,omit_ir_floor=0.2,
                 start_gen=0,gp_timer=gpTimer(),verbose=__debug__,stats=None,**kwargs):  
    """
    ------------------------ Evolutionary Algorithm simple ------------------------
    变异/进化[小循环],从初始种群起步计算适应度并变异,重复n_gen次
    input:
        toolbox:    toolbox that contains all gp utils
        i_iter:     i of outer loop
        pop_num:    population number
        pool_num:   multiprocessing pool number
        n_gen:      [inner loop] generation number
        cxpb:       [inner loop] crossover probability
        mutpb:      [inner loop] mutation probability
        hof_num:    halloffame number
        surv_rate:  how many last generation survivors can go down to next generation
        start_gen:  which gen to start, if None start a new
        gp_timer:   gpTimer to record time cost
    output:
        population: updated population of syntax
        halloffame: container of individuals with best fitness (no more than hof_num)
        omitalways: all individuals with invalid return or goes in the the hof once

    ------------------------ basic code structure ------------------------
    evaluate(population)     # 对随机生成的初代种群评估IR值
    for g in range(n_gen)):
        evaluate(population)   # 对新种群评估IR值
        population = select(population, len(population))    # 选取abs(IR)值较高的个体,以产生后代
        offspring = varAnd(population, toolbox, cxpb, mutpb)   # 交叉、变异
        population = offspring    # 更新种群
    """
    population , halloffame , omitalways = file_manager.load_generation(i_iter , start_gen-1 , hof_num = hof_num , stats = stats)

    for i_gen in range(start_gen, n_gen):
        if verbose and i_gen > 0: print(f'  --> Survive {len(population)} Offsprings, try Populating to {pop_num} ones')
        population = gp_population(toolbox , pop_num , last_gen = population , omitalways = omitalways)
        if verbose and i_gen == 0: print(f'**A Population({len(population)}) has been Initialized')

        # Evaluate the new population
        population = toolbox.evaluate_pop(population , i_iter = i_iter, i_gen = i_gen , pool_num = pool_num)

        # check survivors
        survivors   = [ind for ind in population if ind.if_valid]
        omitalways += [gpHandler.ind2str(ind) for ind in population if not ind.if_valid or ind.fitness.values[0].item() <= omit_ir_floor]

        # Update HallofFame with survivors 
        halloffame.update(survivors)
        omitalways += [gpHandler.ind2str(ind) for ind in halloffame]

        # Selection of population to pass to next generation,based on abs(IR) , consider surv_rate
        offspring = toolbox.select(population , min(int(surv_rate * pop_num) , len(population)))

        # Variation offsprings
        with text_manager(gp_timer.acc_timer('varAnd')):
            offspring = toolbox.syxpop2indpop(offspring)
            offspring = varAnd(offspring, toolbox, cxpb , mutpb) # varAnd means variation part (crossover and mutation)

        # Dump population , halloffame , omitalways , logbooks(record **kwargs) of this generation
        file_manager.dump_generation(population, halloffame, omitalways , i_iter , i_gen , **(stats.compile(survivors) if stats else {}))

    print(f'**A HallofFame({len(halloffame)}) has been ' + ('Loaded' if start_gen >= n_gen else 'Evolutionized'))
    file_manager.dump_generation(population, halloffame, omitalways , i_iter = i_iter , i_gen = -1)
    return population, halloffame, omitalways

# %%
def gp_selection(toolbox, halloffame, i_iter , gp_values , ir_floor=2.5,corr_cap=0.7,insample = None,
                 file_manager=gpFileManager(),gp_timer=gpTimer(),mem_manager=MemoryManager(),device=None,
                 test_code=False,verbose=__debug__,**kwargs):
    """
    ------------------------ gp halloffame evaluation ------------------------
    筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英
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
        hof_elites:    elite hof values who pass the criterions
    """
    
    elite_log , hof_log = file_manager.load_states(['elitelog' , 'hoflog'] , i_iter = i_iter)
    #i_elite = len(elite_log)
    # hof_elites = [] # MF.invalid
    hof_elites = gpEliteGroup(start_i_elite = len(elite_log) , device=device).assign_logs(hof_log = hof_log , elite_log = elite_log)

    new_log  = pd.DataFrame([[i_iter,-1,gpHandler.ind2str(ind),ind.if_valid,False,0.,*ind.rankirs.cpu().numpy()] for ind in halloffame] , 
                            columns = ['i_iter','i_elite','sytax','valid','elite','max_corr','rankir_in_res','rankir_out_res','rankir_in','rankir_out'])
    new_log.elite = new_log.valid & (new_log.rankir_in_res.abs() > ir_floor) & (new_log.rankir_out_res != 0.)
    
    print(f'**HallofFame({len(halloffame)}) Contains {new_log.elite.sum()} Promising Candidates with RankIR >= {ir_floor:.2f}')
    if new_log.elite.sum() <= 0.1 * len(halloffame):
        # Failure of finding promising offspring , check if code has bug
        print(f'  --> Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ')
        print(f'  --> Valid Hof({new_log.valid.sum()}), insample max ir({new_log.rankir_in_res.abs().max():.4f})')

    for i , hof in enumerate(halloffame):
        if not new_log.loc[i,'elite']: continue

        # 根据迭代出的因子表达式,计算因子值, 错误则进入下一循环
        factor_value = gp_syntax2value(toolbox.compile , hof , gp_values , 'inf' , gp_timer)
        mem_manager.check('factor')
        
        new_log.loc[i,'elite'] = not MF.is_invalid(factor_value)
        if not new_log.loc[i,'elite']: continue

        # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
        corr_values , exit_state = hof_elites.max_corr_with_me(factor_value, abs_corr_cap=corr_cap, dim=1,dim_valids=(insample,None))
        new_log.loc[i,'max_corr'] = round(corr_values[corr_values.abs().argmax()].item() , 4)
        new_log.loc[i,'elite'] = not exit_state
        mem_manager.check('corr')

        if not new_log.loc[i,'elite']: continue

        # 通过检验,加入因子库
        new_log.loc[i,'i_elite'] = hof_elites.i_elite
        hof_elites.append(new_log.sytax[i] , factor_value , IR = new_log.rankir_in_res[i] , Corr = new_log.max_corr[i] , starter=f'  --> Hof{i:_>3d}/')
        if False and test_code: file_manager.save_state(factor_value , 'parquet' , i_iter , i_elite = hof_elites.i_elite)

        mem_manager.check(showoff = verbose and test_code, starter = '  --> ')

    hof_elites.update_logs(new_log)
    file_manager.save_states({'elitelog' : hof_elites.elite_log.round(6) , 'hoflog' : hof_elites.hof_log.round(6)} , i_iter = i_iter)
    hof_elites = hof_elites.compile_elite_tensor(device=device).elite_tensor
    
    print(f'**An EliteGroup({hof_elites.shape[-1]}) has been Selected')
    
    if verbose: print(f'  --> Memories of [gp_values , hof_elites , others] take {MemoryManager.object_memory([gp_values , kwargs , hof_elites]):.2f}G')
    
    return hof_elites

# %%
def outer_loop(i_iter , gp_space , start_gen = 0):
    """
    ------------------------ gp outer loop ------------------------
    一次[大循环]的主程序,初始化种群、变异、筛选、更新残差labels
    input:
        i_iter:   i of outer loop
        gp_space:  dict that includes gp parameters, gp datas and gp arguements, and various other datas
    """
    timenow = time.time()
    gp_space.i_iter = i_iter

    '''初始化遗传规划Toolbox'''
    with text_manager(gp_space.gp_timer('Setting' , print_str = f'**Initialize GP Toolbox')) as ptimer:
        toolbox = gpHandler.Toolbox(eval_func = evaluate , eval_pop = evaluate_pop , **gp_space)
        gp_space.file_manager.update_toolbox(toolbox)

    '''进行进化与变异,生成种群、精英和先祖列表'''
    with text_manager(gp_space.gp_timer('Evolution' , print_str = f'**{gp_space.n_gen - start_gen} Generations of Evolution' , memory_check = True)) as ptimer:
        population, halloffame, omitalways = gp_evolution(toolbox , start_gen = start_gen , **gp_space)  #, start_gen=6   #algorithms.eaSimple

    '''衡量精英,筛选出符合所有要求的精英中的精英'''
    with text_manager(gp_space.gp_timer('Selection' , print_str = f'**Selection of HallofFame' , memory_check = True)) as ptimer:
        hof_elites = gp_selection(toolbox , halloffame, **gp_space)

    '''更新残差标签,保存循环状态'''
    with text_manager(gp_space.gp_timer('Residual' , print_str = f'**Update Residual Labels' , memory_check = True)) as ptimer:
        gp_space.file_manager.save_state(gp_space.labels_res , 'labels_res' , i_iter) # useful to form multifactor
        gp_space.labels_res = MF.neutralize_2d(gp_space.labels_res, hof_elites , inplace = True) # out of memory issue
        del hof_elites
        
    gp_space.gp_timer.append_time('AvgVarAnd' , gp_space.gp_timer.acc_timer('varAnd').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('AvgCompile', gp_space.gp_timer.acc_timer('compile').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('AvgEval',    gp_space.gp_timer.acc_timer('eval').avgtime(pop_out = True))
    gp_space.gp_timer.append_time('All' , time.time() - timenow)
    return 
    
def gp_factor_generator(**kwargs):
    '''
    ------------------------ gp factor generator ------------------------
    构成因子生成器,返回输入因子表达式则输出历史因子值的函数
    input:
        kwargs:  specific gp parameters, suggestion is to leave it alone
    output:
        wrapper: lambda syntax:factor value
    '''
    gp_params = gp_parameters(train = False , **kwargs)
    gp_space  = gp_namespace(gp_params)
    toolbox   = gpHandler.Toolbox(eval_func=evaluate , eval_pop=evaluate_pop , **gp_space)
        
    def wrapper(syntax , process_key = 'inf_trim_norm'):
        func = toolbox.compile(syntax) #type:ignore
        value = func(*gp_space.gp_values)
        value = FF.process_factor(value , process_key , dim = 1)
        return value
    
    return wrapper

def gp_multifactor(job_id , from_saving = True , weight_scheme = 'ew' , 
                   window_type = 'rolling' , window_len = 480 , weight_decay = 'constant' , 
                   exp_halflife = 240 , ir_window = 240):
    assert weight_scheme in ['ew' , 'ic' , 'ir']
    assert window_type   in ['rolling' , 'full'] # 'insample' 
    assert weight_decay  in ['constant' , 'linear' , 'exp']
    if job_id < 0:
        old_job_df = os.listdir('./pop/') if os.path.exists('./pop/') else []
        old_job_id = np.array([id for id in old_job_df if id.isdigit()]).astype(int)
        job_id = max(old_job_id)
    DIR_job = f'{_DIR_pop}/{job_id}'
    elite_path = f'{DIR_job}/elite_log.csv'
    fac_paths = [f'{DIR_job}/factor/{p}' for p in os.listdir(f'{DIR_job}/factor')]
    gp_factor = torch.Tensor()
    if from_saving:
        gp_filename = gp_filename_converter()

        for path in tqdm(fac_paths , desc='Loading factor parquets'):
            factor_df = read_gp_data(path)
            gp_factor = MF.concat_factors(gp_factor , df2ts(factor_df , share_memory=False)) 

        df_columns = factor_df.columns.values
        df_index   = factor_df.index.values

        
    else:
        elite_log  = pd.read_csv(elite_path,index_col=0)
        gp_space = gp_namespace(gp_parameters(-1 , train = False , test_code= True))
        toolbox = gpHandler.Toolbox(eval_func = evaluate , **gp_space)
        population = gp_population(toolbox , **gp_space)
        'ts_corr(TURN, TURN, 7)'

    size  = df2ts(read_gp_data(gp_filename('size'),df_columns=df_columns,df_index=df_index) , 'size')
    indus = df2ts(read_gp_data(gp_filename('indus'),df_columns=df_columns,df_index=df_index) , 'indus')
    CP    = df2ts(read_gp_data(gp_filename('CP'),df_columns=df_columns,df_index=df_index) , 'CP')  
    univ  = ~CP.isnan()
    labels = gp_get_labels(CP , size , indus)
    gp_factor[~univ] = torch.nan

    n_factor = gp_factor.shape[-1]
    if weight_scheme == 'ew':
        multifactor = MF.zscore(gp_factor.nanmean(-1),-1)
    else:
        # rankic first
        metric_full = torch.zeros(len(labels),n_factor).to(labels)
        for i_factor in range(n_factor):
            rankic = MF.rankic_2d(gp_factor[...,i_factor] , labels , dim = 1 , universe = univ , min_coverage = 0.)
            metric_full[:,i_factor] = rankic
        if weight_scheme == 'ir': metric_full = MF.ts_zscore(metric_full , ir_window)
        
        multifactor = torch.zeros_like(gp_factor[...,0])
        ts_weight = FF.decay_weight(weight_decay , len(multifactor) , exp_halflife=exp_halflife)
        for i in range(len(multifactor)):
            if i < 10: continue
            d = min(window_len , i) if window_type == 'rolling' else i
            f_weight = ts_weight[:d].reshape(1,-1) @ metric_full[i-d:i]
            factor_of_i = gp_factor[i] @ f_weight.reshape(-1,1)
            multifactor[i] = factor_of_i.flatten()
    
    multifactor = pd.DataFrame(multifactor.cpu().numpy() , columns = df_columns , index = df_index)
    return multifactor

# %%
def main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False , **kwargs):
    """
    ------------------------ gp main process ------------------------
    训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
    input:
        job_id:    when test_code is not True, determines job_dir = f'{_DIR_pop}/{job_id}'   
        start_iter , start_gen: when to start, any of them has positive value means continue training
        noWith:    to shutdown all timers (with xxx expression)
    output:
        pfr:       profiler to record time cost of each function (only available in test_code model)
    """
    global _noWith
    _noWith = noWith
    with text_manager(Profiler(doso = test_code)) as pfr:
        time0 = time.time()
        gp_params = gp_parameters(job_id , continuation = start_iter>0 or start_gen>0 , test_code = test_code , **kwargs)
        gp_space = gp_namespace(gp_params)
        if start_iter == 0:
            gp_space.labels_res = copy.deepcopy(gp_space.labels_raw)
        else:
            gp_space.labels_res = gp_space.file_manager.load_state('labels_res' , start_iter - 1).to(gp_space.device)

        for i_iter in range(start_iter , gp_space.n_iter):
            print('=' * 20 + f' Iteration {i_iter} start from Generation {start_gen * (i_iter == start_iter)} ' + '=' * 20)
            outer_loop(i_iter , gp_space , start_gen = start_gen * (i_iter == start_iter))

    hours, secs = divmod(time.time() - time0, 3600)
    print('=' * 20 + f' Total Time Cost :{hours:.0f} hours {secs/60:.1f} ' + '=' * 20)
    gp_space.file_manager.save_state(gp_space.gp_timer.time_table(showoff=True) , 'runtime' , 0)
    gp_space.mem_manager.print_memeory_record()
    pfr.get_df(output = kwargs.get('profiler_out' , 'cprofile.csv')) #type:ignore
    return pfr

if __name__ == '__main__':
    main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False)
