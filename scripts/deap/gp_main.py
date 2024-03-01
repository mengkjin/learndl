# %%
import pandas as pd
import numpy as np
import os , sys , copy , tqdm , shutil
import torch
import argparse
import array , random , json , operator , time, platform , joblib
from tqdm import tqdm

from deap import base , creator , tools , gp
from deap.algorithms import varAnd
from torch.multiprocessing import Pool

import gp_math_func as F
from gp_utils import gpHandler , gpTimer

# %%
# ------------------------ environment setting ------------------------

np.seterr(over='raise')

_plat      = platform.system().lower()                                    # Windows or Linux
_device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda or cpu
_test_code = True or (_device == torch.device('cpu'))                     # True if just to test code is valid
_DIR_data  = './data/features/parquet'                                    # input路径，即原始因子所在路径。
_DIR_pack  = './data/package'                                             # input路径2, 加载原始因子后保存pt文件加速存储。
_DIR_pop   = './pop'                                                      # output路径，即保存因子库、因子值、因子表达式的路径。
_DIR_job   = f'{_DIR_pop}/bendi'                                          # 测试代码时的job path 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                               # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099

parser = argparse.ArgumentParser()
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
    if test_code is None: test_code = _test_code and train
    if train:
        if job_id is None: job_id = args.job_id
        if test_code:
            assert _DIR_job == f'{_DIR_pop}/bendi'
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
            print('Cuda not available')
        else:
            print('Device name:', torch.cuda.get_device_name(0), ', Available:' ,torch.cuda.is_available())
        print(f'Pop directory is : "{_DIR_job}"')

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

    gp_fac_list = ['close', 'turn', 'volume', 'amt', 'open', 'high', 'low', 'vwap', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']
    gp_raw_list = [f'{_d}_raw' for _d in gp_fac_list] + ['rtn_raw']
    slice_date  = ['2010-01-04', '2021-12-31', '2022-01-04', '2099-12-31']
    slice_date  = pd.to_datetime(slice_date).values
    gp_params = {
        'test_code'   : test_code ,     # just to check code, 
        'gp_fac_list' : gp_fac_list ,   # gp intial factor list 
        'gp_raw_list' : gp_raw_list ,   # gp intial raw data list
        'slice_date'  : slice_date ,    # must be trade date
        'device' : _device ,            # training device, cuda or cpu
        'verbose' : False ,             # if show some text
        'pool_num' : args.poolnm ,      # multiprocessing pool number
        'pop_num': 3000 ,               # [parameter] population number
        'hof_num': 500 ,                # [parameter] halloffame number
        'n_iter' :  5 ,                 # [outer loop] loop number
        'ir_floor' : 2.5 ,              # [outer loop] rankir threshold
        'corr_cap' : 0.7 ,              # [outer loop] cap of correlation with existing factors
        'n_gen' : 2  ,                  # [inner loop] generation number
        'max_depth' : 5 ,               # [inner loop] max tree depth of gp
        'cxpb' : 0.35 ,                 # [inner loop] crossover probability
        'mutpb' : 0.25 ,                # [inner loop] mutation probability
    }
    if test_code:
        # when test code, change some parameters
        gp_params.update({'verbose' : True , 'pop_num': 4 , 'hof_num': 3 , 'n_iter' : 2 , 'max_depth' : 3 ,
                          'slice_date' : ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29']}) 
        
    gp_params['slice_date'] = pd.to_datetime(gp_params['slice_date']).values
    return gp_params

# %%
def gp_dictionary(gp_params , gp_timer = gpTimer()):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    基于遗传规划的参数字典，读取各类主要数据，并放在同一字典中传回
    input:
        gp_params: dict created by gp_parameters
        gp_timer:  gpTimer to record time cost
    output:
        gp_dict:   dict that includes gp parameters, gp datas and gp arguements, and various other datas
    '''

    def Ts(x , gp_key):
        # additional treatment based by gp_key
        if gp_key.startswith('dp'): # dividend factor , nan means 0
            x = x.nan_to_num()
        if isinstance(x , torch.Tensor):
            x = x.to(gp_params.get('device' , _device))
            x.share_memory_() # 执行多进程时使用：将张量移入共享内存
        return x
    
    package_path = f'{_DIR_pack}/gp_data_package{"_test" if gp_params["test_code"] else ""}.pt'
    with gp_timer('Load data' , df_cols = False):
        slice_date = gp_params['slice_date']
        gp_dict = {'gp_values' :[] ,
                   'gp_args'   :gp_params['gp_fac_list'] + gp_params['gp_raw_list'] ,
                   'n_args'    : (len(gp_params['gp_fac_list']) , len(gp_params['gp_raw_list'])) ,
                   **gp_params}
        
        if os.path.exists(package_path):
            print(f'Directly load {package_path}')
            package_data = torch.load(package_path)
            assert np.isin(gp_dict['gp_args'] , package_data['gp_args']).all() , np.setdiff1d(gp_dict['gp_args'] , package_data['gp_args'])
            for gp_arg in gp_dict['gp_args']:
                gp_val = package_data['gp_values'][package_data['gp_args'].index(gp_arg)]
                gp_val = Ts(gp_val , gp_arg)
                gp_dict['gp_values'].append(gp_val)

            for key in ['size' , 'cs_indus_code' , 'labels_raw' , 'labels_res' , 'df_index' , 'df_columns' , 'universe']: 
                gp_val = package_data[key]
                gp_val = Ts(gp_val , key)
                gp_dict[key] = gp_val

        else:
            nrowchar = 0
            for i , gp_key in enumerate(gp_dict['gp_args']):
                gp_val , df_indexes = read_gp_data(gp_key,slice_date,gp_dict.get('df_columns'),first_data=(i==0))
                gp_val = Ts(gp_val , gp_key)
                gp_dict['gp_values'].append(gp_val)
                if df_indexes: gp_dict.update(df_indexes)
                
                if gp_params.get('verbose'):
                    nrowchar += len(gp_key) + 1
                    print(gp_key , end='\n' if (i == len(gp_dict['gp_args']) - 1 or nrowchar >= 100) else ',')
                    if nrowchar >= 100: nrowchar = 0

            for key in ['size' , 'cs_indus_code']: 
                gp_val = read_gp_data(key,slice_date,gp_dict.get('df_columns'))[0]
                gp_val = Ts(gp_val , key)
                gp_dict[key] = gp_val

            if 'close_raw' in gp_dict['gp_args']:
                cp_raw = gp_dict['gp_values'][gp_dict['gp_args'].index('close_raw')]
            else:
                cp_raw = Ts(read_gp_data('close_raw',slice_date,gp_dict.get('df_columns'))[0] , 'close_raw')
            
            labels = F.ts_delay(F.pctchg(cp_raw, 10) , -11)  # t+1至t+11的收益率
            labels = F.neutralize_2d(labels, gp_dict['size'], gp_dict['cs_indus_code'])  # 市值行业中性化

            gp_dict['universe']   = ~cp_raw.isnan()
            gp_dict['labels_raw'] = labels
            gp_dict['labels_res'] = copy.deepcopy(labels)
            torch.save(gp_dict , package_path)

    print(f'{len(gp_params["gp_fac_list"])} factors, {len(gp_params["gp_raw_list"])} raw data loaded!')
    gp_dict['insample']  = (gp_dict['df_index'] >= slice_date[0]) * (gp_dict['df_index'] <= slice_date[1])
    gp_dict['outsample'] = (gp_dict['df_index'] >= slice_date[2]) * (gp_dict['df_index'] <= slice_date[3])
    gp_dict['gp_timer'] = gp_timer
    return gp_dict

def gp_data_filename(gp_key : str):
    '''
    ------------------------ gp input data filenames ------------------------
    原始因子名与文件名映射
    input:
        gp_key:      key of gp data
    output:
        gp_filename: basename of input data
    '''
    raw = gp_key.endswith('_raw')
    k   = gp_key[:-4] if raw else gp_key
    if k == 'open1':
        v = f'open'
    elif k == 'bp':
        v = f'{k}_lf'
    elif k in ['ep' , 'ocfp']:
        v = f'{k}_ttm'
    elif k == 'dp':
        v = 'dividendyield2'
    elif k == 'rtn':
        v = 'return1'
    elif k in ['close']:
        v = f'{k}_adj'
    else:
        v = k
    return f'{v}_day.parquet' if raw or v in ['cs_indus_code' , 'size'] else f'{v}_zscore_day.parquet'

def read_gp_data(gp_key,slice_date,stockcol=None,first_data=False,input_freq='D'):
    '''
    ------------------------ read gp data and convert to torch.tensor ------------------------
    读取单个原始因子文件并转化成tensor，额外返回df表格的行列字典
    input:
        gp_key:      key of gp data
        slice_date:  (insample_start, insample_end, outsample_start, outsample_end)
        stockcol:    if not None, filter columns
        first_data:  if True, return some gp dictionary members
        input_freq:  freq faster than day should code later
    output:
        ts:          result tensor
        df_indexes:  index and columns if first_data is True
    '''
    file = f'{_DIR_data}/{gp_data_filename(gp_key)}'
    df = pd.read_parquet(file, engine='fastparquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: 
        df = df[(slice_date[0] <= df.index.values) & (df.index.values <= slice_date[-1])] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    df = df[stockcol] if stockcol is not None else df # 选取指定股票

    if first_data:
        df_indexes = {
            'df_columns' : df.columns.values ,
            'df_index': df.index.values , # 日期行
        }
    else:
        df_indexes = None

    ts = torch.FloatTensor(df.values)
    return (ts , df_indexes)

# %%
def gp_population(toolbox , pop_num , **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    初始化种群
    input:
        gp_args:    initial gp factor names
        i_iter:     i of outer loop
        pop_num:    population number
    output:
        toolbox:    toolbox that contains all gp utils
        population: initial population of syntax
    '''
    population = toolbox.population(n = pop_num) #type:ignore
    population = toolbox.prune(population) #type:ignore
    return population

# %%
def evaluate(individual, pool_skuname, compiler , gp_values , labels_raw , labels_res, universe, insample , outsample , size , gp_timer , 
             const_annual = 24 , min_coverage = 0.5 , **kwargs):
    '''
    ------------------------ evaluate individual syntax fitness ------------------------
    从因子表达式起步，生成因子并计算适应度
    input:
        individual:     individual syntax, e.g. sigmoid(rank_sub(ts_grouping_decsortavg(turn, dp_raw , 15, 4), high)) 
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
    factor_value = gp_factor(factor_func , gp_values , gp_timer.acc_timer('eval'))
    
    rankir_list = []
    for resid in [True , False]:
        labels = labels_res if resid else labels_raw
        rankic = F.rankic_2d(factor_value , labels , dim = 1 , universe = universe , min_coverage = min_coverage)
        for is_insample in [True , False]:
            rankic_samp = rankic[insample] if is_insample else rankic[outsample]
            if rankic_samp.isnan().sum() / len(rankic_samp) > 0.5:
                # if too many nan rank_ic (due to low coverage)
                rankir_list.append(torch.nan)
            else:
                rankic_std = (((rankic_samp - rankic_samp.nanmean()) ** 2).nanmean()) ** 0.5 + 1e-6
                rankir_samp = (rankic_samp.nanmean() / rankic_std * np.sqrt(const_annual)) # 年化 ir
                rankir_samp[rankir_samp.isinf()] = torch.nan
                rankir_list.append(rankir_samp.nan_to_num().cpu())
    return (abs(rankir_list[0]),) , rankir_list

def gp_compile(compiler , individual , timer = gpTimer.EmptyTimer() , **kwargs):
    '''
    ------------------------ calculate individual syntax factor value ------------------------
    根据迭代出的因子表达式，计算因子值 , 基于函数表达式构建函数：deap.base.Toolbox().compile
    input:
        compiler:     compiler function to realize syntax computation, i.e. return factor function of given syntax
        individual:   individual syntax, e.g. sigmoid(rank_sub(ts_grouping_decsortavg(turn, dp_raw , 15, 4), high)) 
        timer:        record compile time
    output:
        func:         function of a syntax
    '''
    with timer:
        func = compiler(individual)

    return func

def gp_factor(func , gp_values , timer = gpTimer.EmptyTimer() , **kwargs):
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
        factor_value = process_factor(factor_value , 'inf' , dim = 1)
        
    return factor_value

def process_factor(value , process_key = 'inf_trim_norm' , dim = 1 , trim_ratio = 7. , 
                   size = None , cs_indus_code = None , **kwargs):
    '''
    ------------------------ process factor value ------------------------
    处理因子值 , 'inf_trim/winsor_norm_neutral_nan'
    input:
        value:         factor value to be processed
        process_key:   can be any of 'inf_trim/winsor_norm_neutral_nan'
        dim:           default to 1
        trim_ratio:    what extend can be identified as outlier? range is determined as med ± trim_ratio * brandwidth
        size:          market cap (neutralize param)
        cs_indus_code: industry indicator (neutralize param)
    output:
        value:         processed factor value
    '''
    assert isinstance(value , torch.Tensor)

    if 'inf' in process_key or ('trim' not in process_key and 'winsor' not in process_key):
        value[value.isinf()] = torch.nan

    if 'trim' in process_key or 'winsor' in process_key:
        med       = value.nanmedian(dim , keepdim=True).values
        bandwidth = value.nanquantile(0.75 , dim , keepdim=True) - value.nanquantile(0.25 , dim , keepdim=True)
        lbound , ubound = med - trim_ratio * bandwidth , med + trim_ratio * bandwidth
        if 'trim' in process_key:
            value[(value > ubound) + (value < lbound)] = torch.nan
        else:
            value = torch.where(value > ubound , ubound , value)
            value = torch.where(value < lbound , lbound , value)

    if 'norm' in process_key: value = F.zscore(value , dim = dim)
    # '市值中性化（对x做，现已取消，修改为对Y做）'
    # if 'neutral' in method: value = F.neutralize_2d(value , size , cs_indus_code , dim = dim) 
    if 'nan' in process_key: value = value.nan_to_num_()
    return value

# %%
def gp_eaSimple(toolbox , population , i_iter, pool_num=1,
                n_gen=2,cxpb=0.35,mutpb=0.25,hof_num=10,start_gen=None,gp_timer=gpTimer(),verbose=__debug__,stats=None,**kwargs):  
    """
    ------------------------ Evolutionary Algorithm simple ------------------------
    变异/进化小循环，从初始种群起步计算适应度并变异，重复n_gen次
    input:
        toolbox:    toolbox that contains all gp utils
        population: initial population of syntax
        i_iter:     i of outer loop
        pool_num:   multiprocessing pool number
        n_gen:      [inner loop] generation number
        cxpb:       [inner loop] crossover probability
        mutpb:      [inner loop] mutation probability
        hof_num:    halloffame number
        start_gen:  which gen to start, if None start a new
        gp_timer:   gpTimer to record time cost
    output:
        population: updated population of syntax
        halloffame: container of individuals with best fitness (no more than hof_num)
        logbook:    gp logbook

    ------------------------ basic code structure ------------------------
    evaluate(population)     # 对随机生成的初代种群评估IR值
    for g in range(n_gen)):
        population = select(population, len(population))    # 选取abs(IR)值较高的个体，以产生后代
        offspring = varAnd(population, toolbox, cxpb, mutpb)   # 交叉、变异
        evaluate(offspring)   # 对新种群评估IR值
        population = offspring    # 更新种群
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
            # Select the next generation individuals，based on abs(IR)
            offspring = toolbox.select(population, len(population))
            # Vary the pool of individuals: varAnd means variation part (crossover and mutation)
            with gp_timer.acc_timer('varAnd'):
                offspring = varAnd(offspring, toolbox, cxpb , mutpb)
                offspring = toolbox.prune(offspring) #type:ignore

        # Re-Evaluate the individuals with an invalid fitness（对于发生改变的个体，重新评估abs(IR)值）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # 'individual' arg for evaluate
        pool_list = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(invalid_ind))] # 'pool_skuname' arg for evaluate
        if pool_num > 1:
            pool = Pool(pool_num)
            fitnesses = list(tqdm(pool.imap(toolbox.evaluate, invalid_ind, pool_list), total=len(invalid_ind), desc="gen"+str(i_gen))) #type:ignore
            #fitnesses = pool.starmap(toolbox.evaluate, zip(invalid_ind, pool_list), chunksize=1)
            pool.close()
            pool.join()
            # pool.clear()
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, pool_list)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            #ind.fitness.values , ind.ir_list = tuple([fit[0]]) , fit[1] #type:ignore
            ind.fitness.values , ind.ir_list = fit #type:ignore

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

    return population, halloffame , logbook

# %%
def gp_hof_eval(toolbox , halloffame, i_iter , gp_values ,
                ir_floor=2.5,corr_cap=0.7,gp_timer=gpTimer(),verbose=__debug__,**kwargs):
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
    
    if i_iter > 0 and os.path.exists(f'{_DIR_job}/good_log.csv'): 
        good_log = pd.read_csv(f'{_DIR_job}/good_log.csv',index_col=0)
    else:
        good_log = pd.DataFrame(columns=['i_iter','i_good','sytax','max_corr',
                                         'rankir_in_resid','rankir_out_resid','rankir_in','rankir_out'])

    hof_good_list = []
    i_good = 0
    for hof_single in halloffame:
        if ((hof_single.fitness.values[0] > ir_floor) and  # icir larger than floor
            (hof_single.ir_list[1] != 0)): # icir_out_resid not zero (i.e. not nan)
            
            # 根据迭代出的因子表达式，计算因子值
            factor_func  = gp_compile(toolbox.compile , hof_single , gp_timer.acc_timer('compile'))
            factor_value = gp_factor(factor_func , gp_values , gp_timer.acc_timer('eval'))

            # 与已有的因子库做相关性检验，如果相关性大于预设值corr_cap，则不加入因子库
            high_correlation = False
            corr_values = torch.zeros(len(hof_good_list)).to(factor_value)
            for i , hof_good in enumerate(hof_good_list):
                corr_values[i] = F.corrwith(factor_value, hof_good, dim=1).nanmean().abs()
                # print(corr_values[i])
                if high_correlation := (corr_values[i] > corr_cap): break

            # 如果通过相关性检验，则加入因子库
            if not high_correlation:
                i_good += 1
                max_corr = 0. if len(corr_values) == 0 else corr_values.nan_to_num().max()
                print(f'Good {i_good} : ' + f'RankIR {hof_single.fitness.values[0]:.2f} ' + 
                      f'Corr {max_corr:.2f} : {str(hof_single)}')

                hof_good_list.append(factor_value)
                df = pd.DataFrame(data=factor_value.cpu().numpy() , index=kwargs['df_index'] , columns=kwargs['df_columns'])
                df.to_parquet(f'{_DIR_job}/factor/iter{i_iter}_good{i_good}.parquet', engine='fastparquet')

                df = pd.DataFrame([[i_iter,i_good,str(hof_single),max_corr,
                                    *hof_single.ir_list]],columns=good_log.columns) # new good_log
                good_log = pd.concat([good_log,df],axis=0)
    good_log.to_csv(f'{_DIR_job}/good_log.csv')
    return halloffame , hof_good_list

# %%
def outer_loop(i_iter , gp_dict):
    """
    ------------------------ gp outer loop ------------------------
    大循环主程序，初始化种群、变异、筛选、更新残差labels、记录
    input:
        i_iter:   i of outer loop
        gp_dict:  dict that includes gp parameters, gp datas and gp arguements, and various other datas
    output:
        None
    process:
        gp_population : create initial population
        gp_eaSimple   : Evolutionary Algorithm simple
        gp_hof_eval   : gp halloffame evaluation
        update labels : update labels_res , according to hof_good_list
    """
    timenow = time.time()
    gp_dict['i_iter'] = i_iter
    
    '''Initialize GP toolbox and Population'''
    with gp_dict['gp_timer']('Population'):
        toolbox = gpHandler.Toolbox(eval_func=evaluate , **gp_dict)
        population = gp_population(toolbox , **gp_dict)
    
    '''运行进化主程序'''
    with gp_dict['gp_timer']('Evolution'):
        population, halloffame, logbook = gp_eaSimple(toolbox , population , **gp_dict)  #, start_gen=6   #algorithms.eaSimple

    '''good_log文件为因子库，存储了因子表达式、rankIR值等信息。在具体因子值存储在factor文件夹的parquet文件中'''
    with gp_dict['gp_timer']('Selection'):
        halloffame, hof_good_list = gp_hof_eval(toolbox , halloffame, **gp_dict)

    '''更新labels_res，并保存本轮循环的最终结果'''
    with gp_dict['gp_timer']('Neu_dump'):
        # update labels_resid, according to hof_good_list (maybe no need to neutralize according to size and indus)
        gp_dict['labels_res'] = F.neutralize_2d(gp_dict['labels_res'], hof_good_list)
        '''保存该轮迭代的最终种群、最优个体、迭代日志'''
        joblib.dump(population,f'{_DIR_job}/pop_iter{i_iter}_overall.pkl')
        joblib.dump(halloffame,f'{_DIR_job}/hof_iter{i_iter}_overall.pkl')
        joblib.dump(logbook   ,f'{_DIR_job}/log_iter{i_iter}_overall.pkl')

    gp_dict['gp_timer'].append_time('AvgVarAnd' , gp_dict['gp_timer'].acc_timer('varAnd').avgtime(pop_out = True))
    gp_dict['gp_timer'].append_time('AvgCompile', gp_dict['gp_timer'].acc_timer('compile').avgtime(pop_out = True))
    gp_dict['gp_timer'].append_time('AvgEval',    gp_dict['gp_timer'].acc_timer('eval').avgtime(pop_out = True))
    gp_dict['gp_timer'].append_time('All' , time.time() - timenow)
    return 
    
def gp_factor_generator(**kwargs):
    '''
    ------------------------ gp factor generator ------------------------
    构成因子生成器，返回一个函数，输入因子表达式则输出历史因子值
    input:
        kwargs:  specific gp_params members, suggestion is to leave it alone
    output:
        wrapper: lambda syntax:factor value
    '''
    gp_params = gp_parameters(train = False)
    gp_params.update(kwargs)
    gp_dict = gp_dictionary(gp_params)
    toolbox = gpHandler.Toolbox(eval_func=evaluate , **gp_dict)
        
    def wrapper(syntax , process_key = 'inf_trim_norm'):
        func = toolbox.compile(syntax) #type:ignore
        value = func(*gp_dict['gp_values'])
        value = process_factor(value , process_key , dim = 1)
        return value
    
    return wrapper

# %%
def main(test_code = None , job_id = None):
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
    gp_dict = gp_dictionary(gp_parameters(test_code , job_id))
    for i_iter in range(gp_dict['n_iter']):
        print(f' ------------------------------- Outer Loop {i_iter} -------------------------------')
        outer_loop(i_iter , gp_dict)
    print('Total time cost table:')
    gp_dict['gp_timer'].save_to_csv(f'{_DIR_job}/saved_times.csv' , print_out = True)

if __name__ == '__main__':
    main()
