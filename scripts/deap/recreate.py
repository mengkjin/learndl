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

import math_func_gpu as F
from gp_utils import gpPrimatives , gpTimer

# %%
# ------------------------ environment setting ------------------------

np.seterr(over='raise')

_plat      = platform.system().lower()                                    # Windows or Linux
_device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda or cpu
_test_code = True or (_device == torch.device('cpu'))                     # True if just to test code is valid
_DIR_data  = './data/features/parquet'                                    # input路径，即原始因子所在路径。
_DIR_pack  = './data/package'                                             # input路径2, 加载原始因子后所保存的pt文件。
_DIR_pop   = './pop'                                                      # output路径，即保存因子库、因子值、因子表达式的路径。
_DIR_job   = f'{_DIR_pop}/bendi'                                          # specific job path 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                               # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
torch.multiprocessing.set_start_method('spawn' if _plat == 'windows' else 'forkserver', force=True) # multiprocessing method, 单机多进程设置（实际未启用），参考https://zhuanlan.zhihu.com/p/600801803

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=-1)
parser.add_argument("--poolnm", type=int, default=1)
args, _ = parser.parse_known_args()

# ------------------------ gp parameters ------------------------
def gp_parameters(test_code = None , job_id = None):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    input:
        test_code: if only to test code validity
        job_id:    when test_code is not True, determines _DIR_job = f'{_DIR_pop}/{job_id}'   
    output:
        gp_params: dict that includes all gp parameters
    
    ---------参数初始化---------------
    【注：为方便跑多组实验，需设置job_id参数（设置方式为: python xxx.py --job_id 123456）】

    以下参数均为全局参数，需在此处修改
    device:             用cuda还是cpu计算
    verbose:            训练过程是否输出细节信息
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
        
    # directory setting and making
    global _DIR_job
    if test_code is None: test_code = _test_code
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

    df_fac_list = ['close', 'turn', 'volume', 'amt', 'open', 'high', 'low', 'vwap', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']
    df_raw_list = [f'{_d}_raw' for _d in df_fac_list] + ['rtn_raw']
    
    gp_params = {
        'test_code'   : test_code ,   # just to check code, lighter parameters
        'df_fac_list' : df_fac_list , # gp intial factor list 
        'df_raw_list' : df_raw_list , # gp intial raw data list
        'device' : _device , # training device, cuda or cpu
        'verbose' : False ,  # if show some text
        'pool_num' : 1 ,    # args.poolnm # multiprocessing pool number
        'pop_num': 3000 ,   # population number
        'hof_num': 500 ,    # halloffame number
        'slice_date' : ['2010-01-04', '2021-12-31', '2022-01-04', '2023-12-29'] ,  # must be trade date
        'n_iter' :  5 ,     # [outer loop] loop number
        'ir_floor' : 2.5 ,  # [outer loop] rankir threshold
        'corr_cap' : 0.7 ,  # [outer loop] cap of correlation with existing factors
        'n_gen' : 2  ,      # [inner loop] generation number
        'max_depth' : 5 ,   # [inner loop] max tree depth of gp
        'cxpb' : 0.35 ,     # [inner loop] crossover probability
        'mutpb' : 0.25 ,    # [inner loop] mutation probability
    }
    if test_code:
        # when test code, change some parameters
        gp_params.update({'verbose' : True , 'pop_num': 4 , 'hof_num': 3 , 'n_iter' : 2 , 'max_depth' : 3 ,
                          'slice_date' : ['2022-01-04', '2022-12-30', '2023-01-04', '2023-12-29']}) 
    return gp_params

# %%
def gp_dictionary(gp_params , gp_timer = gpTimer()):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    input:
        gp_params: dict created by gp_parameters
        gp_timer:  gpTimer to record time cost
    output:
        gp_dict:   dict that includes gp parameters, gp datas and gp arguements, and various other datas
    '''

    def Ts(df):
        if isinstance(df , torch.Tensor):
            if gp_params.get('device') is not None: df = df.to(gp_params.get('device'))
            df.share_memory_() # 执行多进程时使用：将张量移入共享内存
        return df
    package_path = f'{_DIR_pack}/gp_data_package{"_test" if gp_params["test_code"] else ""}.pt'
    with gp_timer('Load data' , df_cols = False):
        gp_dict = {}
        slice_date = gp_params['slice_date']

        # define what factors and raw datas will be used for initial population
        df_fac_list = ['close', 'turn', 'volume', 'amt', 'open', 'high', 'low', 'vwap', 'bp', 'ep', 'ocfp', 'dp', 'adv20', 'adv60']
        df_raw_list = [f'{fac}_raw' for fac in df_fac_list] + ['rtn_raw']
        
        gp_dict['gp_args'] = df_fac_list + df_raw_list # gp args sequence
        gp_dict['gp_values'] = []
        if os.path.exists(package_path):
            print(f'Directly load {package_path}')
            package_data = torch.load(package_path)
            assert np.isin(gp_dict['gp_args'] , package_data['gp_args']).all() , np.setdiff1d(gp_dict['gp_args'] , package_data['gp_args'])
            for gp_arg in gp_dict['gp_args']:
                ts = package_data['gp_values'][package_data['gp_args'].index(gp_arg)]
                gp_dict['gp_values'].append(Ts(ts))

            for key in ['size' , 'cs_indus_code' , 'labels' , 'labels_resid' , 'df_index' , 'df_columns']: 
                gp_dict[key] = Ts(package_data[key])
        else:
            nrowchar = 0
            for i , gp_key in enumerate(gp_dict['gp_args']):
                ts , first_data_return = read_gp_data(gp_key,slice_date,gp_dict.get('df_columns'),first_data=(i==0))
                if first_data_return: gp_dict.update(first_data_return)
                gp_dict['gp_values'].append(Ts(ts))
                if gp_params.get('verbose'):
                    nrowchar += len(gp_key) + 1
                    print(gp_key , end=',')
                    if i == len(gp_dict['gp_args']) - 1 or nrowchar >= 100: print()
                    if nrowchar >= 100: nrowchar = 0

            for key in ['size' , 'cs_indus_code']: 
                gp_dict[key] = Ts(read_gp_data(key,slice_date,gp_dict.get('df_columns'))[0])

            cp_raw = gp_dict['gp_values'][gp_dict['gp_args'].index('close_raw')]
            labels = F.ts_delaypct(cp_raw, 10)  # t-10至t的收益率
            labels = F.ts_delay(labels, -11)  # t+1至t+11的收益率
            labels_resid = F.neutralize_2d(labels, gp_dict['size'], gp_dict['cs_indus_code'])  # 市值行业中性化
            gp_dict['labels']  = copy.deepcopy(labels_resid)
            gp_dict['labels_resid'] = labels_resid
            torch.save(gp_dict , package_path)

    print(f'{len(df_fac_list)} factors, {len(df_raw_list)} raw data loaded!')
    gp_dict['n_args']  = (len(df_fac_list) , len(df_raw_list))
    gp_dict.update(gp_params)
    slice_date = np.array(slice_date).astype(type(gp_dict['df_index'][0]))
    gp_dict['insample_dates']  = (gp_dict['df_index'] >= slice_date[0]) * (gp_dict['df_index'] <= slice_date[1])
    gp_dict['outsample_dates'] = (gp_dict['df_index'] >= slice_date[2]) * (gp_dict['df_index'] <= slice_date[3])
    gp_dict['gp_timer'] = gp_timer
    return gp_dict

def gp_data_filename(gp_key : str):
    '''
    ------------------------ gp input data filenames ------------------------
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
    input:
        gp_key:      key of gp data
        slice_date:  (insample_start, insample_end, outsample_start, outsample_end)
        stockcol:    if not None, filter columns
        first_data:  if True, return some gp dictionary members
        input_freq:  freq faster than day should code later
    output:
        gp_filename: basename of input data
    '''
    file = f'{_DIR_data}/{gp_data_filename(gp_key)}'
    df = pd.read_parquet(file, engine='fastparquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: df = df[slice_date[0]:slice_date[-1]] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    df = df[stockcol] if stockcol is not None else df # 选取指定股票

    if first_data:
        first_data_return = {
            'df_columns' : df.columns.values ,
            'df_index': df.index.values , # 日期行
        }
    else:
        first_data_return = None

    ts = torch.FloatTensor(df.values)
    return (ts , first_data_return)

# %%
def gp_toolbox(gp_args , i_iter , max_depth = 5 , n_args = (1,1) , **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    input:
        gp_args:   initial gp factor names
        i_iter:    i of outer loop
        max_depth: [inner loop] max tree depth of gp
        n_args:    number of gp factors, (n_of_zscore_factors, n_of_raw_indicators)
    output:
        toolbox:   toolbox that contains all gp utils
    '''
    pset = gpPrimatives.new_pset(*n_args , arg_names=gp_args , i_iter=i_iter)

    '''创建遗传算法基础模块，以下参数不建议更改，如需更改，可参考deap官方文档'''
    # https://zhuanlan.zhihu.com/p/72130823
    [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual']]
    creator.create("FitnessMin", base.Fitness, weights=(+1.0,))   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset) # type:ignore 个体编码：pset，预设的

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_= max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)# type:ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)# type:ignore
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, compiler = toolbox.compile , **kwargs) # type: ignore
    toolbox.register("select", tools.selTournament, tournsize=3) # 锦标赛：第一轮随机选择3个，取最大
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_= max_depth)  # genFull
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # type:ignore
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))  # max=3

    return toolbox

# %%
def cal_gp_factor(compiler , individual , gp_values , timer = None , **kwargs):
    '''
    ------------------------ calculate individual syntax factor value ------------------------
    根据迭代出的因子表达式，计算因子值 , 基于函数表达式构建函数：deap.base.Toolbox().compile
    input:
        compiler:     compiler function to realize syntax computation, i.e. return factor function of given syntax
        individual:   individual syntax, e.g. sigmoid(rank_sub(ts_grouping_decsortavg(turn, dp_raw , 15, 4), high)) 
        gp_values:    initial population factor values
        timer:        record compile time
    output:
        factor_value: 2d tensor
    '''
    if timer is None: timer = gpTimer.EmptyTimer() 
    with timer:
        factor_value = compiler(expr = individual)(*gp_values)
        if isinstance(factor_value, float): factor_value = torch.full_like(gp_values[0] , factor_value)
        factor_value[factor_value.isinf()] = torch.nan # 异常值处理：inf转换为nan
        # factor_value = neutralize_torch(factor_value,size,cs_indus_code) # '市值中性化（对x做，现已取消，修改为对Y做）'
    return factor_value

# %%
def evaluate(individual, pool_skuname, compiler , gp_values , labels , labels_resid, insample_dates , outsample_dates , size , gp_timer , 
             const_annual = 24 , **kwargs):
    '''
    ------------------------ evaluate individual syntax fitness ------------------------
    input:
        individual:     individual syntax, e.g. sigmoid(rank_sub(ts_grouping_decsortavg(turn, dp_raw , 15, 4), high)) 
        pool_skuname:   pool skuname in pool.imap, e.g. iter0_gen0_0
        compiler:       compiler function to realize syntax computation, i.e. return factor function of given syntax
        gp_values:      initial population factor values
        labels:         raw label, e.g. 20 day future return
        labels_resid:   residual label, neutralized by indus and size
        insample_dates: insample_dates
        outsample_dates:outsample_dates
        const_annual:   constant or annualization
    output:
        tuple of (
            abs_rankir: (abs(insample_resid), ) # !! Fitness definition 
            rankir:     (insample_resid, outsample_resid, insample_raw, outsample_raw)
        )
    '''
    # 记录开始时间并输出txt
    # print(str(individual))
    if int(pool_skuname.split('_')[-1])%100 == 0:
        start_time_sku = time.time()
        output_path = f'{_DIR_job}/z_{pool_skuname}.txt'
        with open(output_path, 'w', encoding='utf-8') as file1:
            print(str(individual),'\n start_time',time.ctime(start_time_sku),file=file1)

    # 根据迭代出的因子表达式，计算因子值
    factor_value = cal_gp_factor(compiler , individual , gp_values , gp_timer.acc_timer('compile'))
    
    rankir_list = []
    for resid in [True , False]:
        rankic = F.corrwith(F.rank_pct(factor_value), F.rank_pct(labels_resid if resid else labels),dim=1)
        for in_sample in [True , False]:
            rankic_samp = rankic[insample_dates] if in_sample else rankic[outsample_dates]
            rankic_std = (((rankic_samp - rankic_samp.nanmean()) ** 2).nanmean()) ** 0.5 + 1e-6
            rankir_samp = (rankic_samp.nanmean() / rankic_std * np.sqrt(const_annual)) # 年化 ir
            rankir_samp[rankir_samp.isinf()] = torch.nan
            rankir_list.append(rankir_samp.nan_to_num().cpu())
    return (abs(rankir_list[0]),) , rankir_list

# %%
def gp_eaSimple(toolbox , population , i_iter, pool_num=1,
                n_gen=2,cxpb=0.35,mutpb=0.25,hof_num=10,start_gen=None,gp_timer=gpTimer(),verbose=__debug__,stats=None,**kwargs):  
    """
    ------------------------ Evolutionary Algorithm simple ------------------------
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
        good_log = pd.DataFrame(columns=['i_iter','str_hof','rankir_in_resid','rankir_out_resid','rankir_in','rankir_out'])

    '''因子库存储在hof_good_list中，每个因子为一个二维tensor（时间*股票）'''
    '''对rankIR大于设定值的因子，进行相关性检验，如果相关性低，则加入因子库'''
    hof_good_list = []
    timer = gp_timer.acc_timer('compile')
    for hof_single in halloffame:
        if ((hof_single.fitness.values[0] > ir_floor) and  # icir larger than floor
            (hof_single.ir_list[1] != 0)): # icir_out_resid not zero (i.e. not nan)
            
            # 根据迭代出的因子表达式，计算因子值
            factor_value = cal_gp_factor(toolbox.compile , hof_single , gp_values , timer)

            # 与已有的因子库做相关性检验，如果相关性大于预设值corr_cap，则不加入因子库
            high_correlation = False
            corr_values = torch.zeros(len(hof_good_list)).to(factor_value)
            for i , hof_good in enumerate(hof_good_list):
                corr_values[i] = F.corrwith(factor_value, hof_good, dim=1).nanmean().abs()
                # print(corr_value)
                if high_correlation := (corr_values[i] > corr_cap): break

            # 如果通过相关性检验，则加入因子库
            if not high_correlation:
                max_corr = 0. if len(corr_values) == 0 else corr_values.nan_to_num().max()
                print(f'Good {len(hof_good_list)} : ' +
                      f'RankIR {hof_single.fitness.values[0]:.2f} ' + 
                      f'Corr {max_corr:.2f} : {str(hof_single)}')

                hof_good_list.append(factor_value)
                df = pd.DataFrame(factor_value.cpu(), index=kwargs['df_index'], columns=kwargs['df_columns']) # type: ignore
                df.to_parquet(f'{_DIR_job}/factor/{str(hof_single)}.parquet', engine='fastparquet')

                df = pd.DataFrame([[i_iter,str(hof_single),*hof_single.ir_list]],columns=good_log.columns) # new good_log
                good_log = pd.concat([good_log,df],axis=0)

    good_log.to_csv(f'{_DIR_job}/good_log.csv')
    return halloffame , hof_good_list

# %%
def outer_loop(i_iter , gp_dict):
    """
    ------------------------ gp outer loop ------------------------
    input:
        i_iter:   i of outer loop
        gp_dict:  dict that includes gp parameters, gp datas and gp arguements, and various other datas
    output:
        None
    process:
        gp_toolbox   : create gp toolbox
        gp_eaSimple  : Evolutionary Algorithm simple
        gp_hof_eval  : gp halloffame evaluation
        labels_resid : update labels_resid , according to hof_good_list
    """
    timenow = time.time()
    gp_dict['i_iter'] = i_iter
    
    '''Setting GP toolbox and pop'''
    with gp_dict['gp_timer']('Setting'):
        toolbox = gp_toolbox(**gp_dict)
        population = toolbox.population(n = gp_dict['pop_num']) # type:ignore
    
    '''运行进化主程序'''
    with gp_dict['gp_timer']('Evolution'):
        population, halloffame, logbook = gp_eaSimple(toolbox , population , **gp_dict)  #, start_gen=6   #algorithms.eaSimple

    '''good_log文件为因子库，存储了因子表达式、rankIR值等信息。在具体因子值存储在factor文件夹的parquet文件中'''
    with gp_dict['gp_timer']('Selection'):
        halloffame, hof_good_list = gp_hof_eval(toolbox , halloffame, **gp_dict)

    '''更新labels_resid，并保存本轮循环的最终结果'''
    with gp_dict['gp_timer']('Neu_dump'):
        # update labels_resid, according to hof_good_list (maybe no need to neutralize according to size and indus)
        gp_dict['labels_resid'] = F.neutralize_2d(gp_dict['labels_resid'], hof_good_list)
        '''保存该轮迭代的最终种群、最优个体、迭代日志'''
        joblib.dump(population,f'{_DIR_job}/pop_iter{i_iter}_overall.pkl')
        joblib.dump(halloffame,f'{_DIR_job}/hof_iter{i_iter}_overall.pkl')
        joblib.dump(logbook   ,f'{_DIR_job}/log_iter{i_iter}_overall.pkl')

    gp_dict['gp_timer'].append_time('Avg_varAnd' , gp_dict['gp_timer'].acc_timer('varAnd').avgtime(pop_out = True))
    gp_dict['gp_timer'].append_time('Avg_compile', gp_dict['gp_timer'].acc_timer('compile').avgtime(pop_out = True))
    gp_dict['gp_timer'].append_time('All' , time.time() - timenow)
    return 
    

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
