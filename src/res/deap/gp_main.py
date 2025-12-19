# %%
import pandas as pd
import numpy as np
import torch
import copy , os, platform , shutil , yaml

from argparse import ArgumentParser , Namespace
from datetime import datetime
from deap.algorithms import varAnd
from torch.multiprocessing import Pool
from tqdm import tqdm
from typing import Literal

from src.proj import Logger
from src.basic import torch_load
from . import gp_math_func as MF
from . import gp_factor_func as FF
from .gp_utils import gpHandler , gpTimer , gpContainer , gpFileManager , MemoryManager , gpEliteGroup , gpFitness
from .gp_affiliates import Profiler , EmptyTM

# %%
# ------------------------ environment setting ------------------------

# np.seterr(over='raise')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                                # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099

parser = ArgumentParser()
parser.add_argument("--job_id", type=int, default=-1)
parser.add_argument("--poolnm", type=int, default=1)
args , _ = parser.parse_known_args()
assert args.poolnm == 1 , args.poolnm # 若并行,通信成本过高,效率提升不大

defaults = Namespace(
    plat        = platform.system().lower() ,                               # Windows or Linux
    device      = torch.device(0 if torch.cuda.is_available() else 'cpu') , # cuda or cpu
    test_code   = True or not torch.cuda.is_available()   ,                 # 是否只是测试代码有无bug,默认False
    noWith      = True ,                                                    # 是否取消所有计时器,默认False,有计时器时报错会出问题
    DIR_data    = './data/features/parquet' ,                               # input路径1,原始因子所在路径
    DIR_pack    = './data/package' ,                                        # input路径2,加载原始因子后保存pt文件加速存储
    DIR_pop     = './pop' ,                                                 # output路径,即保存因子库、因子值、因子表达式的路径
    PATH_param  = f'./gp_params.yaml' ,                                     # 保存测试参数的地址 
    encoding    = 'utf-8' ,                                                 # 带中文的encoding
    poolnm      = args.poolnm ,                                             # 单机多进程设置
    job_id      = args.job_id ,
)

# multiprocessing method, 单机多进程设置(实际未启用),参考https://zhuanlan.zhihu.com/p/600801803
torch.multiprocessing.set_start_method('spawn' if defaults.plat == 'windows' else 'forkserver', force=True) 

def text_manager(object):
    if defaults.noWith:
        return EmptyTM(object)
    else:
        return object

def gp_job_dir(job_id = None , train = True , continuation = False , test_code = False):
    '''
    ------------------------ gp job dir ------------------------
    
    确定目标文件夹
    input:
        job_id:         when test_code is not True, determines f'{_DIR_pop}/{job_id}' 
        train:          if True, will first check dirs and device  
        continuation:   if True, will continue on start_iter / start_gen
        test_code:      if only to test code validity
    output:
        gp_params: dict that includes all gp parameters
    '''
    # directory setting and making
    if job_id is None: 
        job_id = args.job_id
    if test_code or job_id == 'bendi':
        test_code = True
        job_dir = f'{defaults.DIR_pop}/bendi'
    else:
        job_id = int(job_id)
        if job_id < 0:
            old_job_df = os.listdir('./pop/') if os.path.exists('./pop/') else []
            old_job_id = np.array([id for id in old_job_df if id.isdigit()]).astype(int)
            if train:
                job_id = np.setdiff1d(np.arange(max(old_job_id) + 2) , old_job_id).min() if len(old_job_id) else 0
            else:
                job_id = old_job_id.max()
        job_dir = f'{defaults.DIR_pop}/{job_id}'
    Logger.stdout(f'**Job Directory is : "{job_dir}"')
    if train:
        if os.path.exists(job_dir) and not continuation: 
            if not test_code and not input(f'Path "{job_dir}" exists , press "yes" to confirm Deletion:')[0].lower() == 'y':
                raise Exception(f'Deletion Denied!')
            shutil.rmtree(job_dir)
            Logger.stdout(f'  --> Start New Training in "{job_dir}"')
        #elif os.path.exists(job_dir):
        #    if not test_code and not input(f'Path "{job_dir}" exists , press "yes" to confirm Continuation:')[0].lower() == 'y':
        #        raise Exception(f'Continuation Denied!')
        elif not os.path.exists(job_dir) and continuation:
            raise Exception(f'No existing "{job_dir}" for Continuation!')
        else:
            Logger.stdout(f'  --> Continue Training in "{job_dir}"')
    return job_dir , test_code

def gp_parameters(job_id = None , train = True , continuation = False , test_code = False , **kwargs):
    '''
    ------------------------ gp parameters ------------------------
    
    主要遗传规划参数初始化
    注：为方便跑多组实验,需设置job_id参数,设置方式为: python xxx.py --job_id 123456
    input:
        job_id:         when test_code is not True, determines f'{defaults.DIR_pop}/{job_id}' 
        train:          if True, will first check dirs and device  
        continuation:   if True, will continue on start_iter / start_gen
        test_code:      if only to test code validity
    output:
        gp_params: dict that includes all gp parameters
    '''

    job_dir , test_code = gp_job_dir(job_id , train , continuation , test_code)
    if train and defaults.device.type == 'cpu':
        Logger.stdout('**Cuda not available')
    elif train:
        Logger.stdout('**Device name:', torch.cuda.get_device_name(defaults.device))

    gp_params = gpContainer(
        job_dir = job_dir ,           # the main directory
        test_code = test_code  ,      # just to check code, will save parquet in this case
        device = defaults.device ,    # training device, cuda or cpu
        pool_num = defaults.poolnm,   # multiprocessing pool number
    )

    with open(defaults.PATH_param ,'r',encoding=defaults.encoding) as f: 
        '''
        参数列表:
        job_dir             工作目录
        test_code           是不是仅仅作为代码测试,若是会有一个小很多的参数组合覆盖本参数
        gp_fac_list         作为GP输入时,标准化因子的数量
        gp_raw_list         作为GP输入时,原始指标的数量
        slice_date:         修改数据切片区间,前两个为样本内的起止点,后两个为样本外的起止点均需要是交易日
        device:             用cuda还是cpu计算
        verbose:            训练过程是否输出细节信息
        pool_num:           并行任务数量,建议设为1,即不并行,使用单显卡单进程运行
        pop_num:            种群数量,初始化时生成多少个备选公式
        hof_num:            精英数量,一般精英数量设为种群数量的1/6左右即可
        n_iter:             [大循环]的迭代次数,每次迭代重新开始一次遗传规划、重新创立全新的种群,以上一轮的残差收益率作为优化目标
        ir_floor:           [大循环]中因子入库所需的最低rankIR值,低于此值的因子不入库
        ir_floor_decay:     [大循环]是否每一轮迭代降低ir的标准,若降低则输入小于1.0的数
        corr_cap:           [大循环]中新因子与老因子的最高相关系数,相关系数绝对值高于此值的因子不入库
        factor_neut_type:   [大循环]i_iter>0时如何中性化因子 0:不中性化, 1:根据样本内的相关性一口气中性化, 2:每天单独中性化
        labels_neut_type:   [大循环]计算残差收益时,怎么使用Elite因子: 'svd' , 'all'
        svd_mat_method:     [大循环]svd factor的矩阵怎么计算: 'total'代表所有日期所有因子值相关矩阵, 'coef_ts'代表所有因子值时序与labels相关性时序的相关矩阵
        svd_top_ratio:      [大循环]svd factor最小解释力度
        svd_top_n:          [大循环]svd factor最少因子数量
        n_gen:              [小循环]的迭代次数,即每次遗传规划进行几轮繁衍进化
        max_depth:          [小循环]中个体算子树的最大深度,即因子表达式的最大复杂度
        select_offspring    [小循环]每次后代如何选择是否进入遗传突变环节,可以是 'best' , '2Tour' , 'Tour'
        surv_rate:          [小循环]上面选best的话,这里输入具体比例
        cxpb:               [小循环]中交叉概率,即两个个体之间进行交叉的概率
        mutpb:              [小循环]中变异概率,即个体进行突变变异的概率
        '''
        gp_params.update(**yaml.load(f , Loader = yaml.FullLoader))

    if test_code: 
        gp_params.update(**gp_params.test_params)
    gp_params.update(**kwargs)
    gp_params.apply('slice_date' , lambda x:pd.to_datetime(x).values)
    return gp_params

# %%
def gp_namespace(gp_params):
    '''
    ------------------------ gp dictionary, record data and params ------------------------
    
    基于遗传规划的参数字典,读取各类主要数据,并放在同一字典中传回
    input:
        gp_params: gpContainer of all gp_parameters
    output:
        gp_space:  gpContainer that includes gp parameters, gp datas and gp arguements, and various other datas
    '''
    gp_space = gpContainer(
        param       = gp_params , 
        device      = gp_params.device ,
        gp_inputs   = [] , 
        gp_argnames = gp_params.gp_fac_list + gp_params.gp_raw_list ,
        n_args      = (len(gp_params.gp_fac_list) , len(gp_params.gp_raw_list)) ,
        tensors     = gpContainer() ,
        records     = gpContainer() ,
        mgr_file    = gpFileManager(gp_params.get('job_dir')) , 
        mgr_mem     = MemoryManager(0) ,
        timer       = gpTimer(True) ,
        fitness     = gpFitness(gp_params.get('fitness_wgt')) , 
        df_columns  = None ,
    )
    with gp_space.timer('Data' , df_cols = False , print_str= '**Load Data'):
        package_path = f'{defaults.DIR_pack}/gp_data_package' + '_test' * gp_space.param.test_code + '.pt'
        package_require = ['gp_argnames' , 'gp_inputs' , 'size' , 'indus' , 'labels_raw' , 'df_index' , 'df_columns' , 'universe']

        load_finished = False
        package_data = torch_load(package_path) if os.path.exists(package_path) else {}

        if not np.isin(package_require , list(package_data.keys())).all() or not np.isin(gp_space.gp_argnames , package_data['gp_argnames']).all():
            if gp_space.param.verbose: 
                Logger.stdout(f'  --> Exists "{package_path}" but Lack Required Data!')
        else:
            assert np.isin(package_require , list(package_data.keys())).all() , np.setdiff1d(package_require , list(package_data.keys()))
            assert np.isin(gp_space.gp_argnames , package_data['gp_argnames']).all() , np.setdiff1d(gp_space.gp_argnames , package_data['gp_argnames'])
            assert package_data['df_index'] is not None

            if gp_space.param.verbose: 
                Logger.stdout(f'  --> Directly load "{package_path}"')
            for gp_key in gp_space.gp_argnames:
                gp_val = package_data['gp_inputs'][package_data['gp_argnames'].index(gp_key)]
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_inputs.append(gp_val)

            for gp_key in ['size' , 'indus' , 'labels_raw' , 'universe']: 
                gp_val = package_data[gp_key]
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.tensors.update(**{gp_key:gp_val})

            for gp_key in ['df_index' , 'df_columns']: 
                gp_val = package_data[gp_key]
                gp_space.records.update(**{gp_key:gp_val})

            load_finished = True

        if not load_finished:
            if gp_space.param.verbose: 
                Logger.stdout(f'  --> Load from Parquet Files:')
            gp_filename = gp_filename_converter()
            nrowchar = 0
            for i , gp_key in enumerate(gp_space.gp_argnames):
                if gp_space.param.verbose and nrowchar == 0: 
                    Logger.stdout('  --> ' , end='')
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.param.slice_date,gp_space.records.get('df_columns'))
                if i == 0: 
                    gp_space.records.update(df_columns = gp_val.columns.values , df_index = gp_val.index.values)
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.gp_inputs.append(gp_val)
                
                if gp_space.param.verbose:
                    Logger.stdout(gp_key , end=',')
                    nrowchar += len(gp_key) + 1
                    if nrowchar >= 100 or i == len(gp_space.gp_argnames):
                        Logger.stdout()
                        nrowchar = 0

            for gp_key in ['size' , 'indus']: 
                gp_val = read_gp_data(gp_filename(gp_key),gp_space.param.slice_date,gp_space.records.get('df_columns'))
                gp_val = df2ts(gp_val , gp_key , gp_space.device)
                gp_space.tensors.update(**{gp_key:gp_val})

            if 'CP' in gp_space.gp_argnames:
                CP = gp_space.gp_inputs[gp_space.gp_argnames.index('CP')]      
            else:
                CP = df2ts(read_gp_data(gp_filename('CP'),gp_space.param.slice_date,gp_space.records.get('df_columns')) , 'CP' , gp_space.device)    
            gp_space.tensors.universe   = ~CP.isnan() 
            gp_space.tensors.labels_raw = gp_labels_raw(CP , gp_space.tensors.size , gp_space.tensors.indus)
            os.makedirs(defaults.DIR_pack , exist_ok=True)
            saved_data = (gp_space.subset(['gp_argnames' , 'gp_inputs'] , require = True) |
                          gp_space.tensors.subset(['size' , 'indus' , 'labels_raw' , 'universe'] , require = True) |
                          gp_space.records.subset(['df_index' , 'df_columns'] , require = True) )
            torch.save(saved_data , package_path)

    if gp_space.param.verbose: 
        Logger.stdout(f'  --> {len(gp_space.param.gp_fac_list)} factors, {len(gp_space.param.gp_raw_list)} raw data loaded!')

    gp_space.mgr_file.save_state(gp_space.param.__dict__, 'params', i_iter = 0) # useful to assert same index as package data
    gp_space.mgr_file.save_state(gp_space.records.subset(['df_index' , 'df_columns']),'df_axis' , i_iter = 0) # useful to assert same index as package data
    
    gp_space.tensors.insample  = torch.Tensor((gp_space.records.df_index >= gp_space.param.slice_date[0]) * 
                                              (gp_space.records.df_index <= gp_space.param.slice_date[1])).bool()
    gp_space.tensors.outsample = torch.Tensor((gp_space.records.df_index >= gp_space.param.slice_date[2]) * 
                                              (gp_space.records.df_index <= gp_space.param.slice_date[3])).bool()
    if gp_space.param.factor_neut_type == 1:
        gp_space.tensors.insample_2d = gp_space.tensors.insample.reshape(-1,1).expand(gp_space.tensors.labels_raw.shape)

    return gp_space

def gp_labels_raw(CP = None , neutral_factor = None , neutral_group = None , nday = 10 , delay = 1 , 
                  slice_date = None, df_columns = None , device = None):
    '''
    ------------------------ gp labels raw ------------------------
    
    生成原始预测标签,中性化后的10日收益
    input:
        CP:             close price
        neutral_factor: size , for instance
        neutral_group:  indus , for instance
    output:
        labels_raw:     
    '''
    if CP is None:
        CP = df2ts(read_gp_data(gp_filename_converter()('CP'),slice_date,df_columns) , 'CP' , device)    
    labels = MF.ts_delay(MF.pctchg(CP, nday) , -nday-delay)  # t+1至t+11的收益率
    neutral_x = MF.neutralize_xdata_2d(neutral_factor, neutral_group)
    labels_raw = MF.neutralize_2d(labels, neutral_x , inplace=True)  # 市值行业中性化
    return labels_raw

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
                     'dp':'dividendyield2','rtn':'return1','indus':'cs_indus_code'}
    def wrapper(gp_key):
        assert gp_key.isupper() or gp_key.islower() , gp_key

        rawkey = gp_key.lower()
        if rawkey in filename_dict.keys(): 
            rawkey = filename_dict[rawkey]

        zscore = gp_key.islower() and rawkey not in ['cs_indus_code' , 'size']

        return f'{defaults.DIR_data}/{rawkey}' + '_zscore' * zscore + '_day.parquet'
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
        df:          result data
    '''
    df = pd.read_parquet(filename, engine='fastparquet')
    assert isinstance(df , pd.DataFrame) , f'{type(df)} is not a DataFrame'
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if slice_date is not None: 
        df = df[(slice_date[0] <= df.index.values) & (df.index.values <= slice_date[-1])] # 训练集首日至测试集末日
    # if freq!='D': df = df.groupby([pd.Grouper(level=0, freq=freq)]).last()
    if df_columns is not None: 
        df = df.loc[:,df_columns]# 选取指定股票
    if slice_date is None and df_index is not None: 
        df = df.loc[df_index]
    return df

def df2ts(x , gp_key = '' , device = None , share_memory = True):
    # additional treatment based by gp_key
    if isinstance(x , pd.DataFrame): 
        x = torch.FloatTensor(x.values)
    if gp_key == 'DP': # raw dividend factor , nan means 0
        x.nan_to_num_()
    if isinstance(x , torch.Tensor):
        if device is not None: 
            x = x.to(device)
        if share_memory: 
            x.share_memory_() # 执行多进程时使用：将张量移入共享内存
    return x

# %%
def gp_syntax2value(compiler, individual, gp_inputs, param, tensors, i_iter=0, 
                    timer=None, mgr_mem = None , process_stream='inf_winsor_norm',
                    **kwargs):
    '''
    ------------------------ calculate individual syntax factor value ------------------------
    
    根据迭代出的因子表达式,计算因子值
    计算因子时容易出现OutOfMemoryError,如果出现了异常处理一下,所以代码比较冗杂
    input:
        compiler:     compiler function to realize syntax computation, i.e. return factor function of given syntax
        individual:   individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        gp_inputs:    initial population factor values
        param:        gp_params
        tensors:      gp_space components of other tensors
    output:
        factor_value: 2d tensor
    '''
    if mgr_mem  is None: 
        mgr_mem  = MemoryManager()
    if timer    is None: 
        timer    = gpTimer()

    #Logger.stdout(individual)
    with timer.acc_timer('compile'):
        func = compiler(individual)
        
    with timer.acc_timer('eval'):
        func = mgr_mem.except_MemoryError(func, print_str=f'evaluating {str(individual)}')
        factor_value = func(*gp_inputs)

    with timer.acc_timer('process'):
        func = mgr_mem.except_MemoryError(FF.process_factor, print_str=f'processing {str(individual)}')
        factor_value = func(factor_value , process_stream , dim = 1)

    with timer.acc_timer('neutralize'):
        factor_neut_type = param.factor_neut_type * (i_iter > 0) * (tensors.get('neutra') is not None)
        assert factor_neut_type in [0,1,2] , factor_neut_type
        if factor_value is None or factor_neut_type == 0:
            pass
        elif factor_neut_type == 1:
            assert isinstance(factor_value , torch.Tensor) , f'{type(factor_value)} is not a Tensor'
            func = mgr_mem.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
            shape2d = factor_value.shape
            factor_value = func(y = factor_value.reshape(-1) , 
                                x = tensors.neutra.to(factor_value).reshape(-1,tensors.neutra.shape[-1]) , 
                                insample = tensors.insample_2d.reshape(-1))
            if isinstance(factor_value , torch.Tensor): 
                factor_value = factor_value.reshape(shape2d)
        elif factor_neut_type == 2:
            func = mgr_mem.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
            factor_value = func(factor_value , tensors.neutra.to(factor_value))

    return FF.FactorValue(name=individual , process=process_stream , value=factor_value)

# %%
def evaluate(individual, pool_skuname, compiler , gp_inputs , param , tensors , fitness , i_iter = 0, 
             timer = None , mgr_file = None , mgr_mem = None , 
             const_annual = 24 , min_coverage = 0.5 , **kwargs):
    '''
    ------------------------ evaluate individual syntax fitness ------------------------
    
    从因子表达式起步,生成因子并计算适应度
    input:
        individual:     individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        pool_skuname:   pool skuname in pool.imap, e.g. iter0_gen0_0
        compiler:       compiler function to realize syntax computation, i.e. return factor function of given syntax
        gp_inputs:      initial population factor values
        param:          gp_params
        tensors:        gp_space components of other tensors
        fitness:        gpFitness object
        const_annual:   constant of annualization
        min_coverage:   minimum daily coverage to determine if factor is valid
    output:
        tuple of (
            abs_rankir: (abs(insample_res), ) # !! Fitness definition 
            rankir:     (insample_res, outsample_res, insample_raw, outsample_raw)
        )
    '''
    if mgr_file is None: 
        mgr_file = gpFileManager()
    if mgr_mem  is None: 
        mgr_mem  = MemoryManager()
    if timer    is None: 
        timer    = gpTimer()

    mgr_file.update_sku(individual, pool_skuname)
    factor = gp_syntax2value(compiler,individual,gp_inputs,param,tensors,i_iter,timer,**kwargs)
    # mgr_mem.check('factor')
    
    metrics = torch.zeros(8)
    if isinstance(factor.value , torch.Tensor): 
        metrics = metrics.to(factor.value)
    if not factor.isnull(): 
        for i , labels in enumerate([tensors.labels_res , tensors.labels_raw]):
            rankic_full = MF.rankic_2d(factor.value , labels , dim = 1 , universe = tensors.universe , min_coverage = min_coverage)
            for j , sample in enumerate([tensors.insample , tensors.outsample]):
                if rankic_full is None: 
                    continue
                rankic = rankic_full[sample]
                if rankic.isnan().sum() < 0.5 * len(rankic): # if too many nan rank_ic (due to low coverage)
                    rankic_avg  = rankic.nanmean()
                    rankic_std  = (rankic - rankic_avg).square().nanmean().sqrt() 
                    metrics[4*i + 2*j + 0] = rankic_avg.item() * const_annual
                    metrics[4*i + 2*j + 1] = (rankic_avg / (rankic_std + 1e-6) * np.sqrt(const_annual)).item()
    factor.infos['metrics'] = metrics.cpu().numpy()

    individual.if_valid = not factor.isnull()
    individual.metrics  = factor.infos['metrics']
    individual.fitness.values = fitness.fitness_value(factor.infos['metrics'] , as_abs=True)      
    # mgr_mem.check('rankic')
    return individual

def evaluate_pop(population , toolbox , param , i_iter = 0, i_gen = 0, desc = 'Evolve Generation' , **kwargs):
    '''
    ------------------------ evaluate entire population ------------------------
    
    计算整个种群的适应度
    input:
        population:     un-updated population of syntax
        toolbox:        toolbox that contains all gp utils
        param:          gp_params
    output:
        population:     updated population of syntax
    '''
    changed_pop   = [ind for ind in population if not ind.fitness.valid]
    pool_skunames = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(changed_pop))] # 'pool_skuname' arg for evaluate
     # record max_fit0 and show
    iterator = tuple(zip(changed_pop, pool_skunames))
    if param.pool_num > 1:
        with Pool(processes=param.pool_num) as pool:
            _ = list(tqdm(
                pool.starmap(toolbox.evaluate, iterator), total=len(iterator) , 
                desc=f'Parallel evalute population ({param.pool_num})'))
    else:
        def desc0(x):
            return (f'  --> {desc} {str(i_gen)} ' +'MaxIRres{:+.2f}|MaxIRraw{:+.2f}'.format(*x))
        def maxir(ir , ind):
            if abs(ind.metrics[1]) > abs(ir[0]): 
                ir[0] = ind.metrics[1] 
            if abs(ind.metrics[5]) > abs(ir[1]): 
                ir[1] = ind.metrics[5]
            return ir
        iterator = tqdm(iterator, total=len(changed_pop))
        ir = [0.,0.]
        for ind , sku in iterator:
            ind = toolbox.evaluate(ind, sku)
            iterator.set_description(desc0(ir := maxir(ir,ind)))

        #iterator = tqdm(toolbox.map(toolbox.evaluate, changed_pop, pool_skunames), total=len(changed_pop))
        #[iterator.set_description(desc0(maxir := maxir_(maxir,ind))) for ind in iterator]

    assert all([ind.fitness.valid for ind in population])
    return population

def gp_residual(param , tensors : gpContainer, i_iter = 0, device = None , 
                mgr_file=None,mgr_mem=None, **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    
    计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化
    input:
        param:          gp_params
        tensors:        gp_space components of other tensors
        i_iter:         i of outer loop
    output:
    '''
    assert param.labels_neut_type in ['svd' , 'all'] , param.labels_neut_type #  'all'
    assert param.svd_mat_method in ['coef_ts' , 'total'] , param.svd_mat_method

    if mgr_file is None: 
        mgr_file = gpFileManager()
    if mgr_mem  is None: 
        mgr_mem  = MemoryManager()

    tensors.neutra = None
    if i_iter == 0:
        labels_res = copy.deepcopy(tensors.labels_raw)
        elites     = None
        
    else:
        labels_res = mgr_file.load_state('res' , i_iter - 1 , device = device)
        elites     = mgr_file.load_state('elt' , i_iter - 1 , device = device)
    neutra = elites

    if isinstance(elites , torch.Tensor) and param.labels_neut_type == 'svd': 
        assert isinstance(labels_res , torch.Tensor) , type(labels_res)
        if param.svd_mat_method == 'total':
            elites_mat = FF.factor_coef_total(elites[tensors.insample],dim=-1)
        else:
            elites_mat = FF.factor_coef_with_y(elites[tensors.insample], labels_res[tensors.insample].unsqueeze(-1), corr_dim=1, dim=-1)
        neutra = FF.top_svd_factors(elites_mat, elites, top_n = param.svd_top_n ,top_ratio=param.svd_top_ratio, dim=-1 , inplace = True) # use svd factors instead
        Logger.stdout(f'  -> Elites({elites.shape[-1]}) Shrink to SvdElites({neutra.shape[-1]})')

    if isinstance(neutra , torch.Tensor) and neutra.numel(): 
        tensors.update(neutra = neutra.cpu())
        Logger.stdout(f'  -> Neutra has {neutra.shape[-1]} Elements')

    assert isinstance(labels_res , torch.Tensor) , type(labels_res)
    labels_res = MF.neutralize_2d(labels_res, neutra , inplace = True) 
    mgr_file.save_state(labels_res, 'res', i_iter) 

    if param.factor_neut_type > 0 and param.labels_neut_type == 'svd':
        lastneutra = None if i_iter == 0 else mgr_file.load_state('neu' , i_iter - 1 , device = device)
        if isinstance(lastneutra , torch.Tensor): 
            lastneutra = lastneutra.cpu()
        if isinstance(neutra , torch.Tensor): 
            lastneutra = torch.cat([lastneutra , neutra.cpu()] , dim=-1) if isinstance(lastneutra , torch.Tensor) else neutra.cpu()
        mgr_file.save_state(lastneutra , 'neu', i_iter) 
        del lastneutra
    
    tensors.update(labels_res = labels_res)
    mgr_mem.check(showoff = True)

# %%
def gp_population(toolbox , pop_num , max_round = 100 , last_gen = None , forbidden = None , **kwargs):
    '''
    ------------------------ create gp toolbox ------------------------
    
    初始化种群
    input:
        toolbox:        toolbox that contains all gp utils
        pop_num:        population number
        max_round:      max iterations to approching 99% of pop_num
        last_gen:       starting population
        forbidden:      all individuals with null return (and those in the hallof fame) 
    output:
        population:     initialized population of syntax
    '''
    last_gen = last_gen or []
    forbidden = forbidden or []
    if last_gen: 
        population = toolbox.indpop_prune(last_gen)
        population = toolbox.deduplicate(population , forbidden = forbidden)
    else:
        population = []

    for _ in range(max_round):
        new_comer  = toolbox.population(n = int(pop_num * 1.2) - len(population))
        new_comer  = toolbox.indpop_prune(new_comer)
        new_comer  = toolbox.deduplicate(new_comer , forbidden = population + forbidden) 
        population = population + new_comer[:pop_num - len(population)]
        if len(population) >= 0.99 * pop_num: 
            break
    return population


# %%
def gp_evolution(toolbox , param , records , i_iter = 0, start_gen=0, forbidden_lambda = None , 
                 mgr_file=gpFileManager(), timer=gpTimer(),verbose=__debug__,stats=None,**kwargs):  
    """
    ------------------------ Evolutionary Algorithm simple ------------------------
    
    变异/进化[小循环],从初始种群起步计算适应度并变异,重复n_gen次
    input:
        toolbox:            toolbox that contains all gp utils
        param:              gp_params
        records:            gp_space components of records (require forbidden)
        i_iter:             i of outer loop
        start_gen:          which gen to start, if None start a new
        forbidden_lambda:   any function to determine if the result of evaluation indicate forbidden syntax
                            for instance: forbidden_lambda = lambda x:all(i for i in x)
    output:
        population:         updated population of syntax
        halloffame:         container of individuals with best fitness (no more than hof_num)
        forbidden:          all individuals with null return or goes in the the hof once
    """
    population , halloffame , forbidden = mgr_file.load_generation(i_iter , start_gen-1 , hof_num = param.hof_num , stats = stats)
    forbidden += records.get('forbidden' , [])

    for i_gen in range(start_gen, param.n_gen):
        if verbose and i_gen > 0: 
            Logger.stdout(f'  --> Survive {len(population)} Offsprings, try Populating to {param.pop_num} ones')
        population = gp_population(toolbox , param.pop_num , last_gen = population , forbidden = forbidden + [ind for ind in halloffame])
        #Logger.stdout([str(ind) for ind in population[:20]])
        population = toolbox.indpop2syxpop(population)
        #Logger.stdout([str(ind) for ind in population[:20]])
        if verbose and i_gen == 0: 
            Logger.stdout(f'**A Population({len(population)}) has been Initialized')

        # Evaluate the new population
        population = toolbox.evaluate_pop(population , i_iter = i_iter, i_gen = i_gen , pool_num = param.pool_num , **kwargs)

        # check survivors
        survivors  = [ind for ind in population if ind.if_valid]
        forbidden += [ind for ind in population if not ind.if_valid or (False if forbidden_lambda is None else forbidden_lambda(ind.fitness.values))]
        # Update HallofFame with survivors 
        halloffame.update(survivors)

        # Selection of population to pass to next generation, consider surv_rate
        select_offspring = getattr(toolbox , f'select_{param.select_offspring}')
        
        if param.select_offspring in ['Tour' , '2Tour']: 
            # '2Tour' will incline to choose shorter ones
            # around 49% will survive
            k = len(population)
        else: 
            k = min(int(param.surv_rate * param.pop_num) , len(population))
        offspring = select_offspring(population , k)
        offspring = toolbox.syxpop2indpop(list(set(offspring)))

        # Variation offsprings
        with timer.acc_timer('varAnd'):
            population = varAnd(offspring, toolbox, param.cxpb , param.mutpb) # varAnd means variation part (crossover and mutation)
        # Dump population , halloffame , forbidden in logbooks of this generation
        mgr_file.dump_generation(population, halloffame, forbidden , i_iter , i_gen , **(stats.compile(survivors) if stats else {}))

    Logger.stdout(f'**A HallofFame({len(halloffame)}) has been ' + ('Loaded' if start_gen >= param.n_gen else 'Evolutionized'))
    return population, halloffame, forbidden

# %%
def gp_selection(toolbox,evolve_result,gp_inputs,param,records,tensors,i_iter=0,
                 mgr_file=None,timer=None,mgr_mem=None,device=None,
                 test_code=False,verbose=__debug__,**kwargs):
    """
    ------------------------ gp halloffame evaluation ------------------------
    
    筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英
    input:
        toolbox:            toolbox that contains all gp utils
        evolve_result:      population,halloffame,forbidden
        gp_inputs:          initial population factor values
        param:              gp_params
        records:            gp_space components of records (require forbidden)
        tensors:            gp_space components of other tensors
        i_iter:             i of outer loop
    output:
    """
    if mgr_file is None: 
        mgr_file = gpFileManager()
    if mgr_mem  is None: 
        mgr_mem  = MemoryManager()
    if timer    is None: 
        timer    = gpTimer()
    
    population , halloffame , forbidden = evolve_result
    elite_log  = mgr_file.load_state('elitelog' , i_iter = i_iter) # 记录精英列表
    hof_log    = mgr_file.load_state('hoflog'   , i_iter = i_iter) # 记录名人堂状态列表
    hof_elites = gpEliteGroup(start_i_elite = len(elite_log) , device=device).assign_logs(hof_log = hof_log , elite_log = elite_log)
    infos   = pd.DataFrame(
        [[i_iter,-1,gpHandler.ind2str(ind),ind.if_valid,False,0.] for ind in halloffame] , 
        columns = pd.Index(['i_iter','i_elite','syntax','valid','elite','max_corr']))
    metrics = pd.DataFrame([getattr(ind,'metrics') for ind in halloffame] , columns = param.fitness_wgt.keys()) #.reset_index(drop=True)
    new_log = pd.concat([infos , metrics] , axis = 1)

    ir_floor = param.ir_floor * (param.ir_floor_decay**i_iter)
    new_log.elite = (
        new_log.valid & # valid factor value
        (new_log.rankir_in_res.abs() > ir_floor) & # higher rankir_res than threshold
        (new_log.rankir_in_raw.abs() > ir_floor) & # higher rankir_res than threshold
        (new_log.rankir_out_res != 0.))
    Logger.stdout(f'**HallofFame({len(halloffame)}) Contains {new_log.elite.sum()} Promising Candidates with RankIR >= {ir_floor:.2f}')
    if new_log.elite.sum() <= 0.1 * len(halloffame):
        # Failure of finding promising offspring , check if code has bug
        Logger.stdout(f'  --> Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ')
        Logger.stdout(f'  --> Valid Hof({new_log.valid.sum()}), insample max ir({new_log.rankir_in_res.abs().max():.4f})')

    for i , hof in enumerate(halloffame):
        # 若超过了本次循环的精英上限数,则后面的都不算,等到下一个循环再来(避免内存溢出)
        if hof_elites.i_elite - hof_elites.start_i_elite >= param.elite_num: 
            new_log.loc[i,'elite'] = False 
        if not new_log.loc[i,'elite']: 
            continue

        # 根据迭代出的因子表达式,计算因子值, 错误则进入下一循环
        factor_value = gp_syntax2value(toolbox.compile,hof,gp_inputs,param,tensors,i_iter,timer,**kwargs)
        mgr_mem.check('factor')
        
        new_log.loc[i,'elite'] = not factor_value.isnull()
        if not new_log.loc[i,'elite']: 
            continue

        # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
        corr_values , exit_state = hof_elites.max_corr_with_me(factor_value, param.corr_cap, dim_valids=(tensors.insample,None), syntax = new_log.syntax[i])
        new_log.loc[i,'max_corr'] = round(corr_values[corr_values.abs().argmax()].item() , 4)
        new_log.loc[i,'elite'] = not exit_state
        mgr_mem.check('corr')

        if not new_log.loc[i,'elite']: 
            continue

        # 通过检验,加入因子库
        new_log.loc[i,'i_elite'] = hof_elites.i_elite
        forbidden.append(halloffame[i])
        hof_elites.append(factor_value , IR = new_log.rankir_in_res[i] , Corr = new_log.max_corr[i] , starter=f'  --> Hof{i:_>3d}/')
        if False and test_code: 
            mgr_file.save_state(factor_value.to_dataframe(index = records.df_index , columns = records.df_columns) , 'parquet' , i_iter , i_elite = hof_elites.i_elite)
        mgr_mem.check(showoff = verbose and test_code, starter = '  --> ')

    mgr_file.dump_generation(population, halloffame, forbidden , i_iter = i_iter , i_gen = -1)
    records.update(forbidden = forbidden)

    # 记录本次运行的名人堂与精英状态
    hof_elites.update_logs(new_log)
    mgr_file.save_state(hof_elites.elite_log.round(6) , 'elitelog' , i_iter)
    mgr_file.save_state(hof_elites.hof_log.round(6)   , 'hoflog'   , i_iter)
    
    elites = hof_elites.compile_elite_tensor(device=device)
    if elites is not None:
        mgr_file.save_state(elites , 'elt' , i_iter)
        Logger.stdout(f'**An EliteGroup({elites.shape[-1]}) has been Selected')
    if True: 
        Logger.stdout(f'  --> Cuda Memories of "gp_inputs" take {MemoryManager.object_memory(gp_inputs):.4f}G')
        Logger.stdout(f'  --> Cuda Memories of "elites"    take {MemoryManager.object_memory(elites):.4f}G')
        Logger.stdout(f'  --> Cuda Memories of "tensors"   take {MemoryManager.object_memory(tensors):.4f}G')

# %%
def outer_loop(i_iter , gp_space , start_gen = 0):
    """
    ------------------------ gp outer loop ------------------------
    
    一次[大循环]的主程序,初始化种群、变异、筛选、更新残差labels
    input:
        i_iter:   i of outer loop
        gp_space:  dict that includes gp parameters, gp datas and gp arguements, and various other datas
    """
    init_time = datetime.now()
    gp_space.i_iter = i_iter

    '''更新残差标签'''
    with gp_space.timer('Residual' , print_str = f'**Update Residual Labels' , memory_check = True):
        gp_residual(**gp_space)
        
    '''初始化遗传规划Toolbox'''
    with gp_space.timer('Setting' , print_str = f'**Initialize GP Toolbox'):
        toolbox = gpHandler.Toolbox(eval_func = evaluate , eval_pop = evaluate_pop , **gp_space)
        gp_space.mgr_file.update_toolbox(toolbox)

    '''进行进化与变异,生成种群、精英和先祖列表'''
    with gp_space.timer('Evolution' , print_str = f'**{gp_space.param.n_gen - start_gen} Generations of Evolution' , memory_check = True):
        evolve_result = gp_evolution(toolbox , start_gen = start_gen , **gp_space)  #, start_gen=6   #algorithms.eaSimple
    
    '''衡量精英,筛选出符合所有要求的精英中的精英'''
    with gp_space.timer('Selection' , print_str = f'**Selection of HallofFame' , memory_check = True):
        gp_selection(toolbox , evolve_result, **gp_space)
            
    gp_space.timer.append_time('AvgVarAnd' , gp_space.timer.acc_timer('varAnd').avgtime(pop_out = True))
    gp_space.timer.append_time('AvgCompile', gp_space.timer.acc_timer('compile').avgtime(pop_out = True))
    gp_space.timer.append_time('AvgEval',    gp_space.timer.acc_timer('eval').avgtime(pop_out = True))
    gp_space.timer.append_time('All' , (datetime.now() - init_time).total_seconds())
    return

class gpGenerator:
    '''
    ------------------------ gp factor generator ------------------------
    
    构成因子生成器,返回输入因子表达式则输出历史因子值的函数
    input:
        kwargs:  specific gp parameters, suggestion is to leave it alone
    output:
        GP:      gp_generator
    '''
    def __init__(self , job_id , 
                 process_key : str = 'inf_winsor_norm' ,  
                 weight_scheme : Literal['ic' , 'ir' , 'ew'] = 'ic', 
                 window_type  : Literal['insample' , 'rolling'] = 'rolling', 
                 weight_decay : Literal['constant' , 'linear' , 'exp'] = 'exp' , 
                 ir_window : int = 40 , 
                 roll_window : int = 40 , 
                 halflife : int = 20 , 
                 min_coverage :float = 0.1 , 
                 **kwargs) -> None:
        gp_params = gp_parameters(job_id = job_id , train = False , **kwargs)
        self.gp_space  = gp_namespace(gp_params)
        self.toolbox   = gpHandler.Toolbox(eval_func=evaluate , eval_pop=evaluate_pop , **self.gp_space)
        self.process_key = process_key
        self.elitelog   = self.gp_space.mgr_file.load_state('elitelog' , i_iter = -1).set_index('i_elite')
        self.df_axis    = self.gp_space.mgr_file.load_state('df_axis' , -1)

        self.Ensembler = FF.MultiFactor(
            universe      = self.gp_space.tensors.universe , 
            insample      = self.gp_space.tensors.insample ,
            weight_scheme = weight_scheme ,
            window_type   = window_type ,
            weight_decay  = weight_decay ,
            ir_window     = ir_window ,
            roll_window   = roll_window ,
            halflife      = halflife ,
            min_coverage  = min_coverage ,
            **kwargs)

    def __call__(self, syntax : str | FF.FactorValue , process_key : str | None = None , as_df = False , print_info = True) -> FF.FactorValue:
        '''
        Calcuate FactorValue of a syntax
        '''
        if isinstance(syntax , FF.FactorValue): 
            return syntax
        if process_key is None: 
            process_key = self.process_key
        factor = gp_syntax2value(getattr(self.toolbox , 'compile'),syntax,**self.gp_space)
        if as_df and not factor.isnull():
            factor.value = factor.to_dataframe(index = self.df_axis['df_index'] , columns = self.df_axis['df_columns'])
        if print_info: 
            Logger.stdout(f'gpGenerator -> process_key : {process_key} , syntax : {syntax}')
        return factor

    def entire_elites(self , verbose = True , block_len = 50 , process_key = None):
        '''
        Load all elite factors
        '''
        elite_log  = self.gp_space.mgr_file.load_state('elitelog' , i_iter = -1) 
        hof_log    = self.gp_space.mgr_file.load_state('hoflog'   , i_iter = -1)
        hof_elites = gpEliteGroup(start_i_elite=0 , device=self.gp_space.device , block_len=block_len).assign_logs(hof_log=hof_log, elite_log=elite_log)
        for elite in elite_log.syntax: 
            hof_elites.append(self(elite , process_key = process_key , print_info = verbose))
        hof_elites.cat_all()
        Logger.stdout(f'Load {hof_elites.total_len()} Elites')
        return hof_elites

    def load_elite(self , i_elite : int , factor = False):
        '''
        Load a single elite factor
        '''
        elite_syntax = self.elitelog.loc[i_elite , 'syntax']
        return self(elite_syntax) if factor else elite_syntax
    
    def multi_factor(self , *factors , labels = None , **kwargs):
        '''
        Calculate MultiFactorValue given factors and labels
        None kwargs will use default values
        '''
        factor_list = list(factors)
        if len(factor_list) == 1 and isinstance(factor_list[0] , torch.Tensor) and factor_list[0].dim() == 3:
            factor = factor_list[0]
        else:
            for i , fac in enumerate(factor_list):
                if isinstance(fac , str): 
                    factor_list[i] = self(fac)
            factor = torch.stack(factor_list , dim = -1)
        if labels is None: 
            labels = self.gp_space.tensors.labels_raw
        metrics = self.Ensembler.calculate_icir(factor , labels , **kwargs) # ic,ir
        multi = self.Ensembler.multi_factor(factor , **metrics , **kwargs)
        return multi # multi对象拥有multi,weight,inputs三个自变量
    
    def multi_elite_factor(
            self , elites = None , verbose = True , 
            process_key : str | None = None ,  
            weight_scheme : Literal['ic' , 'ir' , 'ew'] | None = None, 
            window_type  : Literal['insample' , 'rolling'] | None = None, 
            weight_decay : Literal['constant' , 'linear' , 'exp'] | None = None , 
            ir_window : int | None = None , 
            roll_window : int | None = None , 
            halflife : int | None = None , 
            min_coverage :float | None = None , 
            **kwargs):
        '''
        Load all elite factors and calculate MultiFactorValue.
        None kwargs will use default values
        '''
        if elites is None:
            elites = self.entire_elites(verbose = verbose , process_key = process_key)
        if isinstance(elites , gpEliteGroup):
            elites = elites.compile_elite_tensor()
        assert isinstance(elites , torch.Tensor) , type(elites)
        multi = self.multi_factor(
            elites , labels = self.gp_space.tensors.labels_raw , 
            weight_scheme = weight_scheme , window_type = window_type ,weight_decay = weight_decay ,
            ir_window = ir_window , roll_window = roll_window , halflife = halflife , min_coverage = min_coverage , **kwargs)
        return multi

# %%
def main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False , **kwargs):
    """
    ------------------------ gp main process ------------------------
    
    训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
    input:
        job_id:    when test_code is not True, determines job_dir = f'{defaults.DIR_pop}/{job_id}'   
        start_iter , start_gen: when to start, any of them has positive value means continue training
        noWith:    to shutdown all timers (with xxx expression)
    output:
        pfr:       profiler to record time cost of each function (only available in test_code model)
    """
    defaults.noWith = noWith
    with Profiler(doso = test_code) as pfr:
        time0 = datetime.now()
        
        gp_params = gp_parameters(job_id , continuation = start_iter>0 or start_gen>0 , test_code = test_code , **kwargs)
        gp_space = gp_namespace(gp_params)

        for i_iter in range(start_iter , gp_space.param.n_iter):
            Logger.stdout('=' * 20 + f' Iteration {i_iter} start from Generation {start_gen * (i_iter == start_iter)} ' + '=' * 20)
            outer_loop(i_iter , gp_space , start_gen = start_gen * (i_iter == start_iter))

    hours, secs = divmod((datetime.now() - time0).total_seconds(), 3600)
    Logger.stdout('=' * 20 + f' Total Time Cost :{hours:.0f} hours {secs/60:.1f} ' + '=' * 20)
    gp_space.mgr_file.save_state(gp_space.timer.time_table(showoff=True) , 'runtime' , 0)
    gp_space.mgr_mem.print_memeory_record()
    pfr.get_df(output = kwargs.get('profiler_out' , 'cprofile.csv')) 
    return pfr


if __name__ == '__main__':
    main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False)
