from src.proj import PATH , Logger
import torch
import numpy as np
import pandas as pd
import shutil
from src.res.deap.env import gpDefaults

class gpParameters:
    '''
    主要遗传规划参数初始化
    注：为方便跑多组实验,需设置job_id参数,设置方式为: python xxx.py --job_id 123456
    input:
        job_id:         when test_code is not True, determines f'{gpDefaults.DIR_pop}/{job_id}' 
        train:          if True, will first check dirs and device  
        continuation:   if True, will continue on start_iter / start_gen
        test_code:      if only to test code validity
    '''
    def __init__(self , job_id : int | None = None , train : bool = True , continuation : bool = False , test_code : bool = False , **kwargs):
        self.job_id = job_id
        self.train = train
        self.continuation = continuation
        self.kwargs = kwargs
        self.make_job_dir(job_id , test_code)
        self.load_params()

        if self.worker_num > 1:
            torch.multiprocessing.set_start_method('spawn' if gpDefaults.plat == 'windows' else 'forkserver', force=True) 

    def make_job_dir(self , job_id : int | None = None , test_code : bool = False):
        '''
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
            job_id = -1
        if test_code or job_id == 'bendi':
            test_code = True
            job_dir = gpDefaults.dir_pop.joinpath('bendi')
        else:
            job_id = int(job_id)
            if job_id < 0:
                old_job_id = [int(path.name) for path in gpDefaults.dir_pop.iterdir() if path.is_dir() and path.name.isdigit()]
                if self.train:
                    job_id = np.setdiff1d(np.arange(max(old_job_id) + 2) , old_job_id).min() if len(old_job_id) else 0
                else:
                    job_id = max(old_job_id)
            job_dir = gpDefaults.dir_pop.joinpath(str(job_id))
        Logger.stdout(f'**Job Directory is : "{job_dir}"')
        if self.train:
            if job_dir.exists() and not self.continuation: 
                if not test_code and not input(f'Path "{job_dir}" exists , press "yes" to confirm Deletion:')[0].lower() == 'y':
                    raise Exception(f'Deletion Denied!')
                shutil.rmtree(job_dir)
                Logger.stdout(f'Start New Training in "{job_dir}"' , indent = 1)
            elif not job_dir.exists() and self.continuation:
                raise Exception(f'No existing "{job_dir}" for Continuation!')
            else:
                Logger.stdout(f'Continue Training in "{job_dir}"' , indent = 1)
        self.job_dir = job_dir
        self.test_code = test_code

    def load_params(self):
        if self.train:
            if gpDefaults.device.type == 'cpu':
                Logger.stdout('**Cuda not available')
            else:
                Logger.stdout('**Device name:', torch.cuda.get_device_name(gpDefaults.device))

        self.params = PATH.read_yaml(gpDefaults.path_param , encoding = gpDefaults.encoding)
        if self.test_code: 
            self.params.update(self.params.get('test_params' , {}))
        self.params.update(self.kwargs)
    @property
    def device(self):
        """用cuda还是cpu计算"""
        return gpDefaults.device       # training device, cuda or cpu
    @property
    def gp_fac_list(self):
        """作为GP输入时,标准化因子的数量"""
        return self.params.get('gp_fac_list' , [])
    @property
    def gp_raw_list(self):
        """作为GP输入时,原始指标的数量"""
        return self.params.get('gp_raw_list' , [])
    @property
    def worker_num(self):
        """并行任务数量,建议设为1,即不并行,使用单显卡单进程运行"""
        return self.params.get('worker_num' , gpDefaults.worker_num)  # multiprocessing pool number
    @property
    def slice_date(self):
        """修改数据切片区间,前两个为样本内的起止点,后两个为样本外的起止点均需要是交易日"""
        return [pd.to_datetime(x).values for x in self.params.get('slice_date' , [])]
    @property
    def show_progress(self):
        """训练过程是否输出细节信息"""
        return self.params.get('show_progress' , True)
    @property
    def pop_num(self):
        """种群数量,初始化时生成多少个备选公式"""
        return self.params.get('pop_num' , 3000)
    @property
    def hof_num(self):
        """精英数量,一般精英数量设为种群数量的1/6左右即可"""
        return self.params.get('hof_num' , 500)
    @property
    def elite_num(self):
        """精英数量,一般精英数量设为种群数量的1/6左右即可"""
        return self.params.get('elite_num' , 100)
    @property
    def n_iter(self):
        """[大循环]的迭代次数,每次迭代重新开始一次遗传规划、重新创立全新的种群,以上一轮的残差收益率作为优化目标"""
        return self.params.get('n_iter' , 5)
    @property
    def ir_floor(self):
        """[大循环]中因子入库所需的最低rankIR值,低于此值的因子不入库"""
        return self.params.get('ir_floor' , 3.0)
    @property
    def ir_floor_decay(self):
        """[大循环]是否每一轮迭代降低ir的标准,若降低则输入小于1.0的数"""
        return self.params.get('ir_floor_decay' , 1.0)
    @property
    def corr_cap(self):
        """[大循环]中新因子与老因子的最高相关系数,相关系数绝对值高于此值的因子不入库"""
        return self.params.get('corr_cap' , 0.6)
    @property
    def factor_neut_type(self):
        """[大循环]i_iter>0时如何中性化因子 0:不中性化, 1:根据样本内的相关性一口气中性化, 2:每天单独中性化"""
        return self.params.get('factor_neut_type' , 0)
    @property
    def labels_neut_type(self):
        """[大循环]计算残差收益时,怎么使用Elite因子: 'svd' , 'all'"""
        return self.params.get('labels_neut_type' , 'all')
    @property
    def svd_mat_method(self):
        """[大循环]svd factor的矩阵怎么计算: 'total'代表所有日期所有因子值相关矩阵, 'coef_ts'代表所有因子值时序与labels相关性时序的相关矩阵"""
        return self.params.get('svd_mat_method' , 'coef_ts')
    @property
    def svd_top_ratio(self):
        """[大循环]svd factor最小解释力度"""
        return self.params.get('svd_top_ratio' , 0.75)
    @property
    def svd_top_n(self):
        """[大循环]svd factor最少因子数量"""
        return self.params.get('svd_top_n' , 1)
    @property
    def n_gen(self):
        """[小循环]的迭代次数,即每次遗传规划进行几轮繁衍进化"""
        return self.params.get('n_gen' , 6)
    @property
    def max_depth(self):
        """[小循环]中个体算子树的最大深度,即因子表达式的最大复杂度"""
        return self.params.get('max_depth' , 5)
    @property
    def select_offspring(self):
        """[小循环]每次后代如何选择是否进入遗传突变环节,可以是 'best' , '2Tour' , 'Tour' , 'nsga2'"""
        return self.params.get('select_offspring' , 'nsga2')
    @property
    def surv_rate(self):
        """[小循环]上面选best的话,这里输入具体比例"""
        return self.params.get('surv_rate' , 0.6)
    @property
    def cxpb(self):
        """[小循环]中交叉概率,即两个个体之间进行交叉的概率"""
        return self.params.get('cxpb' , 0.35)
    @property
    def mutpb(self):
        """[小循环]中变异概率,即个体进行突变变异的概率"""
        return self.params.get('mutpb' , 0.25)    
    @property
    def fitness_wgt(self):
        """因子评价权重"""
        return self.params.get('fitness_wgt' , {})
