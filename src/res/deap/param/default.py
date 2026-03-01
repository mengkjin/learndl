import platform , os
import torch
from src.proj import PATH
from src.proj.util import Device

class gpDefaults:
    plat = platform.system().lower() # Windows or Linux
    device = Device.use_device() # cuda / mps cpu
    test_code = True or not torch.cuda.is_available() # 是否只是测试代码有无bug,默认False
    dir_data = PATH.interim.joinpath('gp_data' , 'features' , 'parquet') # input路径1,原始因子所在路径
    dir_pack = PATH.interim.joinpath('gp_data' , 'package') # input路径2,加载原始因子后保存pt文件加速存储
    dir_pop = PATH.result.joinpath('gp') # output路径,即保存因子库、因子值、因子表达式的路径
    path_param = PATH.conf.joinpath('algo' , 'gp' , 'params.yaml') # 保存测试参数的地址 
    encoding = 'utf-8' # 带中文的encoding
    worker_num = 1 # 单机多进程设置,默认不并行.若并行,通信成本过高,效率提升不大
    job_id = -1 # 默认不指定job_id,自动生成

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                                # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
if gpDefaults.worker_num > 1:
    # multiprocessing method, 单机多进程设置(实际未启用),参考https://zhuanlan.zhihu.com/p/600801803
    torch.multiprocessing.set_start_method('spawn' if gpDefaults.plat == 'windows' else 'forkserver', force=True) 

