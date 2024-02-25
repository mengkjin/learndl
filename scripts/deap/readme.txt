---------环境配置要求---------------
deap==1.3.1，不建议采用1.4版本（可能会报错）。使用pip安装即可：pip install deap==1.3.1
【setuptools>58环境下直接安装报错error: subprocess-exited-with-error；参考https://pypi.org/project/deap/，先降级setuptools再安装deap：pip install setuptools==57.5.0】
torch>=1.12.0，并确保cuda可用。安装指南见https://pytorch.org/get-started/previous-versions/
parquet文件读取需要安装fastparquet，版本不限，使用pip或conda均可安装。


------------文件说明---------------
main.py				遗传算法【主程序】，按需修改44行-94行之间的参数，即可运行
math_func_gpu.py		存储数学算子对应函数，如rolling、correlation、grouping等
------
GP_gpu_demo.ipynb	几个demo，对比加速效果
------
good_log.csv			新因子的因子表达式及IR值


---------main程序所需输入---------------
高开低收等基准因子dataframe（以parquet存储，格式为日期*股票），目前已存储在'./data/features/parquet'路径下，程序可直接读取


---------main程序输出结果---------------
'./pop/{job_id}/good_log.csv'		新因子的因子表达式及IR值
'./pop/{job_id}/factor/xxxx.parquet'	每个新因子的具体数值（日期*股票格式）
'./pop/{job_id}/'下的其他文件		无实际含义，主要为log文件，用于记录程序运行时长


---------对比gpu与cpu运行速度的两种方法---------------
方法一：将main.py内第30行的decive('cuda')改为device('cpu')
方法二：运行GP_gpu_demo.ipynb，内有几个写好的demo