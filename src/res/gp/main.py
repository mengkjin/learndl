from src.proj import Logger
from src.res.gp.util import GeneticProgramming

def main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , timer = True , **kwargs):
    """
    训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
    input:
        job_id:    when test_code is not True, determines job_dir = f'{gpDefaults.DIR_pop}/{job_id}'   
        start_iter , start_gen: when to start, any of them has positive value means continue training
        timer:     to enable gpTimer , default is True
    output:
        pfr:       profiler to record time cost of each function (only available in test_code model)
    """
    with Logger.Profiler(title = 'test_gp' if test_code else None , builtins = False , display = True) as pfr:
        gp = GeneticProgramming.main(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , timer = timer , **kwargs)
    return gp , pfr


if __name__ == '__main__':
    main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , timer = True)
    