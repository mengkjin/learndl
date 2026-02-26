from src.proj import Logger
from src.res.deap.util import GeneticProgramming

def main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False , **kwargs):
    """
    训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
    input:
        job_id:    when test_code is not True, determines job_dir = f'{gpDefaults.DIR_pop}/{job_id}'   
        start_iter , start_gen: when to start, any of them has positive value means continue training
        noWith:    to shutdown all timers (with xxx expression)
    output:
        pfr:       profiler to record time cost of each function (only available in test_code model)
    """
    with Logger.Profiler(title = 'test_gp' if test_code and not noWith else None , builtins = False , display = True) as pfr:
        gp = GeneticProgramming.main(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , noWith = noWith , **kwargs)
    return gp , pfr


if __name__ == '__main__':
    main(job_id = None , start_iter = 0 , start_gen = 0 , test_code = False , noWith = False)
    