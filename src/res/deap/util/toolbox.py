from typing import Callable
from deap import base , creator , tools , gp
import operator

from src.res.deap.param import gpParameters
from .status import gpStatus
from .syntax import BaseIndividual
from .primative import Primative
from .fitness import FitnessObjectMin
from .evaluator import gpEvaluator

class BaseToolbox(base.Toolbox):
    """
    创建遗传算法基础模块Toolbox,以下参数不建议更改,如需更改,可参考deap官方文档
    https://zhuanlan.zhihu.com/p/72130823
    """
    create_individual : Callable[... , BaseIndividual]
    create_population : Callable[... , list[BaseIndividual]]

    def __init__(self , param : gpParameters , status : gpStatus , evaluator : gpEvaluator , **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.status = status
        self.evaluator = evaluator
        
        pset_raw , pset_pur = Primative.GetPrimSets(param.n_args , param.argnames , '_' if status.i_iter < 0 else f'_{status.i_iter}')
    
        [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual']]
        fit_weights = evaluator.fitness.fitness_weight() if evaluator.fitness is not None else (+1.0,)
        creator.create('FitnessMin', FitnessObjectMin, weights=fit_weights)   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
        creator.create('Individual', BaseIndividual, fitness=getattr(creator , 'FitnessMin'), 
                       pset_raw = pset_raw , pset_pur = pset_pur) 
        
        self.register('generate_expr', gp.genHalfAndHalf, pset=pset_raw, min_=1, max_= param.max_depth)
        self.register('create_individual', tools.initIterate, BaseIndividual.get_class(), getattr(self , 'generate_expr'))
        # toolbox.register('fitness_value', fitness.fitness_value)
        self.register('create_population', tools.initRepeat, list, getattr(self , 'create_individual')) 
        self.register('mate', gp.cxOnePoint)
        self.register('expr_mut', gp.genHalfAndHalf, pset=pset_raw , min_=0, max_= param.max_depth)  # genFull
        self.register('mutate', gp.mutUniform, expr = getattr(self , 'expr_mut') , pset=pset_raw) 
        self.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3
        self.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3