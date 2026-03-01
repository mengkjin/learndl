from typing import Callable , Sequence , Any
from deap import base , creator , tools , gp
import operator

from src.res.deap.param import gpParameters
from .status import gpStatus
from .syntax import BaseIndividual , SyntaxRecord
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

        pset_raw , pset_pur = Primative.GetPrimSets(param.n_args , param.gp_argnames , '_' if status.i_iter < 0 else f'_{status.i_iter}')
    
        [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual' , 'Syntax']]
        fit_weights = evaluator.fitness.fitness_weight() if evaluator.fitness is not None else (+1.0,)
        creator.create('FitnessMin', FitnessObjectMin, weights=fit_weights)   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
        creator.create('Individual', BaseIndividual, fitness=getattr(creator , 'FitnessMin'), 
                       pset_raw = pset_raw , pset_pur = pset_pur) 
        
        self.register('generate_expr', gp.genHalfAndHalf, pset=pset_raw, min_=1, max_= param.max_depth)
        self.register('create_individual', tools.initIterate, BaseIndividual.get_class(), getattr(self , 'generate_expr'))
        # toolbox.register('fitness_value', fitness.fitness_value)
        self.register('create_population', tools.initRepeat, list, getattr(self , 'create_individual')) 
        self.register('select_nsga2', tools.selNSGA2) 
        self.register('select_best', tools.selBest) 
        self.register('select_Tour', tools.selTournament, tournsize=3) # 锦标赛：随机选择3个，取最大, resulting around 49% of pop
        self.register('select_2Tour', tools.selDoubleTournament, fitness_size=3 , parsimony_size=1.4 , fitness_first=True) # 锦标赛：第一轮随机选择3个，取最大
        self.register('mate', gp.cxOnePoint)
        self.register('expr_mut', gp.genHalfAndHalf, pset=pset_raw , min_=0, max_= param.max_depth)  # genFull
        self.register('mutate', gp.mutUniform, expr = getattr(self , 'expr_mut') , pset=pset_raw) 
        self.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3
        self.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3
    
    def str2ind(self , x : str , fit_value = None) -> BaseIndividual:
        return self.to_ind(x , fit_value = fit_value)

    def ind2str(self , ind : BaseIndividual) -> str:
        return ind.syntax

    def to_ind(self , syntax : BaseIndividual | str | SyntaxRecord , **kwargs) -> BaseIndividual:
        if isinstance(syntax , str):
            return BaseIndividual.get_class().from_syntax(syntax , **kwargs)
        elif isinstance(syntax , BaseIndividual):
            return syntax
        elif isinstance(syntax , SyntaxRecord):
            return syntax.to_ind(**kwargs)
        else:
            raise ValueError(f'Invalid input type: {type(syntax)}')

    def to_record(self , ind : BaseIndividual | str | SyntaxRecord) -> SyntaxRecord:
        return SyntaxRecord.create(ind)

    def prune_ind(self , ind : BaseIndividual | str | SyntaxRecord) -> BaseIndividual:
        return self.to_ind(ind).prune()

    def to_indpop(self , population : Sequence[BaseIndividual | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.to_ind(ind) for ind in population]

    def prune_pop(self , population : Sequence[BaseIndividual | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.prune_ind(ind) for ind in population]

    def deduplicate(self , population : Sequence[BaseIndividual] , forbidden : list[Any] | None = []) -> list[BaseIndividual]:
        # return the unique population excuding specific ones (forbidden)
        ori = [ind.pure_syntax for ind in population]
        fbd = [str(ind) for ind in forbidden] if forbidden else []
        allowed = [ind not in fbd for ind in ori]
        return [ind for ind , allowed in zip(population , allowed) if allowed]

    def purify_pop(self , population : Sequence[BaseIndividual]) -> list[BaseIndividual]:
        # remove Identity primatives of population
        return [ind.purify() for ind in population]
        
    def revert_pop(self , population : Sequence[BaseIndividual]) -> list[BaseIndividual]:
        # revert purified individuals to raw individuals
        return [ind.revert() for ind in population]