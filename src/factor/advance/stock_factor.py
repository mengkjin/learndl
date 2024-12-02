from typing import Literal

from src.factor.analytic import FactorPerfManager
from src.factor.analytic.factor_perf.calculator import BasePerfCalc
from src.factor.util import StockFactor

class AdvanceStockFactor(StockFactor):   
    TASK_LIST = FactorPerfManager.TASK_LIST
    def __init__(self , *args , **kwargs):
        super().__init__(*args , **kwargs)
        self.tasks : dict[str , BasePerfCalc] = {}
    
    def select_analytic(self , task_name : str):
        match_tasks = [task for task in self.TASK_LIST if task.match_name(task_name)]
        assert match_tasks , f'no match tasks : {task_name}'
        assert len(match_tasks) <= 1, f'Duplicate match tasks: {match_tasks}'
        use_name = match_tasks[0].__name__
        if use_name not in self.tasks:
             self.tasks[use_name] = match_tasks[0]()
        return self.tasks[use_name]

    def analyze(self , 
                task_name : Literal['FrontFace', 'IC_Curve', 'IC_Decay', 'IC_Indus',
                                    'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                                    'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                                    'Group_Year','Distrib_Curve'] | str , plot = True , display = True):
        task = self.select_analytic(task_name)
        task.calc(self)
        if plot: task.plot(show = display)
        return self
    
    