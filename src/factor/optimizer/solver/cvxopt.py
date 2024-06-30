import numpy as np
import cvxopt as cvx

from typing import Any , Literal
from src.factor.optimizer.util import SolverInput , SolveCond , SolveVars
from src.factor.basic.var import SYMBOL_INF as INF

_SOLVER_PARAM = {'show_progress': False}

class Solver:
    ...