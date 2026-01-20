from .mosek import Solver as MosekSolver
from .cvxpy import Solver as CvxpySolver
from .setup import setup_solvers

setup_solvers()