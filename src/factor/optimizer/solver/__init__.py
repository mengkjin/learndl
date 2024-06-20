from .mosek import Solver as MosekSolver

SOLVER_CLASS = {
    'mosek' : MosekSolver
}