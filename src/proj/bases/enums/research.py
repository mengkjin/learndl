"""basic types for the project's research purposes"""
from __future__ import annotations

from src.proj.core.types import StrEnum

__all__ = [
    'ModuleType' , 'TestType' , 'FmpType' , 
    'FittingEventType' , 'PortOptimProblem' , 'PortOptimEngine' , 
    'PortOptimCvxpySolver']

class ModuleType(StrEnum):
    NN = 'nn'
    BOOST = 'boost'
    FACTOR = 'factor'
    INVALID = ''

class TestType(StrEnum):
    FACTOR = 'factor'
    OPTIM = 'optim'
    TOP = 'top'
    T50 = 't50'
    SCREEN = 'screen'
    REVSCREEN = 'revscreen'
    REINFORCE = 'reinforce'

    @classmethod
    def fmp_values(cls) -> list[TestType]:
        return [m for m in cls if m != cls.FACTOR]

    def title(self) -> str:
        if self == TestType.FACTOR:
            return self.value.title()
        else:
            return f'{self.value.title()} Port'

class FmpType(StrEnum):
    TOP = 'top'
    T50 = 't50'
    SCREEN = 'screen'
    REVSCREEN = 'revscreen'
    REINFORCE = 'reinforce'
    OPTIM = 'optim'

class FittingEventType(StrEnum):
    NEW_ATTEMPT = 'new_attempt'
    REDO_ATTEMPT = 'redo_attempt'
    END_ATTEMPT = 'end_attempt'
    RECALL_CKPT = 'recall_ckpt'
    NEW_PHASE_RECALL = 'new_phase_recall'
    NEW_PHASE = 'new_phase'
    MILESTONE = 'milestone'
    LOGGING = 'logging'

class PortOptimProblem(StrEnum):
    LINPROG = 'linprog'
    QUADPROG = 'quadprog'
    SOCP = 'socp'

class PortOptimEngine(StrEnum):
    MOSEK = 'mosek'
    CVXOPT = 'cvxopt'
    CVXPY = 'cvxpy'

class PortOptimCvxpySolver(StrEnum):
    MOSEK = 'mosek'
    ECOS = 'ecos'
    OSQP = 'osqp'
    SCS = 'scs'
    CLARABEL = 'clarabel'
