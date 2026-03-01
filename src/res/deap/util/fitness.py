from deap import base

class gpFitness:
    _instance = None
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , fitness_weights : dict | None = None ,  **kwargs) -> None:
        self.initialize(fitness_weights , **kwargs)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'title')

    def initialize(self , fitness_weights : dict | None = None , **kwargs) -> None:
        if self.initiated:
            return

        fitness_weights = fitness_weights or {}
        # assert len(weights) > 0, f'weights must have positive length'
        self.title = list(fitness_weights.keys())
        self.weights = tuple(fitness_weights.values())
        # deap.base.Fitness cannot deal 0 weights'
        self._idx  = [i for i,v in enumerate(self.weights) if v != 0]
        assert len(self._idx) > 0 , f'all fitness weights are 0!'
        self._keys = tuple(k for k,v in zip(self.title , self.weights) if v != 0)
        self._wgts = tuple(v for k,v in zip(self.title , self.weights) if v != 0)

    def fitness_value(self , metrics = None , as_abs = True , **kwargs):
        if metrics is None:
            return tuple([0. for i in self._idx])
        else:
            if as_abs: 
                metrics = abs(metrics)
            return tuple(metrics[self._idx])
    def fitness_weight(self):
        return self._wgts

class FitnessObjectMin(base.Fitness):
    weights : tuple[float,...]