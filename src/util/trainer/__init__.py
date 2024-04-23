from dataclasses import dataclass
from torch import Tensor
from . import (
    ckpt , model , optim , pipeline
)

@dataclass
class Output:
    outputs : Tensor | tuple | list
    @property
    def pred(self) -> Tensor:
        return self.outputs[0] if isinstance(self.outputs , (list , tuple)) else self.outputs
    @property
    def hidden(self) -> Tensor | None:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
    @classmethod
    def empty(cls): return cls(Tensor().requires_grad_())
