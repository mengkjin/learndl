from dataclasses import dataclass
from torch import Tensor
from . import (
    ckpt , model , pipeline
)

@dataclass
class Output:
    outputs : Tensor | tuple | list

    def pred(self):
        if isinstance(self.outputs , (list , tuple)):
            return self.outputs[0]
        else:
            return self.outputs
    
    def hidden(self):
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
