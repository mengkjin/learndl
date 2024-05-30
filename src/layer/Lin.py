import torch
from torch import linalg , nn , Tensor

class HardLinearRegression(nn.Module):
    def __init__(self, intercept = True) -> None:
        super().__init__()
        self.intercept = intercept
    def forward(self, y : Tensor , x : Tensor):
        assert y.ndim == x.ndim , (y.shape , x.shape)
        if self.intercept:  x = nn.functional.pad(x , (1,0) , value = 1)
        beta = linalg.lstsq(y , x).solution
        residuals = y - x @ beta.T 
        return residuals