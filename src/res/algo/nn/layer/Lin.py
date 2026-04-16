"""Closed-form (non-learnable) linear regression residualization layer."""
from torch import linalg , nn , Tensor

class HardLinearRegression(nn.Module):
    """Closed-form OLS residualization via ``torch.linalg.lstsq``.

    For each output feature, regresses ``y[..., i]`` on ``x`` and returns the
    residuals ``y - x @ beta``.  Commonly used to neutralize style / size
    factors from a sequence of hidden states (e.g. in ``gru_dsize``).

    Args:
        intercept: If ``True`` (default), prepend a column of ones to ``x``
                   so the regression includes a bias term.

    Shapes:
        y: ``[..., n_targets]``  — response variables (one per feature)
        x: ``[..., n_regressors]`` — regressor matrix (same leading dims as y)

    Returns:
        Residuals ``y - x @ beta`` with the same shape as ``y``.
    """
    def __init__(self, intercept = True) -> None:
        super().__init__()
        self.intercept = intercept
    def forward(self, y : Tensor , x : Tensor):
        assert y.ndim == x.ndim , (y.shape , x.shape)
        if self.intercept:
            x = nn.functional.pad(x , (1,0) , value = 1)
        residuals = y * 0.
        for i in range(y.shape[-1]):
            beta = linalg.lstsq(y[...,i:i+1] , x).solution
            residuals[...,i:i+1] = y[...,i:i+1] - x @ beta.T
        return residuals