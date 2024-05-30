import torch
import scipy
from torch import Tensor

def factor_standardize(A : Tensor , factor_dim = 0):
    return (A - A.mean(factor_dim,keepdim=True)) / A.std(factor_dim,keepdim=True)

def symmetric_orth(A : Tensor , factor_dim = 0):
    # no grad
    assert A.ndim == 2 , A.shape
    A = factor_standardize(A , factor_dim)
    M = A.T.mm(A) if factor_dim == 0 else A.mm(A.T) 
    eigenvalues , eigenvectors = torch.linalg.eigh(M)
    S : Tensor = eigenvectors.mm(torch.diag(1 / eigenvalues.sqrt())).mm(eigenvectors.T)
    assert ~S.isnan().any() , f'S contains nan , possibly A.T.mm(A) is not positive definite'
    B = A.mm(S) if factor_dim == 0 else S.mm(A) 
    return B