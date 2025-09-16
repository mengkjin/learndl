import torch

def factor_standardize(A : torch.Tensor , dim = 0):
    return (A - A.mean(dim,keepdim=True)) / A.std(dim,keepdim=True)

def symmetric_orth(A : torch.Tensor , dim = 0):
    # no grad
    assert A.ndim == 2 , A.shape
    A = factor_standardize(A , dim)
    if dim == 1: 
        A = A.T
    M = A.T.mm(A)
    M = M * (1 - 1e-6) + torch.eye(len(M)) * 1e-6
    eigenvalues , eigenvectors = torch.linalg.eigh(M)
    S : torch.Tensor = eigenvectors.mm(torch.diag(1 / eigenvalues.sqrt())).mm(eigenvectors.T)
    assert ~S.isnan().any() , f'S contains nan , possibly A.T.mm(A) is not positive definite'
    B = A.mm(S)
    if dim == 1: 
        B = B.T
    B = factor_standardize(B , dim)
    return B