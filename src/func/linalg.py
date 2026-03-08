import torch
import numpy as np

def symmetric_orth(A : torch.Tensor , standardize = True , dim = 0):
    # no grad
    assert A.ndim == 2 , A.shape
    if standardize:
        A = (A - A.mean(dim,keepdim=True)) / A.std(dim,keepdim=True)
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
    if standardize:
        B = (B - B.mean(dim,keepdim=True)) / B.std(dim,keepdim=True)
    return B

def symmetric_orth_np(A : np.ndarray , standardize = True , dim = 0):
    assert A.ndim == 2 , A.shape
    assert dim in [0 , 1] , dim

    if standardize:
        A = (A - A.mean(dim , keepdims = True)) / A.std(dim , keepdims = True)
    if dim == 1: 
        A = A.T

    assert A.shape[0] >= A.shape[1] , f'A.shape[0] >= A.shape[1] , {A.shape}'

    gram_matrix = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    epsilon = 1e-10
    diag_matrix = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
    B = A @ eigenvectors @ diag_matrix @ eigenvectors.T
    assert ~np.isnan(B).any() , f'diag_matrix contains nan , possibly A.T.dot(A) is not positive definite'
    if dim == 1: 
        B = B.T
    if standardize:
        B = (B - B.mean(dim , keepdims = True)) / B.std(dim , keepdims = True)
    return B