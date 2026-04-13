"""Symmetric orthogonalization (Gram / whitening style) for 2-D matrices in torch and NumPy."""
from __future__ import annotations
import torch
import numpy as np

def symmetric_orth(A : torch.Tensor , standardize = True , dim = 0):
    """Symmetric orthogonalization of a 2-D matrix (torch, no autograd through this path).

    Whitens columns (``dim=0``) or rows (``dim=1``) using eigendecomposition of the Gram matrix ``A.T @ A``.

    Args:
        A: 2-D tensor.
        standardize: If True, z-score along ``dim`` before and after the transform.
        dim: ``0`` for column-wise stats, ``1`` for row-wise (internally transposes).

    Returns:
        Orthogonalized matrix with the same shape as ``A``.

    Raises:
        AssertionError: If ``A`` is not 2-D or ``S`` contains NaNs after decomposition.
    """
    assert A.ndim == 2 , A.shape
    if standardize:
        A = (A - A.mean(dim,keepdim=True)) / A.std(dim,keepdim=True)
    if dim == 1: 
        A = A.T
    M = A.T.mm(A)
    M = M * (1 - 1e-6) + torch.eye(len(M)) * 1e-6
    eigenvalues , eigenvectors = torch.linalg.eigh(M)
    S : torch.Tensor = eigenvectors.mm(torch.diag(1 / eigenvalues.sqrt())).mm(eigenvectors.T)
    assert not S.isnan().any() , f'S contains nan , possibly A.T.mm(A) is not positive definite'
    B = A.mm(S)
    if dim == 1: 
        B = B.T
    if standardize:
        B = (B - B.mean(dim,keepdim=True)) / B.std(dim,keepdim=True)
    return B

def symmetric_orth_np(A : np.ndarray , standardize = True , dim = 0):
    """NumPy symmetric orthogonalization; see ``symmetric_orth``.

    Args:
        A: 2-D array.
        standardize: If True, z-score along ``dim`` before and after.
        dim: ``0`` or ``1`` for column- or row-wise convention.

    Returns:
        Orthogonalized array, same shape as ``A``.

    Raises:
        AssertionError: If ``A`` is not 2-D, ``dim`` invalid, ``A.shape[0] < A.shape[1]`` after transpose,
            or result contains NaNs.
    """
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
    assert not np.isnan(B).any() , f'diag_matrix contains nan , possibly A.T.dot(A) is not positive definite'
    if dim == 1: 
        B = B.T
    if standardize:
        B = (B - B.mean(dim , keepdims = True)) / B.std(dim , keepdims = True)
    return B
