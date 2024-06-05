import numpy as np


def _check_bnd_con_inputs(bnd_con, n):
    '''
    bnd_con
        list，3个1维向量，分别为约束类型、下界(lb)、上界(ub)：
        - 约束类型：str格式，有ra(w<=ub且w>=lb)、lo（w>=lb）、up(w<=ub）、fx(w=lb且w=lb)四种。
        - 下界：float格式，目标权重的下界
        - 上界：float格式，目标权重的上界
    '''
    assert isinstance(bnd_con, list)
    bnd_key, lb, ub = bnd_con[0], bnd_con[1], bnd_con[2]
    assert all([key in ("ra", "lo", "up", "fx") for key in bnd_key]) and isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
    assert lb.shape == (n, ) and ub.shape == (n, ) and len(bnd_key) == n


def _check_lin_con_inputs(lin_con, N):
    '''
    lin_con
        tuple格式，是和线性约束有关的参数，不等式约束与等式约束均包含在内，分为A与b两部分。
        - A: array，线性约束的系数。
        - b：list。3个1维向量，分别为约束类型、下界、上界。
    '''
    assert isinstance(lin_con, tuple) and len(lin_con) == 2
    A, b = lin_con
    assert isinstance(A, np.ndarray)
    assert all([key in ("ra", "lo", "up", "fx") for key in b[0]]) and isinstance(b[1], np.ndarray) and isinstance(b[2],
                                                                                                                  np.ndarray)
    assert len(A) == len(b[0]) and len(b) == 3 and len(b[1]) == len(b[0]) and len(b[1]) == len(b[2])
    assert A.shape[1] == N


def _check_turn_con_inputs(turn_con, N):
    '''
    turn_con:
        和换手率有关的参数，格式为tuple，分为w0、to、rho三个部分。其含义如下
        - w0: 长度为N的向量，股票初始权重
        - to: float格式，股票换手率约束（双边）
        - rho: float格式，效用函数中换手率项的惩罚系数
    '''
    assert isinstance(turn_con, tuple)
    w0, to, rho = turn_con
    assert isinstance(rho, float) and rho >= 0.0
    assert isinstance(w0, np.ndarray) and w0.shape == (N,)
    assert isinstance(to, float) and to > 0.0


def _check_cov_info_inputs(cov_info, N):
    '''
    cov_info:
        协方差相关的参数，格式为tuple，
        格式一：分为lmbd、F、C、S四个部分。其含义如下
        - lmbd: float，风险厌恶系数
        - F: L × N的矩阵，风险暴露度
        - C: L × L的矩阵，风险因子协方差矩阵
        - S: 长度为N的向量，特质方差
        格式二：分为lmbd，Cov两个部分。其含义如下：
        - lmbd: float，风险厌恶系数
        - Cov: N x N的矩阵，股票收益率协方差矩阵
    N:
        股票个数
    '''
    assert isinstance(cov_info, tuple)
    if len(cov_info) == 4:
        lmbd, F, C, S = cov_info
        assert isinstance(lmbd, float) and lmbd >= 0.0
        assert isinstance(F, np.ndarray) and F.shape[1] == N
        L = F.shape[0]
        assert isinstance(C, np.ndarray) and C.shape == (L, L)
        assert isinstance(S, np.ndarray) and S.shape == (N,) and (S > 0.0).all()
    elif len(cov_info) == 2:
        lmbd, cov = cov_info
        assert isinstance(lmbd, float) and lmbd >= 0.0
        assert isinstance(cov, np.ndarray) and cov.shape == (N, N)
    else:
        assert False, "  error::optimizer>>input_validator>>unknown cov_info type."


def validate_inputs(u, lin_con, bnd_con, turn_con=None, cov_info=None, wb=None, te=None):
    assert isinstance(u, np.ndarray) and u.ndim == 1
    N = u.shape[0]
    if bnd_con is not None:
        _check_bnd_con_inputs(bnd_con, N)
    if lin_con is not None:
        _check_lin_con_inputs(lin_con, N)
    if turn_con is not None:
        _check_turn_con_inputs(turn_con, N)
    if cov_info is not None:
        _check_cov_info_inputs(cov_info, N)
    if te is not None:
        assert isinstance(te, float) and te > 0.0
    if wb is not None:
        assert isinstance(wb, np.ndarray) and wb.shape == (N,)