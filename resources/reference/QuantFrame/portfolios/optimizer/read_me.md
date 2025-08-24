
#基于Python的组合优化工具包
天风金工：祗飞跃
[TOC]
## 1. 基本介绍
### 1.1. 优化问题
该包为基于Python的、用于单期组合优化的工具包，其所要求解的优化问题的一般形式如下：
假设$\mu$为股票的预期收益率（或打分），$\Sigma$为股票协方差矩阵，$w$为需要求解的目标权重，则优化问题形式如下：
$$
maximize\,\mu'(w-w_b)-\frac{1}{2}\lambda (w-w_b)'\Sigma(w-w_b) - \rho ||w-w^0||_{L1}
$$
且须满足
$$
(w-w_b)'\Sigma(w-w_b)\leq\sigma^2\\\
bl\leq Aw\leq bu\\\
lb\leq w\leq ub\\\
\||w-w^0||_{L1}\le TO
$$其中，
> 1. $\Sigma=F'CF+S$，F为风险因子暴露度，C为风险因子协方差，S为特质风险，这种分解可以提升优化器的效率。（注意，该工具包不能求解一般性的协方差矩阵，如需一般性协方差矩阵的优化工具，请与我们联系开发。）
> 2. $w^0$为初始股票权重
> 3. $w_b$为基准指数
> 4. TO为换手率限制，$\rho$为换手率惩罚
> 5. $\sigma$为目标跟踪误差
### 1.2. 效用函数类型
根据所求解问题的不同类型，我们将上述优化问题分为如下三类：
- **linprog** 
  效用函数和约束条件均不包含二次项（可有换手率约束或控制项）。
- **quadprog**
  仅效用函数中有二次项（可有换手率约束或控制项）
- **socp**
  效用函数和约束条件中均有二次项（可有换手率约束或控制项）。
### 1.3. 引擎介绍
- 本包提供了mosek、cvxopt、cvxpy.ecos、cvxpy.scs、cvxpy.osqp五种优化引擎。其中，后三种是cvxpy的内置优化器。
- 以上优化器中，mosek为商用优化器，cvxpy与cvxopt为开源优化器。优化效率方面，mosek最快，cvxpy的三种内置优化器次之，cvxopt最慢，具体结果可参考我们所写的测试文档。
- 以上包可以通过pip install xxx来下载。
- 建议：从速度来说，优先使用mosek，其次为cvxpy.ecos，不推荐使用cvxopt。

### 1.4. 声明
该工具包经过一定程度的测试，但是受测试场景限制，该工具包仍旧存在或有的bug，如遇到使用问题，请联系作者，不胜感激。
## 2. 使用方法
example.py导入了一个具体的使用工具包的例子。
### 2.1 工具包unit_test
该包设计了一个具体的优化示例，并提供了校验数据。通过example调用api_test.py中的函数，可以测试本地优化结果与预期优化结果之间的差异。

### 2.2 工具包inputs_utils
该包设计了若干用于生成优化函数接口（api.[exec_linprog|exec_quadprog|exec_socp]见第3节)的输入值的辅助函数，这些辅助函数主要是为了更方便的生成各种限制条件等，具体见api_test.py和第3节。

## 3. 优化器函数接口
本节介绍优化器的三种函数接口。
### 3.1. linprog
**api.exec_linprog**(solver_type, u, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True)

输入参数：
- **solver_type: str**
  优化器类型:mosek、cvxopt、cvxpy.ecos、cvxpy.scs、cvxpy.osqp
- **u: 长度为N的向量**
  预期收益。
- **lin_con: tuple**
  线性约束的相关参数，具体含义见第3节。
- **bnd_con: tuple**
  目标权重上下界参数，具体含义见第3节。
- **turn_con: tuple，默认为None**
  换手率相关的参数，若为None则代表没有换手率约束，具体含义见第3节。
- **solver_params: dict，默认为None**
  优化器相关的参数，若为None则使用参数文件中的参数
- **return_detail_infos: bool**
  是否返回优化结果细节
### 3.2. quadprog
**api.exec_quadprog**(solver_type, u, cov_info, wb, lin_con, bnd_con, turn_con=None,solver_params=None,return_detail_infos=True)
输入参数：
- **solver_type: str**
  优化器类型:mosek、cvxopt、cvxpy.ecos、cvxpy.scs
- **u: 长度为N的向量**
  预期收益。
- **lin_con: tuple**
  线性约束的相关参数，具体含义见第3节。
- **bnd_con: tuple**
  目标权重上下界参数，具体含义见第3节。
- **cov_info: tuple**
  股票协方差矩阵相关的参数，具体含义见第3节。
- **turn_con: tuple，默认为None**
  换手率相关的参数，若为None，具体含义见第3节。
- **wb: 长度为N的向量**
  基准指数权重。
- **solver_params: dict，默认为None**
  优化器相关的参数，若为None则使用参数文件中的参数
- **return_detail_infos: bool**
  是否返回优化结果细节
### 3.3. socp
**api.exec_socp**(solver_type, u, cov_info, wb, te, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True)
输入参数：
- **solver_type: str**
  优化器类型:mosek、cvxopt、cvxpy.ecos、cvxpy.scs、cvxpy.osqp
- **u: 长度为N的向量**
  预期收益。
- **lin_con: tuple**
  线性约束的相关参数，具体含义见第3节。
- **bnd_con: tuple**
  目标权重上下界参数，具体含义见第3节。
- **cov_info: tuple**
  股票协方差矩阵相关的参数，具体含义见第3节。
- **turn_con: tuple，默认为None**
  换手率相关的参数，具体含义见第3节。
- **wb: 长度为N的向量**
  基准指数权重。
- **te: float**
  目标跟踪误差
- **solver_params: dict，默认为None**
  优化器相关的参数，若为None则使用参数文件中的参数
- **return_detail_infos: bool**
  是否返回优化结果细节

### 3.4. 函数返回值：
- **w: 长度为N的一维向量**
  优化得到的目标权重
- **is_success: bool**
  是否优化成功
- **status: str**
  status:由引擎返回的优化结果相关信息。
- **效用函数最大值**
- **优化精度信息**  
  优化精度信息：Tuple(数值用于校验线性约束上界是否满足，数值用于校验线性约束下届是否满足，数值用于校验w上界是否满足，数值用于校验w下界是否满足)


## 4. 常见输入值
本节中对api函数输入中的一些常见输入值进行介绍。
### 4.1. bnd_con
含义：目标权重上下界的参数。
变量类型：list[tuple(约束类型 str，下界lb float，上界ub float)]，长度与股票个数一致。
约束类型：ra（区间），lo（下界），up（上界），fx（等式）
### 4.2. lin_con
含义：约束条件中的线性不等式和等式
变量类型：tuple(A,b)
- A: numpy 二维矩阵，行数为约束个数，列数为股票个数。
- b：list[tuple(约束类型 str,下界lb float，上界ub float)]。
### 4.3. cov_info
含义：cov_info是协方差相关的参数，
变量类型：tuple($\lambda$、F、C、S)
各部分含义：
- $\lambda$: float格式，效用函数中跟踪误差项的惩罚系数：
- F: $L × N$的矩阵，风险暴露度
- C: $L × L$的矩阵，风险因子协方差
- S: 长度为N的向量，特质波动率
### 4.4. turn_con
含义：换手率约束和惩罚参数，默认为None
变量类型：tuple($w_0$、to、$\rho$)
各部分含义：
- $w_0$: 初始权重，长度为N的向量
- to: float，股票换手率约束（双边）
- $\rho$: float，效用函数中换手率项的惩罚系数
### 4.5. solve_params
含义：优化中所需要用到的算法参数。例如：最大优化迭代次数、优化结束时所需要满足的优化精度。不同优化器的格式不同，请参阅相关文档。
变量类型：dict, 默认None
默认参数：在utils.py中设定了所有优化器所需要的默认优化参数DEFAULT_SOLVER_PARAM。
## 5. 求解精度
优化器算法会用到优化精度参数，该参数对于输入变量u、C、S的数值范围较为敏感，为了解决这一问题，我们在调用算法前做了放缩处理，将u、C、S的数值范围放缩到O(1)。
### 5.1. 变量放缩
令
$$a = std(\mu) \\\
c = mean(S)$$
输入参数放缩为：
$$
\mu_{scale} = \mu / a \\\
S_{scale} = S / c \\\
C_{scale} = C / c\\\
\sigma^2_{scale} = \sigma^2 / c \\\
\lambda_{scale} = \lambda × c / a \\\
\rho_{scale} = \rho / a
$$
### 5.2. 优化问题等价性证明
为了表明原优化问题与放缩变量后的问题等价，原因如下：
取$\Sigma=F'CF+S$，则原优化问题可转为如下形式：
$$
maximize\,\mu'w-\frac{1}{2}\lambda (w-w_b)'F'CF(w-w_b)-\frac{1}{2}\lambda (w-w_b)'S(w-w_b) - \rho ||w-w^0||_{L1}
$$
subject to 
$$
(w-w_b)'F'CF(w-w_b) + (w-w_b)'S(w-w_b)\leq\sigma^2\\\
bl\leq Aw\leq bu\\\
Gw=h\\\
lb\leq w\leq ub\\\
\||w-w^0||_{L1}\le TO
$$
带入放缩后的变量，有：
$$
maximize\,a\mu'_{scale} w-\frac{a}{2}\lambda_{scale} (w-w_b)'F'C_{scale}F(w-w_b)-\frac{a}{2}\lambda_{scale} (w-w_b)'S_{scale}(w-w_b) - a\rho_{scale} ||w-w^0||_{L1}
$$
subject to 
$$
c(w-w_b)'F'C_{scale}F(w-w_b) + c(w-w_b)'S_{scale}(w-w_b)\leq c\sigma^2_{scale}\\\
bl\leq Aw\leq bu\\\
Gw=h\\\
lb\leq w\leq ub\\\
\||w-w^0||_{L1}\le TO
$$
进一步，转换后的新问题如下：
$$
maximize\,\mu'_{scale} w-\frac{1}{2}\lambda_{scale} (w-w_b)'F'C_{scale}F(w-w_b)-\frac{1}{2}\lambda_{scale} (w-w_b)'S_{scale}(w-w_b) - \rho_{scale} ||w-w^0||_{L1}
$$
subject to 
$$
(w-w_b)'F'C_{scale}F(w-w_b) + (w-w_b)'S_{scale}(w-w_b)\leq \sigma^2_{scale}\\\
bl\leq Aw\leq bu\\\
Gw=h\\\
lb\leq w\leq ub\\\
\||w-w^0||_{L1}\le TO
$$
可以看到，对变量放缩后，新的效用函数相当于旧的效用函数乘以系数a(a>0)，跟踪误差约束条件相当于在不等式的两边同乘以系数c(c>0)，故新的优化问题和旧的优化问题等价。