## 计算因子收益
### 1. 数据获取与处理
输入calc_date_list，获取对应日期的barra因子数据，读取收益数据并计算预期的期间算数收益率。
取universe里的barra数据,在universe里demean再除以std。对行业变量进行one-hot处理，添加全为1的country变量。

### 2. 回归
以barra因子作为X，期间收益为y，在时间截面上进行回归。设共有1+k+n个因子，取$country$为$x_0$，$x_1、x_2、...、x_k$为行业因子，$x_{k+1}、...、x_{k+n}$为style因子,$\beta_i$为因子i的因子收益，$\epsilon$为残差收益,则问题可表述为：
$$y=\beta_0x_0+\beta_1x_1 + \beta_2x_2 + ... +\beta_{k-1}x_{k-1}+\beta_{k}x_{k}+\beta_{k+1}x_{k+1} + ... + \beta_{k+n}x_{k+n}+\epsilon $$
考虑到$country$变量可由行业变量线性表示,故需对变量加上限制条件。设$w$为加权回归时的权重，假设有以下条件成立：
$$\beta_1wx_1+\beta_2wx_2+...+\beta_kwx_k=0\\
wx_{k+1}=0\\
...\\
wx_{k+n}=0$$
此时，有$w\epsilon=wy-(\beta_0wx_0+\beta_1wx_1...+\beta_kwx_k + \beta_{k+1}wx_{k+1}+... + \beta_{k+n}wx_{k+n})=wy-\beta_0wx_0=0$
为便于回归，对于行业变量$x_k$,令
$$\beta_k = -\frac{1}{w^I_k}\sum_{i=1}^{k-1}w^I_i\beta_i$$其中，$w^I_i=\frac{wx_i}{\sum_{i=1}^kwx_i}$，进而，收益可表示为：
$$y = \beta_0x_0+\beta_1(x_1-\frac{w^I_1}{w_k^I}x_k) + \beta_2(x_2-\frac{w^I_2}{w_k^I}x_k) + ... +\beta_{k-1}(x_{k-1}-\frac{w^I_{k-1}}{w_k^I}x_k) + \beta_{k+1}x_{k+1} + ... + \beta_{k+n}x_{k+n}+\epsilon$$
subject
$$
wx_{k+1}=0\\
...\\
wx_{k+n}=0$$
基于处理后的矩阵回归，最后代回计算$\beta_k$
### 3. 存储因子收益数据
存储每期的$\beta_0、\beta_1、...、\beta_{k+n}$到本地，列名分别为CalcDate、CalcDate_y_n与各因子的因子收益
<br />

### 附：市值加权下WLS回归的推导
基础假设：股票残差收益的variance与市值的平方根呈反比，即大市值公司有着更小的variance
设因子个数为n+k+1，当前截面共有m只股票。最小化函数：
$$J(\beta) = \frac{1}{2}(y- X\beta)W(y- X\beta)^T$$其中，$X$为$m\times(n+k+1)$维矩阵，$\beta$为$1\times (n+k+1)$维矩阵，$y$为$m$维向量，$W$为$m\times m$维对角矩阵。其元素为市值的平方根除以所有平方根的和。
目标函数对X求导后，有：$$\frac{\partial J}{\partial\beta}=-X^TW(y-X\beta)$$取导数为0，有$$\beta=(X^TWX)^{-1}X^TWy$$
考虑到需要行业约束，使$\beta_k = -\frac{1}{w_k}\sum_{i=1}^{k-1}w_i\beta_i$，其中，$w_i$为市值加权下的行业i的权重。
取$$R=\left(                
  \begin{array}{ccc}   
    1 & 0 & ... & 0& 0 & 0 & 0\\ 
    0 & 1 & ... & 0& 0 & 0 & 0\\  
    ... & ... & ... & ... & ... & ... & ...\\ 
    0 & 0 & ... & 1 & ... & 0 & 0\\ 
    0 & -\frac{w_1}{w_k} & ... & -\frac{w_{k-1}}{w_k}& 0 & ... & 0\\ 
    ... & ... & ... & ... & ... & ... & ...\\ 
    0 & 0 & ... & 0& 0 & 0 & 1\\ 
  \end{array}
\right)$$其中，R为$(n+k+1)\times(n+k)$维矩阵
则有$$\beta=\left(\begin{array}{ccc}   
    b_0\\ 
    b_1\\ 
    ... \\
    b_{k-1}\\ 
    b_{k}\\  
    ...\\ 
    b_{k+n}
  \end{array}\right)=R
\left(                
  \begin{array}{ccc}   
    b_0\\ 
    b_1\\ 
    ... \\
    b_{k-1}\\  
    ...\\ 
    b_{k+n}
  \end{array}
\right)=R\beta_{adj}$$
原问题变为$$J(\beta) = \frac{1}{2}(y- XR\beta_{adj})W(y- XR\beta_{adj})^T$$
最终解得$$\beta_{adj}=(R^TX^TWXR)^{-1}R^TX^TWy$$
$$\beta=R\beta_{adj}=R(R^TX^TWXR)^{-1}R^TX^TWy$$