# 恒等旁路 ResNet-GRU 基础实验

配置：`configs/schedule/current/resnet_gru_simp_15m.yaml`。
模型：`resnet_gru_simp`，独立名称 `resnet_gru_simp_15m`，种子 0。
初版新增模型时未启动训练；后续单因素实验的 worklist 安排见文末。

## 结构与默认值

输入 `[股票, 天, bar, 特征]`，默认 15m / 30 天 / 16 根 bar / 6 特征。
可选逐 bar `Linear + Tanh` 放在整个残差块之前，两支共用；默认关闭。
块内为 `Conv(C,C,3) → LeakyReLU → Conv(C,C,3) + Identity`，两层均 stride=1、replicate padding，
无 BN、无相加后激活。保留所有日内位置，旁路无参数。
相加后按 `[通道, 日内位置]` 顺序 Flatten，通过 Linear 得到 128 维日级向量。
单层 GRU 输出 64 维，取最后一天；输出头为 `Linear(64,64) → BN → Mean → BN`。
返回的 hidden 是首次输出 BN 后的因子。

`inday_resnet_dim` 是日级向量宽度，不是卷积通道数；卷积通道数始终等于当前输入空间宽度。
`resnet_kernel_size` 必须是正奇数。单层 GRU 的层间 dropout 显式为 0；多层才使用 dropout 配置。
复制边界假设边界外局部保持不变，不等于保证模型更重视开盘/尾盘。

## 参数量

默认 16 bar、6 特征、关闭投影时共 54,182 参数：日内残差块 228，日级 Linear 12,416，
GRU 37,248，输出头 4,290。开启 16 维逐 bar 投影后总计 76,114。
这些数值不包含优化器状态，也不是生产训练峰值显存。

## 损失和权重

固定 `model.special.global2top: false`，避免训练回调覆盖损失。
`mse_hcorr` 返回 MSE 与 `alpha * HiddenCorr` 两项，默认 alpha=0.01。
预测已由模型 BN 处理，标签使用 `train.label_normalization: standardize`。
权重默认 equal；可在同一 schedule 的 `train.criterion.weight.fit` 改为 polar 或 top。
权重只作用于 MSE，HiddenCorr 保持无权重，与 ccc_hcorr 一致。

HiddenCorr 重新中心化/标准化因子后求协方差矩阵 Frobenius 范数；它不除以因子数 K，
也不去掉对角线。与截图直接对 Z 使用 `||ZᵀZ||F/(N*K)` 不严格相同，alpha 不应直接解释为截图的 λ。

## 运行前置条件与启动

在具备完整 fit 数据及训练设备的服务器仓库根目录运行。
先确认当前 15m 预处理数据对应的 histnorm 文件存在、有限且形状为 `[16,6]`。
默认输入配置新增了关闭状态的 15m prenorm 键，以允许 schedule 显式开启时通过配置合并。
这也使其他显式开启 15m histnorm 的 schedule 按其声明生效；运行前同样需要有效统计。
需要重算时，手动运行以下代码（会写统计文件）：

```python
from src.data.util import DataBlock
from src.data.preprocess.hist_norm import PreProHistNorm
PreProHistNorm.calculate(DataBlock.load_preprocess('15m', frame='fit'), '15m',
                        start=20070101, end=20161231)
```

首次训练使用独立名称及 resume=0，不使用旧模型权重：

```python
from src.res.model.model_module.application.trainer import ModelTrainer
ModelTrainer.schedule(schedule_name='resnet_gru_simp_15m', resume=0)
```

保持默认的标准训练报告与产物发布流程。观察 MSE、相关性正则、验证 RankIC、年度表现、
训练/验证差距和收敛速度；本次结构更简单不代表泛化效果已经改善。
种子 1、2 的复核应另存 schedule 并更换 model.name，避免复用已有模型目录。

## 2026-11 单因素对照队列

以已经完成的 `resnet_gru_simp_15m` 为基线，新增五份 schedule，共六个变体，
按以下顺序加入 `configs/schedule/worklist.yaml`，全部 `mode: sparse`、`resume: false`。
保留种子 0、30 天窗口、训练窗口 2400、最多 200 epoch、学习率 0.001、
标准化标签、等权输出及原有 scheduler/callback 配置；Global2Top 始终关闭。

0. `resnet_gru_simp_15m_ccc`：唯一损失改为 `ccc_hcorr: {alpha: 0.01}`。
1. `resnet_gru_simp_15m_nohistnorm`：仅关闭 15m histnorm；divlast/channelnorm 继续关闭。
2. `resnet_gru_simp_15m_polar`：仅将训练样本权重改为 polar，测试权重继续 equal。
3. `resnet_gru_simp_15m_rnn2`：GRU 改为两层；原有 dropout=0.1 因此在层间生效。
4. `resnet_gru_simp_15m_dim`：并列两个参数索引，model_num=0 为日级向量 64，
   model_num=1 为 256；GRU hidden_dim 均为 64。不是同时改两项参数，也不做笛卡尔积。

这五份 schedule 使用独立 model.name，基线不重复入队。除 nohistnorm 外都需要原有有效的
15m fit 历史统计；不在训练中重新拟合。沿用标准报告、TensorBoard 与产物发布流程。
按基线约 3 小时/变体粗估，六个变体约 18 小时，实际会受宽度、GRU 层数及重试次数影响；
“并列模型”表示同一 schedule 的两个独立参数配置，并不保证同时占用 GPU 训练。

对比全期及近期 RankIC、Top5/Top50、训练验证差距、峰值 epoch、重试次数与总耗时。
CCC 与 MSE 的损失尺度不同，固定 alpha 的本轮比较包含正则相对强度变化；
两层 GRU 对照也包含既定层间 dropout 的启用。队列消费由既有服务器 worklist 流程负责。

## 追加输出组合与 Pearson 对照

当前 worklist 路径已迁移到 `configs/schedule/worklists/main.yaml`。追加两份独立 schedule：

- `resnet_gru_simp_15m_learned`：仅 `hidden_mean_pool=false`，将因子等权均值换成
  可学习 `Linear(64,1)`；继续使用 `mse_hcorr`。
- `resnet_gru_simp_15m_pearson`：仅将损失换成 `pearson_hcorr: {alpha: 0.01}`；
  保持等权输出。复用现有 PearsonLoss，即 `exp(-Pearson)`，不是 `1-Pearson`。
  返回 `pearson`、`hidden_corr` 两个分项；权重和 dim 只传给 Pearson，HiddenCorr 不加权。

其余配置均沿用原始 simp 基线，种子 0、histnorm=true、fit/test 权重 equal，
单层 GRU、日级宽度 128、Global2Top=false。以 sparse、resume=false 入队，
不重新加入已经从队列移除的旧实验。新增两项按基线粗估共约 6 小时。

## best / swalast 同轨迹对照（2026-11-19）

新增 `resnet_gru_simp_15m_learned_swa` 与 `resnet_gru_simp_15m_pearson_swa`，
仅更换模型名称，并将 submodels 改为 `[best, swalast]`。各自保留原实验的全部
网络、损失、种子、学习率及停止设置。两项在 main worklist 前部以 sparse 入队，
沿用 resume=false；不改变其他任务或 rerun_mark，不重置服务器台账。

每份 schedule 只训练一条轨迹，两种权重共享训练数据、epoch 序列以及最终选择的
trial。`dump_model` 对两种 submodel 使用同一个 best_attempt，避免独立选 trial
造成混淆；分别落盘、预测和生成标准报告，由已有模型产物流程发布。
新增 BN 校准和第二套测试会增加耗时，但不需要为两种 submodel 分别训练。

当前 SWALast 的含义不是“最后五轮”：默认 interval=3、n_last=5，以本 phase 的
最佳 epoch 为中心，取之前最多两个按 3 取整采样的 epoch、最佳点、之后的周期采样点，
截取前五个等权平均参数。例如 best=8，训练到18时，取3/6/8/9/12；靠近边界时
可能不足五个。之后在训练数据上重新估计 BN 统计量，不平均预测，不使用验证或测试
数据校准 BN。两份实验 Global2Top 均关闭，不涉及跨 phase 权重平均。

检验主问题：在相同模型日期、参数索引、测试日期下，计算 swalast-best 的年度和全期
RankIC、Top5%、Top50 收益、回撤及换手差值，并报告胜出年份数。分别在 learned 和
pearson 内配对，再比较两者改善幅度。特别检查2022、2025及2026不完整年度，避免
由单一年份贡献主导结论；统一交易成本与回测区间。检查 swa 是否仅降低收益波动，
还是同时改善排序和头部选择。不能用训练期 best 验证分数代替 swa 的测试成绩。

这些是重新训练的独立实验，不从已归档的单份 best 权重重建 SWA。种子相同不保证
跨进程、设备或代码版本逐位复现，因此主要对照为每份新实验内部 best vs swalast，
旧报告仅作参考。此轮是单种子筛选；观察到改善后再安排其他种子复核。

本地合成检查覆盖两种输出头的真实网络：best 权重精确恢复、swalast 所选点及参数
算术平均、BN 统计重算、原模型不变、输出有限，以及实际 ModelConfig 合并。
它验证实验链路，不证明样本外收益改善；真实结果需服务器完成训练与测试。
