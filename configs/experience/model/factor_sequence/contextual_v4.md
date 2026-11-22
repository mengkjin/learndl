# Contextual v4：共享表现摘要与有界贡献调节

历史草案。该 schedule 已归档，当前实验见 [v4 时序模型卡](contextual_v4_sequence.md)。

状态：已实现，尚未进行服务器历史训练。普通 schedule 为
`factorseq_v4_contextual`，注册模块仍为 `factorseq_residual`，模式为
`contextual_v4`。使用新的模型根目录，不恢复 v3 Contextual 权重。
没有自动加入 worklist 或启动训练。

递进的连续市场版与连续＋事件版见 [市场扩展模型卡](contextual_v4_market.md)。
它们采用独立 schedule，原始纯 v4 的输入与参数量保持不变。

## 结构

保留 Temporal 的全部股票主干：132 通道股票输入、60 日序列、32 维投影、
两层 GRU、dropout 0.1、当前截面直连、零初始化的非线性输出层。
仅增加一个所有 33 因子共用的 `14 → 8 → 1` GELU 网络。

对每个因子的当前线性贡献作如下调节：

```
base = linear(v_t) + residual(v_t, metadata_t, stock_gru)
g_f = 1 + 0.2 * tanh(shared_mlp(summary_f))
prediction = base + sum_f(w_f * v_f,t * (g_f - 1))
```

`w_f` 复用线性分支权重；偏置只加一次。无效的当前因子值不产生情境修正。
调节不改变股票 GRU 历史，也不直接进入非线性预测头。门控范围为 `[0.8,1.2]`，
保持当前线性系数符号；不约束整个模型的非线性暴露或预测幅度。
不加载市场数据，不使用情境 GRU、FiLM、因子嵌入或逐因子独立网络。

默认总参数 **22,404**，相比 Temporal 的 22,275 只增加 **129**；
旧 Contextual 为 48,515。最后的门控线性层零初始化，使初始调节严格为 1。
在股票主干权重相同的情况下，初始预测与 Temporal 一致。

## 输入与缺失

只读 repaired 数据合同：stock 按 value/valid/gap/never 各 33 通道排列；
performance 按冻结因子顺序，每因子为 IC、IC valid、spread、spread valid。
因子方向沿用数据块；IC 保留原值，spread 保持百分点。
没有在服务器重新构建数据块的要求。

每个日期从公共 120 日表现窗口得到每因子 14 个特征。IC 和 spread 各自包含：

- 半衰期 5/20/60 交易日的三个指数加权均值；
- 同三个半衰期下的有效权重覆盖率；
- 5 日半衰期均值减 60 日半衰期均值。

权重为 `2 ** (-交易日距离 / 半衰期)`。均值只除以有效观测的权重和；
覆盖率除以全部窗口日期的权重和。无效值及非有限值排除，不前向填充。
全无效的均值编码为零，其覆盖率也为零；这不是经济零收益。
IC 和 spread 在窗口内各自至少有 20 个有效日，才允许该因子偏离恒等门控。
不足时即使 MLP 学到了偏置也强制 `g=1`。20 是观测数门槛而非连续完整日门槛；
暂不对最近一次有效观测另设硬阈值，稀疏程度由多尺度覆盖率表达。
每次仅用提供的历史窗口计算，不维护跨 batch 的可变状态，不读取未来标签。

## 训练与比较

使用普通 schedule 接口：CLI 的 Train Schedule Model 选择
`factorseq_v4_contextual`。继续采用 `repaired_20261117` 的冻结数据、标签截止日和
历史切分，保持 v3 的种子 17、MSE、最多 30 epoch、验证 Rank IC 早停耐心 5，
DetailedAlphaAnalysis 的 `factor/t50` 任务及普通模型级邮件。
切分与生命周期的实际训练日期由现有接口决定，不另造逐折任务或邮件。

门控参数在 `worklist.residual_config` 的 v4 分支生成并保存到实际 algo 配置；
schedule 源文件是解析入口，不在那里添加会被源解析器忽略的算法参数。
损失附加 `0.01 * mean_ready((g-1)^2)`，通过训练器现有 `loss_*` 接口汇总；
没有可用门控时该项为零。验证模型选择仍按 Rank IC。

首版采用**从头联合训练**，没有隐式加载或冻结 v3 Temporal 权重。此时 `base` 是
v4 内联合训练的未调节路径，不能当作独立重训 Temporal 的消融结果。
冻结已训练 Temporal、只训练门控可作为后续独立实验，需要逐期匹配权重与切分。

输出保留 `hidden/linear_score/residual_score`；其中 linear_score 已含情境修正，
因此二者之和仍等于预测。额外提供 `base_linear_score/temporal_score/context_correction`、
`factor_gate/gate_summaries/gate_ready` 和 `loss_context_shrinkage` 供诊断。
现有回调保存通常的线性/非线性分支结果；额外门控张量仅由 forward 返回，
不宣称已被历史报告持久化。

验收应比较与 v3 Temporal 的逐日配对 ΔIC、年度表现、训练/验证差距及成本后收益。
小参数量和有界门控降低自由度，但不证明 IC/spread 存在可利用信号，亦不保证收益提升。

## 本地验证

覆盖独立 NumPy 摘要对照、缺失/成熟门槛、因子排列对应、前缀因果性、门控边界、
Temporal 恒等路径、实际 loss 汇总和梯度、序列化，以及 repaired 数据的真实
ScheduleConfig/AlgoConfig 加载链路。旧三个模式保留原结构及参数键。
CPU 人工数据测试不代表服务器真实训练、内存或样本外表现已通过。
