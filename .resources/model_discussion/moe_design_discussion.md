# MOE 股票预测模型设计讨论

> 背景：五数据集独立 GRU（day / 30m / week / dfl2 / dfl2cs）预测 `std_lag1_10`，隐状态由 LGBM 二层融合。拟在单数据集内引入 MOE，并保留「分模型训练 + 树模型整合」的非端到端架构。
>
> 日期：2026-06-25

---

## 问题 1：防同质化——损失项 vs 结构/数据设计？

你的理解**大体正确**，但需要补充边界：1 和 2 主要通过**辅助损失**实现；3–5 主要通过**结构与输入设计**实现，其中部分手段也可辅以轻量正则，但核心不在 loss。

### 分类对照

| 手段 | 主要实现层 | 说明 |
|------|-----------|------|
| **1. Load-balancing** | 辅助损失 | 惩罚 gate 长期只激活少数 expert（Switch Transformer 思路）。典型形式：最小化 expert 负载分布与均匀分布的 KL，或对 batch 内平均 gate 权重加方差惩罚。 |
| **2. Diversity 正则** | 辅助损失 | 约束 expert 输出（hidden 或 pred）两两相似度不要过高：如余弦相似度惩罚、协方差去相关。直接针对「同质化」表征。 |
| **3. Gate 与 expert 输入分离** | 模型结构 | Gate 只看原始输入 `x`（或 `x + market`），各 expert 独立参数、独立前向。不共享 encoder 权重。这是**结构性**防塌缩，通常不单独加 loss。 |
| **4. 异构 expert** | 模型结构 | 不同 expert 用不同 backbone（GRU / TCN / 不同层数 / 不同 hidden_dim）。强迫函数类不同，比同质 GRU 更难塌成同一映射。 |
| **5. Feature subspace mask** | 数据管线 + 模型输入 | 每个 expert 只看特征子集（价格 / 量 / 微观结构等）。在数据侧定义 mask，模型侧用 masked input 或 separate embedding。 |

### 需要细化的几点

**1 和 2 并不互斥，且都应克制使用**

- Load-balancing 解决的是「**利用率**」问题（某个 expert 被 gate 忽略），不保证 expert 学到不同东西。
- Diversity 解决的是「**表征重叠**」问题，不保证 gate 均匀调用。
- 实践中 **1 + 2 组合**最常见；权重宜小（如 0.01 / 0.001 量级），避免主任务 `ccc_hcorr` 被干扰。

**3–5 也可以有 loss 的弱形式，但不是主路径**

- 异构 expert（4）有时会对不同 expert 加不同的 dropout / noise，略带正则色彩，但本质仍是结构。
- Feature mask（5）若 mask 重叠度过高，仍可能同质化——mask 设计本身比 loss 更重要。

**还有一类「隐式防同质化」未列入原表**

- **Noisy gating / temperature**：推理前对 gate logits 加噪声或调高 temperature，训练时更平滑，减少过早锁定单一 expert。
- **Expert capacity 上限**：限制单 expert 在 batch 内最大样本占比（与 load-balance 相关但可在采样层实现）。
- **Top-k sparse gating**：只保留 top-k 权重再归一化——这改变的是融合方式（见问题 4），对同质化影响复杂：可能促进分工，也可能使未选中 expert 梯度稀疏。

### 小结

> **1、2 → 以辅助损失为主；3、4、5 → 以结构与输入设计为主。**  
> 最稳妥的组合是：**3（结构分离）+ 4 或 5（制造 inductive bias）+ 1（防 gate 塌缩）+ 可选 2（防表征重叠）**。

---

## 问题 2：市场数据如何处理？（8 维微观风险 + 指数涨跌幅）

市场数据与个股数据有三类差异，需分开处理：

1. **维度与语义**：8 维风险特征（已加工）vs 指数涨跌幅（原始、有量级意义）
2. **序列长度**：可与股票一致，也可更长
3. **标准化**：股票特征通常已预处理；市场序列未必标准化，且「原始数值有意义」

### 原则：不要对市场数据做与股票相同的「一刀切」截面标准化

市场序列对同一天所有股票相同，是**纯时间序列**变量。应使用：

- **时间序列归一化**（rolling z-score、rolling rank、expanding z-score），或
- **分通道处理**（不同字段不同规则），或
- **交给独立 encoder** 学习尺度

### 建议的数据处理方案

#### A. 8 维 `market_daily/risk`

与现有因子类数据类似，建议：

- 对每列做 **rolling 时间序列 z-score**（窗口如 60/120 交易日，与 day 序列尺度匹配）
- 或 rolling rank → 映射到 `[-1, 1]`，对极值更稳
- 缺失日 forward-fill（市场特征无个股差异，ffill 合理）

#### B. 指数涨跌幅（及衍生）

保留「原始有意义」时，推荐**拆通道、分语义**，而非全部 z-score：

| 通道 | 建议处理 | 理由 |
|------|---------|------|
| 日收益 `r_t` | 可直接使用，或 `log(1+r)` | 量级小，本身近似平稳 |
| 累计收益（N 日） | 保留原值或 log 累积 | 表达趋势强度；标准化会抹掉「涨跌 20% vs 2%」的 regime 差异 |
| 实现波动率（rolling std of r） | 原始或 log | 表达市场状态，与收益正交 |
| 成交量 / 成交额变化 | log1p + 可选 ts z-score | 右偏分布 |

**核心思想**：  
- **水平/累积类**（累计涨跌、趋势）→ 倾向保留原尺度或 log 变换  
- **波动/微观结构类** → 倾向 rolling 标准化  
- 不要对「累计涨 15%」和「日内 true range」用同一套 global z-score

#### C. 序列长度不一致

完全可行，且往往更合理：

```text
stock encoder:  seq_len = 30（与现 gru_day_new 一致）
market encoder: seq_len = 60 或 120（更长历史捕捉 regime）
```

对齐方式：

- 预测日为 `T`，股票窗口 `[T-29, T]`，市场窗口 `[T-59, T]`（末端对齐）
- Market expert 或 market-conditioned gate 只吃市场 encoder 输出；stock-only expert 不变
- 不要求 market 与 stock **逐步对齐**同一长度，只要求 **决策日对齐**

#### D. 在 MOE 中的接法（与之前计划一致，补充数据细节）

1. **PrePro_market**：输出 `(date, feature)` 市场块，在 `ModuleData` broadcast 到 `(secid, date)`
2. **双输入**：`(stock_seq, market_seq)`，长度可不同
3. **Market-aware expert**：`concat(stock_t, market_t)` 逐步融合，或双塔 encoder 后 concat hidden
4. **Gate 条件化**：gate 输入 = `stock_summary` + `market_summary`（两路 pooling 后拼接）

#### E. 一个可落地的特征包示例

```text
market_features = [
  # 来自 market_daily/risk（8 维，ts z-score）
  risk_8d_ts_zscore,

  # 指数（如 CSI300 / 中证全指）
  index_log_return_1d,           # 原始或轻微 winsorize
  index_cum_return_5d,           # 保留尺度
  index_cum_return_20d,
  index_realized_vol_20d,        # rolling std，可 log
  index_drawdown_from_60d_high,  # 水平型 regime 特征
]
```

训练初期可只用 8 维风险 + 1 个核心指数收益，避免维度过高淹没 stock signal。

---

## 问题 3：不同 MOE 应拟合相同还是不同的 label？

### 结论（针对你当前的二层架构）

**所有数据集上的 MOE，应拟合同一个主 label：`std_lag1_10`。**

理由：

1. **LGBM 层的契约**：`lgbm_of_hidden` 将五路 hidden 映射到**一个**目标。若某路 MOE 改预测 `std_lag1_20`，该路 hidden 的语义与 LGBM 期望不一致，融合难度上升。
2. **MOE 的定位**：MOE 是「单任务内的软集成」，不是多任务头。Expert 分工应在**表征空间**，而非**标签空间**。
3. **与 PLE 的区分**：PLE 用不同 expert 服务不同 task（多 label），适合端到端多任务。你的架构是「多表征 → 单树 → 单 label」，不宜混用。

### 何时可以考虑不同 label？

| 场景 | 是否推荐 | 说明 |
|------|---------|------|
| 五数据集 MOE 各预测不同 horizon | 不推荐（在你现架构下） | 破坏 LGBM 输入语义一致性 |
| MOE 内 expert 分别预测 5d / 10d / 20d | 可选（需改架构） | 变成多任务 MOE，hidden 需明确定义取哪一头；LGBM 要改输入 |
| 主 label `std_lag1_10` + expert 辅助 loss 对齐子目标 | 可作消融 | 如某 expert 加弱权重 aux loss 对齐短期收益，仅作归纳偏置 |
| 不同**数据集**预测不同 label，不用 LGBM 融合 | 另一套实验 | 与当前 pipeline 正交 |

### 实操建议

- **Phase 1（day MOE 试点）**：五 expert 全部监督于 `std_lag1_10`，与 `gru_day_new` 完全一致。
- **若想做「horizon 分工」**：优先在 **LGBM 之前**用不同 schedule 预测不同 label 并平行融合，而不是在单数据集 MOE 内拆 label——那更接近多模型集成，而非 MOE。

---

## 问题 4：GatingNetwork 是「软融合」还是「硬路由」？

### 结论

**PLE 及计划中的 `moe_gru` 风格，属于软融合（soft combination），不是硬选择器（hard router）。**

对照你代码库中 [`PLE.py` 的 `GatingNetwork`](../../src/res/algo/nn/model/PLE.py)：

```python
v = self.softmax(v)  # gate 权重，和为 1
v = torch.stack([v[:,i] * vec for i, vec in enumerate(vecs)]).sum(0)  # 加权求和
```

### 具体含义

| 维度 | 软融合（当前 PLE） | 硬路由（Switch Transformer 等） |
|------|-------------------|--------------------------------|
| Expert 是否全部前向 | **是**，每个 expert 每次都算 |  often 只算 top-1 / top-k |
| 权重 | 连续，softmax 后 `(0,1)` |  one-hot 或 sparse top-k |
| 梯度 | 所有 expert 都收到梯度（经 gate 权重缩放） | 未选中 expert 可能无梯度 |
| 输出 | `h = Σ w_i * h_i` | `h = h_{argmax}` 或稀疏组合 |
| 计算成本 | `O(num_experts)` 全算 | 可节省计算 |

因此你的理解正确：

> **每一个 expert 的表征都参与最终融合，只是权重由 GatingNetwork 决定；不是「只选一个 expert」的离散路由。**

### 需要区分的两个「层次」

1. **融合的是 hidden 还是 pred？**  
   - PLE / 计划 moe_gru：在 **hidden 层**加权求和，再经 decoder → 单一 pred。  
   - 也可做 **pred 层**加权：`y = Σ w_i * y_i`，语义更直观但 hidden 不可分。

2. **Top-k 归一化仍是软融合**  
   - 若取 top-2 再 softmax 归一化：仍是 2 个 expert 的凸组合，不是 hard switch。  
   - 只有 one-hot + straight-through 或 REINFORCE 才是硬路由。

### 对你架构的影响

- **优点**：训练稳定，expert 持续得到梯度，适合金融小信噪比场景。
- **缺点**：推理成本 = 全部 expert 之和；gate 与 LGBM 都在做「加权组合」，功能部分重叠。
- **与 LGBM 的关系**：MOE gate = 频率内一阶软集成；LGBM = 跨频率二阶非线性集成。二者叠加合理，但应用 day 试点量化边际收益。

### 何时考虑硬路由？

- 专家数很多（>8）、计算成为瓶颈时。
- 明确希望 expert 专精某一 regime 且可解释「本次由谁负责」时。  
- 量化场景下 **默认仍建议软融合**，硬路由更容易训练不稳定、expert 饿死。

---

## 综合建议（讨论阶段，不涉及实现）

1. **防同质化**：结构先行（gate/expert 分离 + 异构或 mask），再用小权重 load-balance + diversity loss 兜底。
2. **市场数据**：分通道处理；指数收益保留 regime 尺度；市场序列可长于个股；末端日期对齐。
3. **Label**：全系统统一 `std_lag1_10`；MOE 做表征分工，不做 label 分工。
4. **Gate**：默认软融合；所有 expert 每次有效；与 LGBM 形成两层集成。

---

## 开放问题（后续可继续讨论）

- [ ] day 数据集的 feature mask 如何按语义划分（需对照 `PrePro_day` 特征列表）
- [ ] 市场指数选 CSI300 vs 中证全指 vs 全 A 等权
- [ ] MOE hidden 导出：仅 gated 64d vs 拼接 `gate_weights` 给 LGBM
- [ ] 是否在 MOE 层导出各 expert hidden 供 LGBM 显式使用（类似 Stacking）
