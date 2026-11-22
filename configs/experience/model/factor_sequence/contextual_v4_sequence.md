# Contextual v4 因子表现时序实验

三个普通 schedule 共用注册模块 `factorseq_contextual_v4`，采用同一冻结
`repaired_20261117` 数据、标签和历史切分。从头联合训练，复用 Temporal 的
60 日双层股票 GRU、当前因子原值旁路和预测头。因子表现读取 60 个交易日，
每因子只读 IC、spread 和对应掩码。每种指标至少 20 个有效日才启用 FiLM。
使用收盘时可得的已实现表现，按日期给整个截面一份公共输入。

| Schedule | 表现编码 | 可训练参数 |
| --- | --- | ---: |
| `factorseq_v4_shared` | 33 个因子共用一个 4→16 GRU，各自保留隐藏状态 | 23,979 |
| `factorseq_v4_private` | 共享版加 33 个独立 4→4 GRU，共享 4→16 投影和有界残差 | 28,003 |
| `factorseq_v4_graph` | 共享版加 4 维因子身份和单头完整注意力残差，排除自身与不成熟邻居 | 24,975 |

Temporal 为 22,275。共同新增 1,704 个参数，分别是表现 GRU 1,056、
个股因子编码 40、FiLM 272、共享读出 336。专属时序新增 4,024，
因子身份与图合计新增 996。图由当日因子表示学习，不输入相关性先验。
市场信息留给后续 v5。

每个因子独立生成 8 维 gamma 和 beta，范围分别为 ±0.2 和 ±0.1。
共享读出用同一个股票表示分别读取调节前后的当前因子表示。
两者相减，掩码后按 33 个因子取平均，再加回 Temporal 分数。
FiLM 初始恒等。表现和个股当前值任一无效时，该因子修正为零。
三个版本均使用 0.1 的情境 dropout，以及 0.01 系数的幅度惩罚。

按默认维度估算，公共表现 GRU 每日约 190 万次矩阵乘加，私有 GRU
约再加 19 万次，图的线性映射与注意力矩阵计算约再加 5.4 万次。
逐因子修正每股约新增 5,800 次矩阵乘加，股票 Temporal 主干每股约 99.6 万。
这些估算未计偏置、激活和反向传播。33 个独立 GRU 的实际调用开销需在服务器验证。

闲时队列 `configs/schedule/worklists/factorseq_v4.yaml` 按 shared、private、
graph 排列，采用 `resume: false`、`mode: sparse`、步长 2，与已有
v3 Temporal 的训练日期网格一致。手动使用 CLI 的 Train Schedule Model 时，
可选择对应的 `resume=False`、`training_mode=sparse`、`training_step=2`。
三者各自使用一个模型目录，
每个任务覆盖整段历史并沿用普通模型级报告、邮件与投递。
种子 17、最多 30 epoch、验证 Rank IC 耐心 5，开启 DetailedAlphaAnalysis 的
`factor/t50`。
模型结果的 `factor_context_v4/` 保存所选 epoch 的训练/验证差、
每日 FiLM 和修正量、覆盖率统计、图注意力熵与有效节点数。
逐股文件仍保存在普通 `snapshot/factor_residual/` 中。

完成四个模型后，用实际模型根目录在服务器生成配对报告：

```bash
.venv/bin/python -m src.res.model.analytic.factor_sequence.compare_v4 \
  --temporal /path/to/temporal --shared /path/to/shared \
  --private /path/to/private --graph /path/to/graph \
  --output /path/to/new/report
```

报告核对冻结数据、切分、日期、证券和标签，导出各版本对 Temporal 的逐日、
年度及覆盖率分层 ΔIC，以及保存的训练/验证曲线。V4-1 的差异包含容量，
V4-2 的差异包含身份和图的组合。服务器运行状态以实际训练和模型级报告为准。
