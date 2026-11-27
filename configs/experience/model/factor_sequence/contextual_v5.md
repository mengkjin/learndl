# FactorSeq v5 市场贡献实验

对照为 `factorseq_v4_graph`。两个正式任务共用 v4 的股票 GRU、因子表现 GRU、因子图、FiLM 和原始因子旁路。新增市场编码每个日期计算一次。股票和因子表现各回看 60 日，市场只读取同日一条公共记录。

- `factorseq_v5_graph_market` 使用 `market_up`、八项风险及其有效性。输出 33 个因子贡献倍率和 32 个股票隐藏通道倍率。总参数为 25,141，比 Graph 多 166。
- `factorseq_v5_graph_market_events` 加入暴跌反弹、横盘突破和高位切换的触发标记及衰减。事件只调节因子贡献。总参数为 25,186，比市场版多 45。

市场和事件共用因子侧的有界倍率，股票侧使用自己的有界倍率。两者范围均为 0.8–1.2。零初始化输出头使模型在载入相同父版本权重时从同分数开始。训练损失沿用 0.01 的收缩系数，并惩罚新增各门的偏移。因子当前无效时修正为零，因子表现未成熟时仍可调节当前有效的原始值。市场和事件各自处理有效性。

## 普通训练入口

闲时训练读取 `configs/schedule/worklists/factorseq_v5.yaml`，依次安排市场版、事件版和学习率小试。前两项使用 stage 0 和步长 2，小试使用 stage 1 和步长 18。worklist 文件随代码进入 Git 历史后才会被自动调度。

在 CLI 的 Train Schedule Model 选择以上两个正式 schedule，依次运行。两者均选 stage 0、`short_test=False`、`resume=False`、`sparse`、步长 2。服务端必须存在 Graph 已经建立的 `repaired_20261117` pin。对照复用原 19 个训练点、`std_lag1_10`、种子 17、恒定学习率 0.001、最多 30 epoch 和验证 Rank IC 早停耐心 5。普通训练接口按模型归档和投递，启用 DetailedAlphaAnalysis 的 `factor` 和 `t50`。

学习率小试选择 `factorseq_v4_graph_lr_probe`，设置 **stage 1（只训练）、sparse、步长 18**，首次 `resume=False`，日期使用默认完整范围。检查器要求实际命中 20161230、20210827、20260512 三期。该模型结构和停机规则与 Graph 相同，基础学习率为 0.0003。研究结论以这三折的验证表现为依据。

## 从服务器快照构建报告

以下命令使用服务端实际模型目录。报告输出到 `results/custom/factor_sequence` 下的独立运行目录。加入 `--publish` 可经既有投递系统发布报告与明细 CSV。

```bash
.venv/bin/python -m src.res.model.analytic.factor_sequence.compare_v5 market \
  --graph /path/to/graph --market /path/to/v5-market --events /path/to/v5-events

.venv/bin/python -m src.res.model.analytic.factor_sequence.compare_v5 learning-rate \
  --graph /path/to/graph --probe /path/to/graph-lr-probe
```

样本外对照校验同一冻结日历、股票集合和逐股训练标签。原始收益 ΔRank IC 来自 BasicTest 的逐日记录，Top 50 组合业绩来自 DetailedAlphaAnalysis。同行为诊断使用同权重下的 Graph 基础分数、市场因子修正、股票修正和事件修正。事件修正计入合并门内的交互。换入与换出明细单独报告训练标签的均值，作为排序诊断。三折小试使用 epoch 指标表、训练批次表和日志。
