# 因子时序模型 v1

专用实现位于 `src/data/preprocess/factor_sequence` 和
`src/res/algo/nn/model/factor_sequence`。模型目录见 `models.json`，冻结数据定义见
`data.json`，实际模型参数见 `configs/algo/nn/factorseq_*.yaml`。
M3（历史情境注意力）、M4（快慢分支）、M5（股票图）仅登记，不可运行。

## 执行

在项目根目录使用项目 Python 环境。入口复用现有 ModelTrainer，不运行全市场调度、
不发送邮件。真实训练沿用现有训练器的设备选择；离线自动测试使用 CPU。

```bash
.venv/bin/python -m src.data.preprocess.factor_sequence.pipeline build --run-id fs_v1_001 --data-config configs/experience/model/factor_sequence/data.json
.venv/bin/python -m src.data.preprocess.factor_sequence.pipeline audit --run-id fs_v1_001
.venv/bin/python -m src.data.preprocess.factor_sequence.pipeline smoke --run-id fs_v1_001
.venv/bin/python -m src.data.preprocess.factor_sequence.pipeline compare --run-id fs_v1_001
```

也可使用 `all` 顺序执行。每阶段只运行一次，已有状态文件禁止覆盖。
重试必须使用新 run ID，并用 `--dataset /绝对路径/数据版本` 复用已完成的数据。
新 run ID 的 compare 需要先完成该 run 的 smoke，不会拼接其他实验的输出。
`--start/--end` 显式选择构建范围；默认共同因子截止日前回看 720 个交易日。
不会自动缩小股票池、减少因子或缩短必需样本。

研究模式：在 smoke 阶段增加 `--research`（三个种子 17/29/43，最多 30 epoch，
验证 Rank IC 连续 5 epoch 未改善早停）；`--ablations` 增加 M1 去 P、M1 去 M、
M2 关闭 FiLM。默认五个模型各 3 epoch、种子 17，结果只验证流程。
共同标签 `std_lag1_10`、MSE、日频等权、无相关附加损失、无 global2top。

## 数据合同

- `factorseq` 为 `[股票,日期,1,10×因子数]`，通道按类别排列：value、valid、observed、
  gap、never、change、change_valid、source_age、source_age_valid、degenerate。
  manifest 保存完整顺序。方向→合法股票池→非有限值缺失→1%/99% 缩尾→总体 z-score；
  少于 20 个观测不可用，常量截面为零并保留退化标记。最后填零。
- gap 是距上一个有效**文件观测**的交易日间隔，change 是与该观测的差值再做截面标准化。
  两者均不代表真实披露更新；首次观测无 change。source_age 及其 valid 首版始终为零。
  无覆盖和对齐空位的 never 为 1；已有历史但当前缺失不会丢失 gap。
- `factorperf`、`factormarket` 为 `[1,日期,1,特征]`，每个数值紧跟 `__valid`。
  存储及单日期 batch 都保持一个公共序列，禁止复制全股票轴。
- 表现使用前日因子与当日已实现收益、同一有效集合等权超额、至少 50 只股票。
  排名划分十组；并列值不按证券代码拆分，缺少极端组时对应收益无效。
  IC 原值；收益与其均值/波动乘 100；ICIR 不年化并裁剪 ±5。
  20/60 日摘要至少 10/30 个观测，风险尺度只用此前 120 日、至少 60 个有效值的 RMS。
- 事件区间固定开始前的因子截面，区间内每日收益必须完整，确认日才可见；
  同类型新事件无法评估时旧事件表现失效，不悄悄延续旧记录。
  不同事件同日独立保存。无近期事件不等于数据缺失。
- 市场连续指标只用此前 120 日统计（至少 60 个有效值）；收益保留百分点、
  比例保留 [0,1]、标签保留 0/1。间隔统一 log1p 截至 252 日。
- 上市满 91 个交易日至退市前作为当日合法研究股票池，标准化不依赖标签覆盖。
  读取原始每日文件，不做通用 ffill/autofill/divlast/channelnorm/histnorm。
  源文件原本已经填充或修订的信息无法恢复，不据此宣称完整 PIT。

## 版本与审计

数据保存于 `PATH.datablock/factor_sequence/<内容版本>`。manifest 包含配置、源快照哈希、
转换代码哈希、因子方向、股票/日期快照、特征顺序与公共标记；加载校验哈希。
`--parent` 验证历史前缀；修订必须显式 `--allow-revision` 并生成新版本。
源码变更同样产生新版本。中断留下 incomplete 标记，不复用半成品。

分块参数 `build(..., chunk_size=...)` 是因果前缀重放参考实现：携带完整历史，
用于验证分块/增量一致性，**不是内存受限的流式构建器**。当前股票块全量构建；
完整研究池需要充足内存，首版尚未优化该部分的峰值内存。

audit 输出覆盖率 CSV、原始/标准化分位点、退化比例、公共指标成熟度、事件确认记录，
并独立复算选定日期的 IC、分组收益及事件区间收益。没有合格对照的指标保持未核对，
不会伪造核对记录。缺少必需数据或块为空时构建/审计失败。

## 实验记录与评价

结果在 `PATH.result/factor_sequence/<run-id>`；阶段状态包含调用参数和失败堆栈。
training 目录保存冻结切分、标签、真实未来收益、模型目录、每项状态与实际配置、
源码/数据哈希、模型路径、用时、进程峰值 RSS（累计进程指标，不是独立模型增量内存）。

切分从成熟标签末日向前取 252/63/63 个交易日，再按 lag1+10 的真实交易日终点清除
跨边界样本；另要求输入/统计预热。训练入口再次校验显式日期及标签边界，
缺失日期或过小 batch 直接失败。

逐股逐日预测 CSV 保留标签、有效性和版本；compare 要求全部实验成功且样本集合一致。
每日 Rank IC/非年化 ICIR 与头部十组的真实 10 日超额收益分开报告，
绝不把重叠标签累计为日净值。未产出持仓与成交序列时成本后收益为 null。

## 验证

```bash
MPLCONFIGDIR=/tmp/learndl-factorseq-mpl .venv/bin/python -m unittest discover -s tests -p test_factor_sequence.py -v
```

人工数据仅用于离线测试，不用于宣称真实 build/audit/smoke/compare 通过。
首次执行的实测状态及环境限制记录于同目录 `validation.md`。
